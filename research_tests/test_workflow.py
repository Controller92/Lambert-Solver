"""Scientific workflow contracts; optional explicit CUDA integration."""
from dataclasses import replace
from datetime import datetime
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

from src.research.config import GridConfig,from_dict,load_config,save_config
from src.research.grid import compute_grid,policies
from src.lambert_geometry import LambertInput
from src.lambert_solver import solve_zero_rev
from src.lambert_multirev_solver import solve_multi_rev
from src.lambert_native import NativeEngine


class ConfigTests(unittest.TestCase):
    def test_closed_fractional_grid_and_legacy_roundtrip(self):
        c=GridConfig(dep_start='2035-01-01',dep_end='2035-01-02',dep_step=.3,
                     tof_min=10.5,tof_max=11.5,tof_step=.25)
        self.assertEqual(c.validate().shape,(4,5))
        legacy=dict(simulation_type='two_body_grid',parameters=dict(dep_id='399',arr_id='499',
                    dep_start='2035-01-01',dep_end='2035-01-02',dep_step='0.5',
                    tof_min='10.5',tof_max='11.5',tof_step='0.25',dep_body_name='Earth',arr_body_name='Mars'))
        c=from_dict(legacy)
        self.assertEqual(c.shape,(3,5));self.assertEqual(c.tof_min,10.5)
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'config.json';save_config(c,path)
            self.assertEqual(load_config(path).to_dict(),c.to_dict())

    def test_reject_invalid_or_misleading_specs(self):
        for fields in (dict(dep_step=0),dict(tof_min=0),dict(tof_step=math.nan),dict(dep_end='2000-01-01'),
                       dict(time_scale='UTC'),dict(frame='ITRF93'),dict(max_revolutions=True),
                       dict(backend='auto'),dict(orientation='low'),dict(plot=1),dict(max_cells=1),
                       dict(dep_start='2035-01-01T00:00:00+00:00'),dict(dep_step=-1),dict(max_memory_mb=0)):
            with self.subTest(fields=fields),self.assertRaises(ValueError):GridConfig(**fields).validate()
        with self.assertRaises(ValueError):from_dict(dict(typo=1))
        with self.assertRaises(ValueError):from_dict(dict(kernels=[{}]))
        with self.assertRaises(ValueError):from_dict(dict(kernels=['same.bsp','same.bsp']))
        with self.assertRaises(ValueError):from_dict(dict(simulation_type='two_body_grid'))
        with self.assertRaises(ValueError):from_dict(dict(simulation_type='chain3',parameters={}))

    def test_cli_help_menu_exit_and_dry_run_without_ephemerides(self):
        for argv,stdin in ((['--help'],None),([], '6\n'),(['transfer-grid','--dry-run'],None),(['batch','--config','configs/smoke-earth-mars.json','--dry-run'],None)):
            result=subprocess.run([sys.executable,'run.py',*argv],input=stdin,text=True,capture_output=True)
            self.assertEqual(result.returncode,0,result.stderr)
        result=subprocess.run([sys.executable,'run.py','transfer-grid','--dep-step','0','--dry-run'],capture_output=True,text=True)
        self.assertNotEqual(result.returncode,0)


class GridTests(unittest.TestCase):
    backend='cpu'

    def circular(self,config):
        # Distinct flight-time and departure axes catch transposition/indexing.
        dep=np.array((0.,.3,.6));tof=np.array((.25,1.25,2.25))*2*math.pi/86400.
        ds=np.array([[math.cos(t),math.sin(t),0.,-math.sin(t),math.cos(t),0.] for t in dep])
        at,inv=np.unique(dep[:,None]+tof[None,:]*86400.,return_inverse=True)
        # Arrival body on a different circular orbit, in the same inertial frame.
        ar=1.4;motion=ar**-1.5
        ast=np.array([[ar*math.cos(motion*t+.4),ar*math.sin(motion*t+.4),0.,
                       -math.sin(motion*t+.4)/math.sqrt(ar),math.cos(motion*t+.4)/math.sqrt(ar),0.] for t in at])
        out,meta=compute_grid(config,dep,tof,ds,ast,inv.reshape(3,3),1.)
        return dep,tof,ds,ast,inv.reshape(3,3),out,meta

    def test_native_reduction_matches_reference_all_candidates(self):
        for orientation in ('short','long','prograde','retrograde','both'):
            c=GridConfig(backend=self.backend,orientation=orientation,max_revolutions=2,batch_size=4,plot=False)
            dep,tof,ds,ast,inv,out,meta=self.circular(c)
            expected=np.full((3,3),np.inf)
            for index,(m,b,w) in enumerate(policies(c)):
                for i in range(3):
                    for j in range(3):
                        r1,r2=ds[i,:3],ast[inv[i,j],:3]
                        hz=np.cross(r1,r2)[2]
                        long=w=='long' if w in ('short','long') else hz<0 if w=='prograde' else hz>0
                        p=LambertInput(tuple(r1),tuple(r2),tof[j]*86400.,1.,bool(long),m)
                        result=solve_zero_rev(p) if not m else solve_multi_rev(p,'left' if b==0 else 'right')
                        self.assertEqual(out['candidate_status'][index,i,j],int(result.status))
                        if result.status==0:
                            c3=np.linalg.norm(np.array(result.v1_m_s)-ds[i,3:])**2/1e6
                            expected[i,j]=min(expected[i,j],c3)
            np.testing.assert_allclose(out['C3'],expected,rtol=2e-11,atol=1e-16)
            np.testing.assert_allclose(np.einsum('ijk,ijk->ij',out['vinf_dep'],out['vinf_dep'])/1e6,out['C3'],rtol=1e-15)
            self.assertEqual(meta['candidate_solves'],9*len(policies(c)))
            self.assertEqual(out['C3'].shape,(3,3))
            self.assertTrue(np.all(out['revolutions']>=0))

    def test_failures_do_not_create_zero_energy_minima(self):
        c=GridConfig(backend=self.backend,orientation='both',max_revolutions=1,batch_size=2,plot=False)
        dep=np.array([0.,1.]);tof=np.array([1.,2.])
        ds=np.array([[1.,0.,0.,0.,1.,0.]]*2)
        # Collinear inputs fail every candidate and must stay NaN, not C3=0.
        ast=np.array([[-1.,0.,0.,0.,-1.,0.]])
        out,_=compute_grid(c,dep,tof,ds,ast,np.zeros((2,2),dtype=int),1.)
        self.assertTrue(np.isnan(out['C3']).all())
        self.assertTrue(np.isnan(out['vinf_dep']).all())
        self.assertTrue(np.all(out['revolutions']==-1))
        self.assertTrue(np.all(out['candidate_status']!=0))

    def test_postprocessing_overflow_retains_explicit_failure(self):
        c=GridConfig(backend=self.backend,orientation='short',plot=False)
        ds=np.array([[1.,0.,0.,1e300,0.,0.]])
        ast=np.array([[0.,1.,0.,0.,0.,0.]])
        out,_=compute_grid(c,np.array([0.]),np.array([math.pi/2/86400.]),ds,ast,np.zeros((1,1),dtype=int),1.)
        self.assertEqual(out['candidate_status'][0,0,0],5)
        self.assertTrue(np.isnan(out['C3'][0,0]))

    def test_fixed_axis_orientation_is_not_short_long(self):
        c=GridConfig(backend=self.backend,orientation='prograde',batch_size=1,plot=False)
        dep=np.array([0.]);tof=np.array([1.])
        ds=np.array([[1.,0.,0.,0.,0.,0.]])
        ast=np.array([[0.,0.,1.,0.,0.,0.]]) # vertical orbit plane: hz=0
        out,_=compute_grid(c,dep,tof,ds,ast,np.zeros((1,1),dtype=int),1.)
        self.assertEqual(out['candidate_status'][0,0,0],4)
        self.assertTrue(np.isnan(out['C3'][0,0]))


@unittest.skipUnless(os.environ.get('LAMBERT_TEST_CUDA')=='1','explicit GPU opt-in required')
class CudaGridTests(GridTests):
    backend='cuda'


class PlotTests(unittest.TestCase):
    def test_saved_plot_regeneration_and_hash_validation(self):
        from src.research.plot import plot_grid
        from src.research.ephemeris import sha256
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'grid.npz'
            np.savez(path,dep_et=np.array([0.,86400.]),tof_days=np.array([10.,20.]),
                     C3=np.array([[20.,30.],[np.nan,40.]]))
            metadata=dict(schema=1,config=GridConfig().to_dict(),grid_sha256=sha256(path))
            path.with_name('run.json').write_text(json.dumps(metadata))
            output=plot_grid(path)
            self.assertGreater(output.stat().st_size,1000)
            with self.assertRaises(ValueError):plot_grid(path,path)
            with path.open('ab') as stream:stream.write(b'changed')
            with self.assertRaises(ValueError):plot_grid(path)

@unittest.skipUnless(os.environ.get('LAMBERT_TEST_KERNEL_DIR'),'explicit SPICE kernel directory required')
class EphemerisTests(unittest.TestCase):
    def test_real_cache_integrity_and_owned_kernel_lifetime(self):
        import spiceypy as sp
        from src.research.ephemeris import SpiceEphemeris
        kernels=[str((Path(os.environ['LAMBERT_TEST_KERNEL_DIR'])/name).resolve())
                 for name in ('naif0012.tls','de440s.bsp','gm_de440.tpc','20000001.bsp')]
        before=sp.ktotal('ALL')
        with tempfile.TemporaryDirectory() as d:
            with SpiceEphemeris(kernels,'ECLIPJ2000',d) as e:
                et=np.array([e.epoch('2035-06-24'),e.epoch('2035-06-25')])
                states=e.states('399',et)
                self.assertEqual(e.calls,2)
                self.assertGreater(e.mu,1e20);self.assertLess(e.mu,2e20)
                self.assertGreater(np.linalg.norm(states[0,3:]),2e4)
                self.assertLess(np.linalg.norm(states[0,3:]),4e4)
                np.testing.assert_array_equal(e.states('399',et),states)
                self.assertEqual(e.cache_hits,2)
                self.assertGreater(np.linalg.norm(states[0,:3]),1e11) # SI, not km
                path=next(Path(d).glob('*.npz'))
                with np.load(path,allow_pickle=False) as saved:
                    values=dict(saved)
                values['states']=values['states'].copy();values['states'][0,0]+=1.
                np.savez(path,**values)
                with self.assertRaises(ValueError):e.states('399',et)
            self.assertEqual(sp.ktotal('ALL'),before)
            # Full configured coverage is checked via actual state retrieval.
            with SpiceEphemeris(kernels,'ECLIPJ2000',d) as e:
                with self.assertRaises(ValueError):e.states('20000001',np.array([e.epoch('2050-01-01')]))
        self.assertEqual(sp.ktotal('ALL'),before)
        sp.furnsh(kernels[0])
        try:
            with self.assertRaises(ValueError):SpiceEphemeris(kernels,'ECLIPJ2000',d)
            self.assertEqual(sp.ktotal('ALL'),before+1)
        finally:sp.unload(kernels[0])


if __name__=='__main__':unittest.main()
