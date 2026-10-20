"""Complete-chain physics, branch selection and persistence contracts."""
from copy import deepcopy
import itertools
import json
import os
from pathlib import Path
import tempfile
import unittest
import numpy as np
from src.research.mission_config import load_mission
from src.research.mission_dynamics import dynamic_path, MissionEvaluator
from src.research.mission_local import surface_pair, parking
from src.research.mission_validation import asymptote, validate_candidate
from src.research.mission_search import run_mission_search, fingerprint, write_json

ROOT = Path(__file__).resolve().parents[1]
X = [586.164156882561,385.13631335057016,1455.0581539038058,1225.2101423075399,141.2178508491192,30.188743617606075]


class Contracts(unittest.TestCase):
    def test_pairwise_branch_dp_matches_exhaustive_chains(self):
        rng = np.random.default_rng(9587)
        start = rng.uniform(0,10,(5,3)); end = rng.uniform(0,10,(5,3))
        links = [rng.uniform(0,100,(5,3,3)) for _ in range(3)]
        links[1][0,0,:] = np.inf
        score,path = dynamic_path(start,links,end)
        for n in range(5):
            brute = []
            for p in itertools.product(range(3),repeat=4):
                cost = start[n,p[0]]+end[n,p[-1]]+sum(link[n,p[j],p[j+1]] for j,link in enumerate(links))
                brute.append((cost,p))
            best = min(brute)
            self.assertAlmostEqual(score[n],best[0]);self.assertEqual(tuple(path[n]),best[1])
        # The lowest launch cost can be the wrong complete mission.
        score,path = dynamic_path(np.array([[0.,5.]]),[np.array([[[100.,100.],[1.,1.]]])],np.zeros((1,2)))
        self.assertEqual(path[0,0],1);self.assertEqual(score[0],6)
        score,_ = dynamic_path(np.zeros((1,2)),[np.full((1,2,2),np.inf)],np.zeros((1,2)))
        self.assertTrue(np.isinf(score[0]))

    def test_strict_config(self):
        original = load_mission(ROOT/'examples/missions/emc-ten-year.json')
        for name,value in [('max_revolutions',True),('model','auto'),('time_scale','UTC'),
                           ('max_duration_days',1),('seeds',[1,1]),('speed_atol_m_s',0),('kernels',[]),('popsize',1)]:
            c = deepcopy(original);setattr(c,name,value)
            with self.subTest(name=name),self.assertRaises(ValueError):c.validate()
        c = deepcopy(original);c.visits[1]['body']='4'
        with self.assertRaises(ValueError):c.validate()
        c = deepcopy(original);c.visits[2]['stay_days']=[30,-1]
        with self.assertRaises(ValueError):c.validate()

    def test_rotating_surface_conics_and_signed_burns(self):
        # Construct actual local hyperbolas at one fixed rotating site, recover
        # their asymptotes independently, and ask the planner to connect them.
        mu=6.2629e10;R=482100.;site=np.array([.6,.3,np.sqrt(.55)])*R;omega=1.9e-4
        def transform(angle):
            c,s=np.cos(angle),np.sin(angle);F=np.array([[c,-s,0],[s,c,0],[0,0,1.]])
            W=np.array([[0,-omega,0],[omega,0,0],[0,0,0.]])
            M=np.zeros((6,6));M[:3,:3]=M[3:,3:]=F;M[3:,:3]=W@F;return M
        fi,fo=transform(.2),transform(1.3)
        ri,ro=fi[:3,:3]@site,fo[:3,:3]@site
        ti=np.cross(ri,[1.,0.,.2]);ti/=np.linalg.norm(ti)
        to=np.cross(ro,[.3,1.,0.]);to/=np.linalg.norm(to)
        pi=ti*np.sqrt(2700**2+2*mu/R);po=to*np.sqrt(3300**2+2*mu/R)
        vi,vo=asymptote(ri,pi,mu,True),asymptote(ro,po,mu,False)
        r=surface_pair(vi[None,None,:],vo[None,None,:],mu,R,fi[None],fo[None],True)
        self.assertTrue(r['valid'][0,0,0]);get=lambda k:r[k][0,0,0]
        np.testing.assert_allclose(fi[:3,:3]@get('site_body_fixed_m'),get('rperi_in_m'),atol=1e-7)
        np.testing.assert_allclose(fo[:3,:3]@get('site_body_fixed_m'),get('rperi_out_m'),atol=1e-7)
        for incoming,suffix,target in [(True,'in',vi),(False,'out',vo)]:
            np.testing.assert_allclose(asymptote(get('rperi_'+suffix+'_m'),get('vperi_'+suffix+'_m_s'),mu,incoming),target,atol=1e-8)
        np.testing.assert_allclose(get('vperi_in_m_s')+get('landing_velocity_change_m_s'),get('ground_in_m_s'),atol=1e-10)
        np.testing.assert_allclose(get('ground_out_m_s')+get('departure_velocity_change_m_s'),get('vperi_out_m_s'),atol=1e-10)
        np.testing.assert_allclose(get('ground_in_m_s'),fi[3:,:3]@get('site_body_fixed_m'),atol=1e-10)

    def test_parking_hyperbolas_and_capture_sign(self):
        mu=3.986e14;R=6778137.;target=np.array([3000.,-4200.,1800.])
        for incoming in (True,False):
            p=parking(target,mu,R,incoming)
            np.testing.assert_allclose(asymptote(p['rperi_m'],p['hyperbolic_velocity_m_s'],mu,incoming),target,atol=1e-9)
            before=p['hyperbolic_velocity_m_s'] if incoming else p['circular_velocity_m_s']
            after=p['circular_velocity_m_s'] if incoming else p['hyperbolic_velocity_m_s']
            np.testing.assert_allclose(before+p['burn_vector_m_s'],after,atol=1e-10)


@unittest.skipUnless(os.environ.get('LAMBERT_TEST_SPICE')=='1','explicit local-kernel opt-in')
class Integration(unittest.TestCase):
    def config(self):
        c=load_mission(ROOT/'examples/missions/emc-ten-year.json');c.backend='cpu';return c

    def test_shared_ephemeris_ownership_and_export_acceptance(self):
        import spiceypy as sp
        c=self.config()
        with tempfile.TemporaryDirectory() as d:
            d=Path(d)
            with MissionEvaluator(c,d) as outer:
                count=sp.ktotal('ALL')
                fixed=[dict(orientation='short',revolutions=0,branch=0)]*4
                with MissionEvaluator(c,d,fixed,ephemeris=outer.e) as inner:
                    self.assertIs(outer.e,inner.e)
                self.assertEqual(sp.ktotal('ALL'),count)
                candidate=outer.candidate(X);self.assertIsNotNone(candidate)
                candidate['fingerprint']=fingerprint(c,outer)
                write_json(d/'best.json',candidate);write_json(d/'config.json',c.to_dict())
            self.assertEqual(sp.ktotal('ALL'),0)
            result=validate_candidate(d/'best.json');self.assertEqual(result['status'],'passed_patched_conic')
            candidate['visits'][2]['site_body_fixed_m'][0]+=100.
            write_json(d/'bad.json',candidate)
            result=validate_candidate(d/'bad.json');self.assertEqual(result['status'],'failed')

    def test_checkpoint_completed_resume_and_reject_changed_config(self):
        c=self.config();c.generations=1;c.popsize=5;c.seeds=[9587];c.refine_starts=1;c.refine_iterations=1
        c.max_revolutions=0
        with tempfile.TemporaryDirectory() as d:
            run,summary=run_mission_search(c,d)
            self.assertTrue((run/'population.npz').exists())
            _,resumed=run_mission_search(c,d,resume=run);self.assertEqual(summary,resumed)
            # Reconstruct the checkpoint at the first callback, as if the
            # process stopped before recording seed/refinement completion.
            checkpoint=json.loads((run/'checkpoint.json').read_text())
            checkpoint.update(seed_index=0,generation=1,completed_seeds=[],refinement_done=False)
            write_json(run/'checkpoint.json',checkpoint)
            _,continued=run_mission_search(c,d,resume=run)
            self.assertEqual(len(continued['completed_seeds']),1)
            checkpoint.update(seed_index=0,generation=1,completed_seeds=[],refinement_done=False,population_sha256='bad')
            write_json(run/'checkpoint.json',checkpoint)
            with self.assertRaises(ValueError):run_mission_search(c,d,resume=run)
            changed=deepcopy(c);changed.generations+=1
            with self.assertRaises(ValueError):run_mission_search(changed,d,resume=run)
            fp=json.loads((run/'fingerprint.json').read_text());fp['modules']['src/research/mission_local.py']='changed'
            write_json(run/'fingerprint.json',fp)
            with self.assertRaises(ValueError):run_mission_search(c,d,resume=run)

    @unittest.skipUnless(os.environ.get('LAMBERT_TEST_CUDA')=='1','explicit GPU opt-in')
    def test_cpu_cuda_complete_branch_and_pair_cost_parity(self):
        c=self.config();x=np.array([X,X]);x[1,-1]+=.03
        with tempfile.TemporaryDirectory() as d:
            with MissionEvaluator(c,d) as cpu:
                a=cpu.block(x);g=deepcopy(c);g.backend='cuda'
                with MissionEvaluator(g,d,ephemeris=cpu.e) as gpu:b=gpu.block(x)
            np.testing.assert_array_equal(a['legs']['status'],b['legs']['status'])
            np.testing.assert_array_equal(a['feasible_path'],b['feasible_path'])
            np.testing.assert_allclose(a['feasible_score'],b['feasible_score'],rtol=1e-10,atol=1e-6)
            for na,nb in zip(a['nodes'],b['nodes']):
                np.testing.assert_array_equal(na['valid'],nb['valid'])
