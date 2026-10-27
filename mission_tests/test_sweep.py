"""Independent geometry/metric contracts for the EMC survey."""
import unittest
import numpy as np
from src.research.mission_sweep import angle,encounter_metrics,sample_arc
from src.lambert_native import NativeEngine,pack_problems
from src.lambert_geometry import LambertInput


class SweepContracts(unittest.TestCase):
    def test_speed_gain_and_plane_rotation_are_different_quantities(self):
        result=encounter_metrics(np.array((1.,0,0)),np.array((0,1.,0)),np.array((0,1.,0)),
            np.array((0,0,1.)),np.array((0.,-1.,1.)))
        self.assertAlmostEqual(result['plane_change_deg'],45.)
        self.assertAlmostEqual(result['vinf_turn_deg'],90.)
        self.assertEqual(result['vinf_speed_gain_km_s'],0.)
        self.assertAlmostEqual(result['heliocentric_speed_gain_km_s'],(np.sqrt(2)-2)/1000)
        self.assertAlmostEqual(result['ceres_plane_after_deg'],0.)
        self.assertAlmostEqual(result['energy_gain_km2_s2'],-1e-6)

    def test_all_conic_loci_preserve_endpoint_and_plane(self):
        engine=NativeEngine()
        for m in (0,2):
            for long_way in (False,True):
                for branch in ('left','right'):
                    p=LambertInput((1.,.2,-.1),(-.3,1.1,.4),.2 if m==0 else 17.,1.,long_way,m)
                    r=engine.cpu(pack_problems([p],branch))[0];self.assertEqual(r['status'],0)
                    points=sample_arc(np.array([p.r1_m]),r['v1'][None,:],np.array([p.r2_m]),np.array([m]),1.,129)[0]
                    np.testing.assert_allclose(points[0],p.r1_m,rtol=0,atol=2e-14)
                    np.testing.assert_allclose(points[-1],p.r2_m,rtol=0,atol=3e-12)
                    h=np.cross(p.r1_m,r['v1']);h/=np.linalg.norm(h)
                    np.testing.assert_allclose(points@h,0.,atol=1e-13)

    def test_circular_multirevolution_track_winding(self):
        r=np.array([[1.,0,0]]);v=np.array([[0.,1.,0]]);end=np.array([[0.,1.,0]])
        curve=sample_arc(r,v,end,np.array([2]),1.,257)[0]
        phase=np.unwrap(np.arctan2(curve[:,1],curve[:,0]))
        self.assertAlmostEqual(phase[-1],4.5*np.pi)
        np.testing.assert_allclose(np.linalg.norm(curve,axis=1),1.,atol=3e-16)


if __name__=='__main__':unittest.main()
