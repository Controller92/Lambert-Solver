"""Physics invariants and independent two-body conic reconstruction."""
from dataclasses import replace
import math
import unittest
import numpy as np
from scipy.integrate import solve_ivp
from src.flyby import Encounter,Mode,Status,solve

EARTH=3.9860043543609598e14
BASE=Encounter((10000.,0.,0.),EARTH,6.6e6,1e11,rp=7e6,azimuth=.4)


def asymptotes(r,v,mu):
    """Independently recover asymptotes from Cartesian orbital elements."""
    r=np.asarray(r);v=np.asarray(v)
    h=np.cross(r,v); H=np.linalg.norm(h); R=np.linalg.norm(r)
    energy=np.dot(v,v)/2-mu/R
    if energy<=0: raise ValueError('Not hyperbolic')
    ev=np.cross(v,h)/mu-r/R; e=np.linalg.norm(ev)
    p=ev/e; q=np.cross(h/H,p)
    speed=math.sqrt(2*energy)
    inbound=speed*(p/e+math.sqrt(e*e-1)/e*q)
    outbound=speed*(-p/e+math.sqrt(e*e-1)/e*q)
    return inbound,outbound,H*H/(mu*(1+e))


class Physics(unittest.TestCase):
    def assertSuccess(self,r): self.assertEqual(r.status,Status.SUCCESS)

    def test_energy_angular_momentum_and_conic_asymptotes(self):
        for theta in (0.,.3,math.pi,1e6):
            p=replace(BASE,azimuth=theta);r=solve(p);self.assertSuccess(r)
            np.testing.assert_allclose(np.linalg.norm(r.vout),10000.,rtol=5e-16)
            np.testing.assert_allclose(np.dot(r.rperi,r.vperi_in),0.,atol=1e-4)
            np.testing.assert_allclose(np.cross(r.b,p.vin),np.cross(r.rperi,r.vperi_in),rtol=2e-15,atol=1e-5)
            incoming,outgoing,rp=asymptotes(r.rperi,r.vperi_in,p.mu)
            np.testing.assert_allclose(incoming,p.vin,rtol=2e-15,atol=1e-10)
            np.testing.assert_allclose(outgoing,r.vout,rtol=2e-15,atol=1e-10)
            self.assertAlmostEqual(rp/r.rp,1.,places=14)

    def test_inverse_and_tangential_powered_recover_periapsis(self):
        original=solve(BASE)
        for scale in (.5,1.,1.8):
            # Construct required outbound asymptote independently from a
            # Cartesian post-burn periapsis state, rather than the turn formula.
            rp=np.asarray(original.rperi); direction=np.asarray(original.vperi_in)/np.linalg.norm(original.vperi_in)
            target_speed=10000*scale
            vpost=direction*math.sqrt(target_speed**2+2*BASE.mu/BASE.rp)
            _,target,_=asymptotes(rp,vpost,BASE.mu)
            p=replace(BASE,mode=Mode.POWERED,vout=tuple(target))
            r=solve(p);self.assertSuccess(r)
            self.assertLess(abs(r.rp/BASE.rp-1),5e-13)
            np.testing.assert_allclose(r.vout,target,rtol=2e-12,atol=1e-8)
            self.assertLess(abs(r.angle_residual),2e-13)
            self.assertLessEqual(r.lower,r.upper)
            self.assertAlmostEqual(r.delta_v,np.linalg.norm(vpost)-np.linalg.norm(original.vperi_in) if scale>=1 else np.linalg.norm(original.vperi_in)-np.linalg.norm(vpost),places=9)
        match=solve(replace(BASE,mode=Mode.MATCH,vout=original.vout))
        self.assertSuccess(match);self.assertLess(abs(match.rp/BASE.rp-1),5e-15)
        self.assertEqual(match.delta_v,0.)

    def test_rotation_units_and_time_reversal(self):
        r=solve(BASE)
        q,_=np.linalg.qr(np.array([[.2,.4,.1],[.3,.1,.7],[.8,.2,.1]]))
        rotated=solve(replace(BASE,vin=tuple(q@BASE.vin),reference=tuple(q@BASE.reference)))
        np.testing.assert_allclose(rotated.vout,q@r.vout,atol=1e-10)
        scaled=solve(replace(BASE,vin=tuple(np.array(BASE.vin)*.001),mu=BASE.mu*1e-9,
                            rp_min=BASE.rp_min*.001,rp_max=BASE.rp_max*.001,rp=BASE.rp*.001))
        np.testing.assert_allclose(np.array(scaled.vout)*1000,r.vout,atol=1e-10)
        reversed=solve(replace(BASE,mode=Mode.MATCH,vin=tuple(-np.array(r.vout)),vout=tuple(-np.array(BASE.vin))))
        self.assertSuccess(reversed);self.assertLess(abs(reversed.rp/BASE.rp-1),5e-15)
        np.testing.assert_allclose(reversed.rperi,r.rperi,atol=1e-8)

    def test_actual_numerical_two_body_integration_through_encounter(self):
        r=solve(BASE)
        def ode(t,state):
            pos=state[:3];return np.r_[state[3:],-BASE.mu*pos/np.linalg.norm(pos)**3]
        # Propagate backwards from periapsis, then independently forwards all
        # the way across the encounter. Endpoint prediction uses the ODE only.
        initial=np.r_[r.rperi,r.vperi_in]
        back=solve_ivp(ode,(0,-7200),initial,method='DOP853',rtol=3e-13,atol=1e-7)
        across=solve_ivp(ode,(-7200,7200),back.y[:,-1],method='DOP853',rtol=3e-13,atol=1e-7)
        self.assertTrue(back.success and across.success)
        vi,_,_=asymptotes(back.y[:3,-1],back.y[3:,-1],BASE.mu)
        _,vo,_=asymptotes(across.y[:3,-1],across.y[3:,-1],BASE.mu)
        np.testing.assert_allclose(vi,BASE.vin,rtol=1e-10,atol=1e-6)
        np.testing.assert_allclose(vo,r.vout,rtol=1e-10,atol=1e-6)

    def test_declared_speed_tolerance_preserves_incoming_energy(self):
        target=np.asarray(solve(BASE).vout)*(1+5e-11)
        p=replace(BASE,mode=Mode.MATCH,vout=tuple(target))
        r=solve(p);self.assertSuccess(r)
        self.assertGreater(r.speed_error,0)
        self.assertAlmostEqual(np.linalg.norm(r.vout),10000.,places=10)
        self.assertEqual(solve(replace(p,speed_atol=0,speed_rtol=0)).status,Status.SPEED_MISMATCH)

    def test_inclusive_bounds_and_extreme_turns(self):
        for u in (1e-24,1e-16,1.,1e16,1e24):
            rp=BASE.mu/1e8*u
            p=replace(BASE,rp=rp,rp_min=rp,rp_max=rp)
            out=solve(p)
            self.assertSuccess(out)
            for mode in (Mode.MATCH,Mode.POWERED):
                r=solve(replace(p,mode=mode,vout=out.vout))
                self.assertSuccess(r)
                self.assertLess(abs(r.rp/rp-1),2e-14)
                if u<1e-16 or u>1e16:self.assertEqual(r.precision_limited,1)

    def test_small_dimensionless_parameter_without_intermediate_overflow(self):
        p=Encounter((1.,0.,0.),1.,5e-301,2e-300,rp=1e-300)
        r=solve(p);self.assertSuccess(r)
        np.testing.assert_allclose(r.bmag,math.sqrt(2.)*1e-150,rtol=1e-15,atol=0)
        np.testing.assert_allclose(r.vout,[-1.,0.,-2*math.sqrt(2.)*1e-150],rtol=2e-15,atol=0)
        for mode in (Mode.MATCH,Mode.POWERED):
            out=solve(replace(p,mode=mode,vout=r.vout));self.assertSuccess(out)
            self.assertLess(abs(out.rp/p.rp-1),3e-13)

    def test_failures_never_publish_velocities(self):
        cases=[(replace(BASE,mu=-1),Status.INVALID_INPUT),
               (replace(BASE,vin=(0.,0.,0.)),Status.ZERO_SPEED),
               (replace(BASE,reference=(1.,0.,0.)),Status.BAD_BASIS),
               (replace(BASE,mode=Mode.MATCH,vout=(11000.,0.,0.)),Status.SPEED_MISMATCH),
               (replace(BASE,mode=Mode.MATCH,vout=BASE.vin),Status.DEGENERATE_TURN),
               (replace(BASE,mode=Mode.MATCH,vout=(-10000.,0.,0.)),Status.TURN_OUT_OF_RANGE),
               (replace(BASE,rp=BASE.rp_min-1),Status.TURN_OUT_OF_RANGE),
               (replace(BASE,rp=1e308,rp_max=1e308,mu=1e-300),Status.NUMERICAL_RANGE),
               (replace(BASE,vin=(math.nan,0.,0.)),Status.INVALID_INPUT),
               (replace(BASE,mode=Mode.POWERED,vout=(-12000.,1.,0.)),Status.TURN_OUT_OF_RANGE)]
        for p,status in cases:
            with self.subTest(p=p):
                r=solve(p);self.assertEqual(r.status,status);self.assertTrue(np.isnan(r.vout).all())


if __name__=='__main__': unittest.main()
