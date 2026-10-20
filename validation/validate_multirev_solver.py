"""Independent complete M>0 validation: Lagrange roots, conics and ODE winding.

Multiprecision references rebuild the actual received geometry and invert the
Lagrange anomaly equation. Gooding is a separate algorithm. Numerical ODE
propagation checks endpoint closure AND the requested number of complete turns.
"""
import argparse
from collections import Counter
from dataclasses import replace
from functools import lru_cache
import json
import math
from pathlib import Path
import random
import sys

import mpmath as mp
import numpy as np
from lamberthub import gooding1990
from scipy.integrate import solve_ivp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.lambert_geometry import LambertInput
from src.lambert_multirev import find_minimum_multi_rev
from src.lambert_multirev_root import MultiRootStatus, RootBranch, solve_x_multi_rev_pair
from src.lambert_multirev_solver import solve_multi_rev_pair
from src.lambert_solver import LambertStatus
from validate_multirev import lagrange, minimum_reference, precision, reference_lambda
from validate_precision import norm, cross
from validate_solver import basis, conic_case


@lru_cache(None)
def cached_minimum(lam, m, q):
    with mp.workdps(max(120, precision(q))):
        return minimum_reference(lam, m, q)


def bisect_roots(target, lam, m, minimum):
    roots = []
    for branch, lo, hi in ((RootBranch.LEFT, mp.mpf(-1), minimum),
                           (RootBranch.RIGHT, minimum, mp.mpf(1))):
        for _ in range(240):
            x = (lo+hi)/2
            if (lagrange(x, lam, m) > target) == (branch == RootBranch.LEFT):
                lo = x
            else:
                hi = x
        roots.append((lo+hi)/2)
    return roots


def scalar_targets():
    for m in (1, 2, 5, 100, 2**31-1):
        for lam in (-.999, -.5, 0., .5, .999):
            for known in (-.99, -1e-5, 0., .3, .9, math.nextafter(1., 0.)):
                with mp.workdps(120):
                    yield float(lagrange(mp.mpf(known), mp.mpf(lam), m)), lam, m, None
    for q in (1e-12, 1e-16, 1e-100, math.ulp(0.)):
        for sign in (-1, 1):
            lam = sign*math.sqrt(1-q)
            for m in (1, 5):
                for known in (-.8, -1e-5, 0., .5):
                    with mp.workdps(max(120, precision(q))):
                        yield float(lagrange(mp.mpf(known), reference_lambda(lam, q), m)), lam, m, q
    for lam in (-.99, 0., .99):
        for m in (1, 2, 20):
            _, t = cached_minimum(lam, m, None)
            for gap in (-1e-8, -1e-14, 0., 1e-14, 1e-12, 1e-8, .1):
                with mp.workdps(120):
                    yield float(t*(1+mp.mpf(gap))), lam, m, None


def validate_scalar():
    counts = Counter()
    maxima = dict(root_x=0., unflagged_x=0., forward_ulps=0., error_over_estimate=0.)
    for target, lam, m, q in scalar_targets():
        pair = solve_x_multi_rev_pair(target, lam, m, q=q)
        with mp.workdps(max(120, precision(q))):
            l = reference_lambda(lam, q)
            xmin, tmin = cached_minimum(lam, m, q)
            references = bisect_roots(mp.mpf(target), l, m, xmin) if mp.mpf(target) > tmin else None
            for result, branch in zip(pair, RootBranch):
                counts[result.status.name] += 1
                if result.status == MultiRootStatus.NO_SOLUTION:
                    if not mp.mpf(target) < tmin:
                        raise AssertionError('False infeasibility')
                    continue
                if result.status == MultiRootStatus.NEAR_MINIMUM:
                    if result.x is not None:
                        raise AssertionError('Uncertain multiplicity published a root')
                    continue
                if references is None:
                    raise AssertionError('Root accepted below true minimum')
                reference = references[0 if branch == RootBranch.LEFT else 1]
                if result.status == MultiRootStatus.RESOLUTION_LIMIT:
                    from src.lambert_multirev import evaluate_tof_multi_rev
                    x = float(reference)
                    neighbors = (x, math.nextafter(x, -math.inf), math.nextafter(x, math.inf))
                    for point in neighbors:
                        if -1 < point < 1:
                            v = evaluate_tof_multi_rev(point, lam, m, q=q)
                            if abs(v.time_nondim/target-1) <= 5e-13:
                                raise AssertionError('Resolution failure despite a qualifying representable root')
                    continue
                if result.status != MultiRootStatus.OK:
                    raise AssertionError(f'Unexpected scalar root failure: {target,lam,m,q,result}')
                error = float(abs(mp.mpf(result.x)-reference))
                estimate = result.estimated_x_error or 0.
                if error > max(5e-14*(1+float(abs(reference))), estimate):
                    raise AssertionError(f'Independent root error beyond diagnostic: {target,lam,m,q,error,result}')
                if error > 5e-14*(1+float(abs(reference))) and result.x_precision_limited is not True:
                    raise AssertionError('Unflagged ill-conditioned root')
                if (result.x < result.minimum.lower) != (branch == RootBranch.LEFT):
                    raise AssertionError('Computed minimum branch interval violated')
                if abs(result.relative_residual) > 5e-13 or abs(result.last_step) > 1e-14+1e-14*abs(result.x):
                    raise AssertionError('Stopping requirements violated')
                forward_error = float(abs(mp.mpf(result.time_nondim)-lagrange(mp.mpf(result.x), l, m))
                                      /math.ulp(result.time_nondim))
                if forward_error > 16:
                    raise AssertionError('Forward roundoff allowance exceeded')
                maxima['root_x'] = max(maxima['root_x'], error)
                maxima['forward_ulps'] = max(maxima['forward_ulps'], forward_error)
                if estimate:
                    maxima['error_over_estimate'] = max(maxima['error_over_estimate'], error/estimate)
                if result.x_precision_limited is False:
                    maxima['unflagged_x'] = max(maxima['unflagged_x'], error)
                counts['precision_warning'] += result.x_precision_limited is True
                counts['precision_unknown'] += result.x_precision_limited is None
    result = dict(counts=dict(counts), maxima=maxima)
    print('Independent scalar roots:', json.dumps(result), flush=True)
    return result


def physical_reference(problem):
    """Actual-input MP geometry/roots/velocities and optional x sensitivity."""
    with mp.workdps(140):
        r1, r2 = (tuple(map(mp.mpf, r)) for r in (problem.r1_m, problem.r2_m))
        mu, tof, m = mp.mpf(problem.mu_m3_s2), mp.mpf(problem.tof_s), problem.revolutions
        n1, n2 = norm(r1), norm(r2)
        chord = norm(tuple(b-a for a, b in zip(r1, r2)))
        s = (n1+n2+chord)/2
        q = chord/s
        lam = mp.sqrt(1-q)*(-1 if problem.long_way else 1)
        target = tof*mp.sqrt(2*mu/s**3)
        xmin, tmin = minimum_reference(lam, m)
        if target <= tmin:
            return None
        roots = bisect_roots(target, lam, m, xmin)
        rho, gamma = (n1-n2)/chord, mp.sqrt(mu*s/2)
        sigma = mp.sqrt(1-rho*rho)
        h = cross(r1, r2)
        h = tuple(v/norm(h)*(-1 if problem.long_way else 1) for v in h)
        u1, u2 = tuple(v/n1 for v in r1), tuple(v/n2 for v in r2)
        t1, t2 = cross(h, u1), cross(h, u2)

        def velocities(x):
            y = mp.sqrt(q+lam*lam*x*x)
            vr1 = gamma/n1*((lam*y-x)-rho*(lam*y+x))
            vr2 = -gamma/n2*((lam*y-x)+rho*(lam*y+x))
            vt = gamma*sigma*(y+lam*x)
            v1 = tuple(vr1*u+vt/n1*t for u, t in zip(u1, t1))
            v2 = tuple(vr2*u+vt/n2*t for u, t in zip(u2, t2))
            return v1, v2
        return roots, [velocities(x) for x in roots], velocities


def kepler_propagate(r1, v1, elapsed):
    """100-digit propagation from received state by solving Kepler's equation.

    Uses orbital elements/eccentric anomaly, not Lambert's TOF or velocity
    reconstruction. Elapsed time determines the unwrapped revolution count.
    """
    with mp.workdps(100):
        r, v = tuple(map(mp.mpf, r1)), tuple(map(mp.mpf, v1))
        radius = norm(r)
        h = cross(r, v)
        normal = tuple(x/norm(h) for x in h)
        a = 1/(2/radius-sum(x*x for x in v))
        ev = tuple(x-y/radius for x, y in zip(cross(v,h),r))
        e = norm(ev)
        if not a > 0 or not e < 1:
            raise AssertionError('Expected bound elliptic propagation')
        if e == 0:
            peri = tuple(x/radius for x in r)
            transverse = cross(normal,peri)
            delta = mp.mpf(elapsed)/mp.sqrt(a**3)
            return (tuple(a*(mp.cos(delta)*x+mp.sin(delta)*y) for x,y in zip(peri,transverse)),
                    tuple((-mp.sin(delta)*x+mp.cos(delta)*y)/mp.sqrt(a) for x,y in zip(peri,transverse)),delta)
        peri = tuple(x/e for x in ev)
        transverse = cross(normal,peri)
        root = mp.sqrt(1-e*e)
        e1 = mp.atan2(sum(x*y for x,y in zip(r,v))/(e*mp.sqrt(a)), (1-radius/a)/e)
        if e1 == mp.pi:
            e1 = -mp.pi
        mean = e1-e*mp.sin(e1)+mp.mpf(elapsed)/mp.sqrt(a**3)
        turns = mp.floor((mean+mp.pi)/(2*mp.pi))
        reduced = mean-turns*2*mp.pi
        lo,hi = -mp.pi,mp.pi
        for _ in range(260):
            middle = (lo+hi)/2
            if middle-e*mp.sin(middle) < reduced:
                lo = middle
            else:
                hi = middle
        e2 = (lo+hi)/2
        co,si = mp.cos(e2),mp.sin(e2)
        out_r = tuple(a*((co-e)*x+root*si*y) for x,y in zip(peri,transverse))
        out_v = tuple((-si*x+root*co*y)/(mp.sqrt(a)*(1-e*co)) for x,y in zip(peri,transverse))
        nu1 = mp.atan2(root*mp.sin(e1),mp.cos(e1)-e)
        nu2 = mp.atan2(root*si,co-e)
        return out_r,out_v,nu2-nu1+2*mp.pi*turns


def propagate_winding(r1, v1, r2, v2, elapsed, m, angle):
    """Dimensionless DOP853 with adaptive samples, endpoint closure and winding."""
    def rhs(t, state):
        r = state[:3]
        return np.concatenate((state[3:], -r/np.linalg.norm(r)**3))
    solution = solve_ivp(rhs, (0., elapsed), np.concatenate((r1, v1)), method='DOP853',
                         rtol=3e-14, atol=3e-15, max_step=elapsed/(128*(m+1)))
    if not solution.success:
        raise AssertionError(solution.message)
    n = np.cross(r1, v1)
    n /= np.linalg.norm(n)
    u = r1/np.linalg.norm(r1)
    v = np.cross(n, u)
    phase = np.unwrap(np.arctan2(v@solution.y[:3], u@solution.y[:3]))
    if max(abs(np.diff(phase))) >= .5:
        raise AssertionError('Propagation winding undersampled')
    sweep = float(phase[-1]-phase[0])
    if abs(sweep-(2*math.pi*m+angle)) > 2e-8:
        raise AssertionError(f'Wrong revolution count or orientation: {sweep, m, angle}')
    return (float(np.linalg.norm(solution.y[:3, -1]-r2)),
            float(np.linalg.norm(solution.y[3:, -1]-v2)), sweep)


def conics():
    rng = random.Random(958710)
    scales = ((1., 1.), (7e6, 3.986004418e14), (1.495978707e11, 1.3271244004193937e20))
    with mp.workdps(80):
        for i in range(48):
            m, e = (1, 2, 5, 10)[(i//2) % 4], (0., .2, .7, .9)[(i//8) % 4]
            residual = rng.uniform(.2, math.pi-.1) if i % 2 else rng.uniform(math.pi+.1, 2*math.pi-.2)
            start = mp.mpf(rng.uniform(-math.pi, math.pi))
            stop = start+2*mp.pi*m+mp.mpf(residual)
            length, mu = scales[i % 3]
            case = conic_case(f'multi-conic-{i}', e, start, stop, length, mu, basis(rng))
            yield replace(case, problem=replace(case.problem, revolutions=m, long_way=residual > math.pi))


def validate_conics():
    maxima = dict(known_velocity=0., gooding_velocity=0., energy=0., momentum=0.,
                  propagated_position=0., propagated_velocity=0., winding_angle=0.,
                  ode_position=0., ode_velocity=0., control_position=0., control_velocity=0.)
    count = warnings = 0
    for case in conics():
        pair = solve_multi_rev_pair(case.problem)
        p, scale, problem = case.length_scale, case.speed_scale, case.problem
        r1, r2 = np.array(problem.r1_m)/p, np.array(problem.r2_m)/p
        elapsed = problem.tof_s/(p/scale)
        expected = [np.array(case.v1)/scale, np.array(case.v2)/scale]
        control_r,control_v,_ = kepler_propagate(r1,expected[0],elapsed)
        control_position = float(np.linalg.norm(np.array(list(map(float,control_r)))-r2))/max(1.,np.linalg.norm(r2))
        control_velocity = float(np.linalg.norm(np.array(list(map(float,control_v)))-expected[1]))/max(1.,*(np.linalg.norm(v) for v in expected))
        if max(control_position,control_velocity)>3e-9:
            raise AssertionError('Independent conic/Kepler control mismatch')
        maxima['control_position']=max(maxima['control_position'],control_position)
        maxima['control_velocity']=max(maxima['control_velocity'],control_velocity)
        actual = []
        for result in pair:
            if result.status != LambertStatus.OK:
                raise AssertionError(f'Known conic failed: {case.label,result}')
            actual.append((np.array(result.v1_m_s)/scale, np.array(result.v2_m_s)/scale))
        velocity_scale = max(1., *(np.linalg.norm(v) for v in expected))
        known = min(max(np.linalg.norm(a-expected[0]), np.linalg.norm(b-expected[1]))/velocity_scale
                    for a, b in actual)
        if known > 2e-11:
            raise AssertionError(f'Known conic velocity failure: {case.label,known}')
        maxima['known_velocity'] = max(maxima['known_velocity'], known)
        references = []
        for low in (False, True):
            a, b, iterations, _ = gooding1990(
                1., r1, r2, elapsed, M=problem.revolutions,
                prograde=bool(case.normal[2] > 0), low_path=low,
                maxiter=100, atol=1e-13, rtol=1e-13, full_output=True)
            if iterations >= 100 or not np.isfinite([a, b]).all():
                raise AssertionError('Independent Gooding reference failed to converge')
            references.append((a, b))
        permutation = min(((0, 1), (1, 0)), key=lambda order: sum(
            np.linalg.norm(actual[j][0]-references[i][0]) for j, i in enumerate(order)))
        for j, ((v1, v2), result) in enumerate(zip(actual, pair)):
            count += 1
            warnings += result.root.x_precision_limited is True
            gv1, gv2 = references[permutation[j]]
            gooding_error = max(np.linalg.norm(v1-gv1), np.linalg.norm(v2-gv2))/velocity_scale
            h1, h2 = np.cross(r1, v1), np.cross(r2, v2)
            if np.dot(h1, case.normal) <= 0:
                raise AssertionError('Wrong conic direction')
            en1, en2 = np.dot(v1, v1)/2-1/np.linalg.norm(r1), np.dot(v2, v2)/2-1/np.linalg.norm(r2)
            energy = abs(en1-en2)/max(1., abs(en1), abs(en2), 1/np.linalg.norm(r1), 1/np.linalg.norm(r2))
            momentum = np.linalg.norm(h1-h2)/max(1., np.linalg.norm(h1))
            dr, dv, sweep = propagate_winding(r1, v1, r2, v2, elapsed, problem.revolutions,
                                               result.geometry.geometry.transfer_angle_rad)
            kp,kv,kepler_sweep=kepler_propagate(r1,v1,elapsed)
            kepler_dr=np.linalg.norm(np.array(list(map(float,kp)))-r2)/max(1.,np.linalg.norm(r2))
            kepler_dv=np.linalg.norm(np.array(list(map(float,kv)))-v2)/velocity_scale
            if abs(float(kepler_sweep)-(2*math.pi*problem.revolutions+result.geometry.geometry.transfer_angle_rad))>2e-8:
                raise AssertionError('Independent Kepler propagation has wrong winding')
            maxima['ode_position']=max(maxima['ode_position'],dr/max(1.,np.linalg.norm(r2)))
            maxima['ode_velocity']=max(maxima['ode_velocity'],dv/velocity_scale)
            errors = dict(gooding_velocity=gooding_error, energy=energy, momentum=momentum,
                          propagated_position=kepler_dr, propagated_velocity=kepler_dv,
                          winding_angle=abs(sweep-(2*math.pi*problem.revolutions+result.geometry.geometry.transfer_angle_rad)))
            bounds = dict(gooding_velocity=2e-11, energy=2e-12, momentum=2e-12,
                          propagated_position=3e-9, propagated_velocity=3e-9, winding_angle=2e-8)
            for key, error in errors.items():
                if not math.isfinite(error) or error > bounds[key]:
                    raise AssertionError(f'Conic validation failure: {case.label,j,key,error}')
                maxima[key] = max(maxima[key], float(error))
    result = dict(problems=count//2, branch_solutions=count, precision_warnings=warnings, maxima=maxima)
    print('Independent conics/Gooding/propagation:', json.dumps(result), flush=True)
    return result


def hard_problems():
    with mp.workdps(140):
        u = np.array((.3, -.4, .5)); u /= np.linalg.norm(u)
        v = np.cross(u, (.7, .1, -.2)); v /= np.linalg.norm(v)
        for angle in (1e-4, 1e-8, 1e-12, 2e-14, math.pi-1e-8):
            for ratio in (1., 2.):
                for m in (1, 3):
                    for long_way in (False, True):
                        r1 = 7e6*u
                        r2 = 7e6*ratio*(math.cos(angle)*u+math.sin(angle)*v)
                        a, b = tuple(map(mp.mpf,r1)),tuple(map(mp.mpf,r2))
                        s = (norm(a)+norm(b)+norm(tuple(y-x for x,y in zip(a,b))))/2
                        q = norm(tuple(y-x for x,y in zip(a,b)))/s
                        lam = mp.sqrt(1-q)*(-1 if long_way else 1)
                        _, time = minimum_reference(lam,m)
                        tof = float(time*mp.mpf('1.5')*mp.sqrt(s**3/(2*mp.mpf(3.986004418e14))))
                        yield LambertInput(tuple(r1),tuple(r2),tof,3.986004418e14,long_way,m)


def validate_actual_inputs():
    maximum = 0.
    count = 0
    for problem in hard_problems():
        pair = solve_multi_rev_pair(problem)
        reference = physical_reference(problem)
        if reference is None:
            raise AssertionError('Constructed feasible actual-input case became infeasible')
        _, velocities, _ = reference
        for result, (v1,v2) in zip(pair,velocities):
            if result.status != LambertStatus.OK:
                raise AssertionError(f'Actual-input solve failed: {problem,result}')
            with mp.workdps(140):
                scale = max(mp.sqrt(mp.mpf(problem.mu_m3_s2)/norm(tuple(map(mp.mpf,problem.r1_m)))),norm(v1),norm(v2))
                error = float(max(norm(tuple(mp.mpf(x)-y for x,y in zip(result.v1_m_s,v1))),
                                  norm(tuple(mp.mpf(x)-y for x,y in zip(result.v2_m_s,v2))))/scale)
            if error > 2e-12:
                raise AssertionError(f'Actual-input velocity error: {problem,result.branch,error}')
            maximum = max(maximum,error)
            count += 1
    result = dict(problems=count//2, branch_solutions=count, max_scaled_velocity_error=maximum)
    print('Actual-input multiprecision velocities:', json.dumps(result), flush=True)
    return result


def validate_conditioning():
    counts = Counter()
    maximum_error = maximum_envelope_ratio = 0.
    for m in (1,2,20):
        minimum = find_minimum_multi_rev(0.4142135623730951,m,q=.8284271247461902)
        time_scale = math.sqrt(((2+math.sqrt(2))/2)**3/2)
        for gap in (-1e-8,0.,1e-14,1e-12,1e-8,1e-4):
            problem=LambertInput((1.,0.,0.),(0.,1.,0.),minimum.time_nondim*(1+gap)*time_scale,1.,revolutions=m)
            pair=solve_multi_rev_pair(problem)
            reference=physical_reference(problem)
            for j,result in enumerate(pair):
                counts[result.root.status.name]+=1
                if result.root.status in (MultiRootStatus.NO_SOLUTION,MultiRootStatus.NEAR_MINIMUM):
                    if result.v1_m_s is not None: raise AssertionError('Unresolved conditioning published velocities')
                    continue
                if result.status != LambertStatus.OK or reference is None:
                    raise AssertionError(f'Conditioned case failed: {m,gap,result}')
                roots,velocities,velocity_function=reference
                with mp.workdps(140):
                    expected1,expected2=velocities[j]
                    error=float(max(norm(tuple(mp.mpf(a)-b for a,b in zip(result.v1_m_s,expected1))),
                                    norm(tuple(mp.mpf(a)-b for a,b in zip(result.v2_m_s,expected2)))))
                    gradients=[tuple(mp.diff(lambda x:velocity_function(x)[k][i],roots[j]) for i in range(3)) for k in (0,1)]
                    sensitivity=float(max(map(norm,gradients)))
                    envelope=max(2e-11,2*sensitivity*(result.root.estimated_x_error or 0.))
                if error>envelope: raise AssertionError(f'Conditioned velocity exceeded estimate: {m,gap,error,envelope}')
                if error>2e-11 and result.root.x_precision_limited is not True:
                    raise AssertionError('Unflagged conditioned velocity degradation')
                counts['precision_warning']+=result.root.x_precision_limited is True
                maximum_error=max(maximum_error,error)
                maximum_envelope_ratio=max(maximum_envelope_ratio,error/envelope)
    result=dict(counts=dict(counts),max_velocity_error_unit_scale=maximum_error,max_error_over_envelope=maximum_envelope_ratio)
    print('Near-minimum conditioning:',json.dumps(result),flush=True)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--section',choices=('all','scalar','conics','actual','conditioning'),default='all')
    parser.add_argument('--json',type=Path)
    args=parser.parse_args()
    checks=dict(scalar=validate_scalar,conics=validate_conics,actual=validate_actual_inputs,conditioning=validate_conditioning)
    results={name:check() for name,check in checks.items() if args.section in ('all',name)}
    print('All requested complete multi-revolution checks passed.',flush=True)
    if args.json: args.json.write_text(json.dumps(results,indent=2)+'\n')
