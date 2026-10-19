"""Independent conic states, Gooding solves, and two-body ODE propagation.

Reference states and elapsed times come from 80-digit true-anomaly conic
equations and quadrature of dt/dnu=r^2/sqrt(mu*p). They use no production
geometry, TOF, starter, root, or velocity formula. Dependencies are validation
only; see requirements.txt and docs/fifth-increment.md for scopes and scales.
"""
from collections import Counter
from dataclasses import dataclass
from importlib.metadata import version
import math
from pathlib import Path
import random
import sys

from lamberthub import gooding1990
import mpmath as mp
import numpy as np
from scipy.integrate import solve_ivp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.lambert_geometry import LambertInput
from src.lambert_solver import LambertStatus, solve_zero_rev

# Bounds are fixed before validation, in conic p and sqrt(mu/p) units.
BOUNDS = dict(known_velocity=2e-11, gooding_velocity=2e-11,
              energy=2e-12, angular_momentum=2e-12,
              propagated_position=2e-9, propagated_velocity=2e-9,
              control_position=2e-9, control_velocity=2e-9,
              propagated_position_refined=5e-10, propagated_velocity_refined=5e-10,
              control_position_refined=5e-10, control_velocity_refined=5e-10)


@dataclass(frozen=True)
class Case:
    label: str
    problem: LambertInput
    v1: tuple
    v2: tuple
    length_scale: float  # semilatus rectum p
    speed_scale: float   # sqrt(mu/p)
    normal: tuple
    eccentricity: float


def basis(rng):
    """Random proper 3D orbital basis, avoiding Gooding's polar ambiguity."""
    normal = np.array([rng.uniform(-1, 1) for _ in range(3)])
    if abs(normal[2]) < .1:
        normal[2] = .1
    normal /= np.linalg.norm(normal)
    first = np.cross(normal, [0., 0., 1.])
    first /= np.linalg.norm(first)
    return first, np.cross(normal, first), normal


def conic_case(label, e, start, stop, length, mu, axes):
    e, start, stop = map(mp.mpf, (e, start, stop))
    p, gm = mp.mpf(length), mp.mpf(mu)
    speed = mp.sqrt(gm/p)
    u, v, normal = axes

    def state(nu):
        co, si = mp.cos(nu), mp.sin(nu)
        radius = p/(1+e*co)
        # Rotate with exact binary64 basis entries lifted into multiprecision.
        r = tuple(float(radius*(co*float(a)+si*float(b))) for a, b in zip(u, v))
        velocity = tuple(float(speed*(-si*float(a)+(e+co)*float(b))) for a, b in zip(u, v))
        return r, velocity

    segments = max(1, math.ceil(float(stop-start)/(math.pi/4)))
    nodes = [start+(stop-start)*k/segments for k in range(segments+1)]
    integral = mp.quad(lambda nu: 1/(1+e*mp.cos(nu))**2, nodes)
    tof = float(p/speed*integral)
    r1, v1 = state(start)
    r2, v2 = state(stop)
    problem = LambertInput(r1, r2, tof, mu, bool(stop-start > mp.pi))
    return Case(label, problem, v1, v2, length, float(speed), tuple(normal), float(e))


def cases():
    rng = random.Random(958705)
    scales = ((1., 1.), (7e6, 3.986004418e14), (1.495978707e11, 1.32712440018e20))
    eccentricities = (0., .1, .6, .9, 1-1e-8, 1-1e-12, 1., 1+1e-12, 1+1e-8, 1.2, 2., 5.)
    for i in range(240):
        e = eccentricities[i % len(eccentricities)]
        if e < .95:
            start = rng.uniform(-math.pi, math.pi)
            stop = start+rng.uniform(.1, 2*math.pi-.1)
        else:
            bound = math.acos(-1/e) if e > 1 else math.pi
            start = rng.uniform(-bound+.2, -.1)
            stop = rng.uniform(.1, bound-.2)
        length, mu = scales[i % len(scales)]
        yield conic_case(f'conic-{i}-e{e}', e, start, stop, length, mu, basis(rng))
    # Explicit short/long, clockwise/counterclockwise circles and near-pi arcs.
    for i, angle in enumerate((1e-4, math.pi-1e-6, math.pi+1e-6, 2*math.pi-1e-4)):
        for sense in (-1., 1.):
            axes = (np.array([1., 0., 0.]), np.array([0., sense, 0.]), (0., 0., sense))
            yield conic_case(f'circle-{i}-{sense}', 0., 0., angle, 1., 1., axes)


def propagate(r1, v1, elapsed, refined=False):
    def rhs(t, state):
        r = state[:3]
        return np.concatenate((state[3:], -r/np.linalg.norm(r)**3))
    rtol, atol = (3e-14, 3e-15) if refined else (2e-13, 2e-14)
    solution = solve_ivp(rhs, (0., elapsed), np.concatenate((r1, v1)),
                         method='DOP853', rtol=rtol, atol=atol,
                         max_step=elapsed/(128 if refined else 64))
    if not solution.success or solution.t[-1] != elapsed:
        raise AssertionError(f'ODE propagation failed: {solution.message}')
    return solution.y[:3, -1], solution.y[3:, -1]


def validate():
    mp.mp.dps = 80
    maximum = {name: (0., None) for name in BOUNDS}
    populations = Counter()
    gooding_stress_known_error = (0., None)
    gooding_stress_disagreement = (0., None)
    largest_root_residual = 0.
    largest_iterations = 0
    for case in cases():
        p, vscale, problem = case.length_scale, case.speed_scale, case.problem
        result = solve_zero_rev(problem)
        if result.status != LambertStatus.OK:
            raise AssertionError(f'Custom solve failed: {case.label}: {result}')
        r1, r2 = np.array(problem.r1_m)/p, np.array(problem.r2_m)/p
        v1, v2 = np.array(result.v1_m_s)/vscale, np.array(result.v2_m_s)/vscale
        known1, known2 = np.array(case.v1)/vscale, np.array(case.v2)/vscale
        elapsed = problem.tof_s/(p/vscale)
        position_scale = max(1., np.linalg.norm(r2))
        velocity_scale = max(1., np.linalg.norm(known1), np.linalg.norm(known2))

        # Gooding's fixed-z prograde convention differs from our geometric flag.
        # For these nonpolar planes the intended angular-momentum z sign maps it.
        gv1, gv2, iterations, _ = gooding1990(
            1., r1, r2, elapsed, M=0, prograde=bool(case.normal[2] > 0),
            maxiter=100, atol=1e-13, rtol=1e-13, full_output=True)
        if iterations >= 100 or not np.isfinite([gv1, gv2]).all():
            raise AssertionError(f'Independent Gooding solve failed: {case.label}')

        h1, h2 = np.cross(r1, v1), np.cross(r2, v2)
        if np.dot(h1, case.normal) <= 0:
            raise AssertionError(f'Wrong angular-momentum direction: {case.label}')
        e1 = np.dot(v1, v1)/2-1/np.linalg.norm(r1)
        e2 = np.dot(v2, v2)/2-1/np.linalg.norm(r2)
        energy_scale = max(1., np.dot(v1, v1)/2, 1/np.linalg.norm(r1),
                           np.dot(v2, v2)/2, 1/np.linalg.norm(r2))
        gooding_difference = max(np.linalg.norm(v1-gv1), np.linalg.norm(v2-gv2))/velocity_scale
        errors = dict(known_velocity=max(np.linalg.norm(v1-known1), np.linalg.norm(v2-known2))/velocity_scale,
                      energy=abs(e1-e2)/energy_scale,
                      angular_momentum=np.linalg.norm(h1-h2)/max(1., np.linalg.norm(h1)))
        if case.label.startswith('circle-'):
            # Gooding 1.0.0 obtains theta via acos(dot/(norm*norm)); it loses
            # angle precision near 0/pi. These eight stress comparisons remain
            # visible diagnostics. Analytical velocities retain the strict
            # acceptance bound, rather than weakening it to match this reference.
            known_error = max(np.linalg.norm(gv1-known1), np.linalg.norm(gv2-known2))/velocity_scale
            gooding_stress_known_error = max(gooding_stress_known_error, (known_error, case.label))
            gooding_stress_disagreement = max(gooding_stress_disagreement, (gooding_difference, case.label))
            populations['gooding_stress_diagnostics'] += 1
        else:
            errors['gooding_velocity'] = gooding_difference
            populations['gooding_strict_comparisons'] += 1
        # Also integrate the independently known state as a control: this
        # exposes integrator error rather than attributing all mismatch to Lambert.
        for refined in (False, True):
            actual_r, actual_v = propagate(r1, v1, elapsed, refined)
            control_r, control_v = propagate(r1, known1, elapsed, refined)
            suffix = '_refined' if refined else ''
            errors['propagated_position'+suffix] = np.linalg.norm(actual_r-r2)/position_scale
            errors['propagated_velocity'+suffix] = np.linalg.norm(actual_v-v2)/velocity_scale
            errors['control_position'+suffix] = np.linalg.norm(control_r-r2)/position_scale
            errors['control_velocity'+suffix] = np.linalg.norm(control_v-known2)/velocity_scale
        for name, error in errors.items():
            if not math.isfinite(error) or error > BOUNDS[name]:
                raise AssertionError(f'{name}={error:.17g} > {BOUNDS[name]}: {case.label}')
            if error > maximum[name][0]:
                maximum[name] = float(error), case.label
        largest_root_residual = max(largest_root_residual, abs(result.root.relative_residual))
        largest_iterations = max(largest_iterations, result.root.iterations)
        populations['total'] += 1
        populations['long' if problem.long_way else 'short'] += 1
        populations['positive_hz' if case.normal[2] > 0 else 'negative_hz'] += 1
        populations['elliptic' if case.eccentricity < 1 else 'parabolic' if case.eccentricity == 1 else 'hyperbolic'] += 1
        populations['x_precision_warning'] += result.root.x_precision_limited is True
        populations['x_precision_unknown'] += result.root.x_precision_limited is None

    print('Validation dependencies:', ', '.join(f'{n}={version(n)}' for n in ('mpmath', 'lamberthub', 'numpy', 'scipy', 'numba', 'llvmlite')))
    print('Corpus:', dict(populations))
    print('Unexpected failures: 0; Gooding comparisons and four propagations per case:', populations['total'])
    for name, (error, label) in maximum.items():
        print(f'Max {name}: {error:.17g}; bound={BOUNDS[name]:g}; case={label}')
    print(f'Gooding stress diagnostic max disagreement: {gooding_stress_disagreement}; max Gooding error against analytical velocity: {gooding_stress_known_error}')
    print(f'Max computed relative root residual: {largest_root_residual:.17g}; max completed root updates: {largest_iterations}')
    print('All errors are scaled as documented; propagation includes numerical integrator error.')


def validate_small_arcs():
    """Regression acceptance for the formerly degraded small circular arcs."""
    print('Tiny circular arc validation (velocity bound 3e-14):')
    for angle in (1e-8, 1e-10, 1e-12, 1e-13, 2e-14):
        for long_way in (False, True):
            tof = 2*math.pi-angle if long_way else angle
            p = LambertInput((1., 0., 0.), (math.cos(angle), math.sin(angle), 0.), tof, 1., long_way)
            result = solve_zero_rev(p)
            if result.status != LambertStatus.OK:
                raise AssertionError(f'Tiny-arc solve failed: {angle}, {long_way}, {result}')
            sign = -1 if long_way else 1
            # cos(angle) rounds to 1: use the asymptotic solution for the
            # received endpoint (1,angle), not an unrounded circular orbit.
            known1 = (sign*angle/2, sign, 0.)
            known2 = (-sign*angle/2, sign, 0.)
            error = max(math.dist(result.v1_m_s, known1), math.dist(result.v2_m_s, known2))
            if error > 3e-14:
                raise AssertionError(f'Tiny-arc velocity error: {angle}, {long_way}, {error}')
            print(f'angle={angle:g}, long_way={long_way}, velocity_error={error:.17g}, '
                  f'q={result.geometry.geometry.q:.17g}, '
                  f'root_warning={result.root.x_precision_limited}')


if __name__ == '__main__':
    validate()
    validate_small_arcs()
