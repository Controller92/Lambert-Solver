"""Validate native scalar C++ and optional actual CUDA against independent math.

CPU/GPU/Python agreement is secondary to multiprecision Lagrange inversion,
analytical conic states, independent Gooding solves, and ODE propagation.
CUDA runs only with --gpu and fails explicitly if unavailable.
"""
import argparse
from collections import Counter
import math
import re
from pathlib import Path
import sys

import mpmath as mp
import numpy as np
from lamberthub import gooding1990

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.lambert_geometry import LambertInput, prepare_geometry
from src.lambert_native import NativeEngine, pack_problems
from src.lambert_solver import solve_zero_rev
from validate_tof import cases as tof_cases, lagrange_reference
from validate_root import bisection_reference, conditioning_cases
from validate_root_preparation import cases as derivative_cases, numerical_derivatives
from validate_precision import scalar_cases, physical_cases, physical_reference, norm
from validate_solver import cases as conic_cases, propagate
from validate_mars2020 import EPOCHS_UTC, load, states


def check_roots(rows, targets, references):
    maximum, ulps, warnings, unknown = 0., 0., 0, 0
    for r, target, (known, lam) in zip(rows, targets, references):
        if r['status'] != 0:
            raise AssertionError(f'Native root failed: {known}, {lam}, {r}')
        reference, _ = bisection_reference(target, lam, known)
        error = abs(mp.mpf(float(r['x']))-reference)
        scaled = error/(1+abs(reference))
        estimate = float(r['estimated_x_error'])
        allowance = max(mp.mpf('5e-14')*(1+abs(reference)), estimate if math.isfinite(estimate) else 0.)
        if error > allowance or (scaled > mp.mpf('5e-14') and r['precision_limited'] != 1):
            raise AssertionError(f'Native inverse error/conditioning: {known}, {lam}, {error}, {r}')
        exact = lagrange_reference(float(r['x']), lam, 160)
        forward_ulps = float(abs(mp.mpf(float(r['time']))-exact)/math.ulp(float(r['time'])))
        if forward_ulps > 16 or abs(float(r['relative_residual'])) > 5e-13:
            raise AssertionError(f'Native root residual/roundoff: {forward_ulps}, {r}')
        if abs(r['last_step']) > 1e-14+1e-14*abs(r['x']):
            raise AssertionError('Native root failed its step tolerance')
        if r['iterations'] != r['householder']+r['bisection']+r['verification']:
            raise AssertionError('Native update counts are inconsistent')
        maximum, ulps = max(maximum, float(scaled)), max(ulps, forward_ulps)
        warnings += r['precision_limited'] == 1
        unknown += r['precision_limited'] == -1
    return dict(count=len(rows), max_scaled_x_error=maximum, max_forward_ulps=ulps,
                precision_warnings=int(warnings), precision_unknown=int(unknown))


def scalar_validation(engine):
    mp.mp.dps = 120
    ordinary = [(x, mp.mpf(l), math.nan) for x, l in tof_cases()+conditioning_cases()]
    retained = [(x, sign*mp.sqrt(1-mp.mpf(q)), q) for x, q, sign in scalar_cases()]
    references = ordinary+retained
    triples = np.array([(x, float(lam), q) for x, lam, q in references])
    actual = engine.probe('tof', triples)
    targets, worst = [], 0.
    for r, (x, lam, _) in zip(actual, references):
        expected = lagrange_reference(x, lam, 160)
        error = float(abs(mp.mpf(float(r['time']))/expected-1))
        if r['status'] != 0 or error > 5e-14:
            raise AssertionError(f'Native TOF failure: {x}, {lam}, {r}, {error}')
        worst = max(worst, error)
        targets.append(float(expected))
    print(f'C++ independent TOF cases={len(actual)}, max_relative_error={worst:.17g}')
    triples[:, 0] = targets
    roots = engine.probe('roots', triples)
    print('C++ independent roots:', check_roots(roots, targets, [(x, l) for x, l, _ in references]))

    references = [(x, mp.mpf(l), math.nan) for x, l in derivative_cases()]
    references += [(x, lam, q) for x, lam, q in retained if abs(x) < 1e100]
    actual = engine.probe('derivatives', [(x, float(l), q) for x, l, q in references])
    worst = [0., 0., 0.]
    for r, (x, lam, _) in zip(actual, references):
        if r['status'] != 0:
            raise AssertionError(f'C++ derivative failure: {x}, {lam}, {r}')
        expected = numerical_derivatives(x, lam, '1e-12', 160)
        for i, (name, reference, bound) in enumerate(zip(('first', 'second', 'third'), expected, (5e-14, 2e-13, 2e-12))):
            error = float(abs(mp.mpf(float(r[name]))/reference-1))
            if error > bound:
                raise AssertionError(f'C++ derivative error {name}: {x}, {lam}, {error}')
            worst[i] = max(worst[i], error)
    print(f'C++ independent derivative cases={len(actual)}, max_relative_errors={worst}')


def physical_validation(cpu, gpu=None):
    mp.mp.dps = 120
    conics = list(conic_cases())
    hard = list(physical_cases())
    problems = [case.problem for case in conics]+[p for _, p in hard]
    # Stress full solves with derivative underflow and representable velocities
    # at huge hyperbolic x; compare these to the separately verified Python core.
    stress = []
    for angle in (.1, 1e-12, math.pi-1e-8):
        for long_way in (False, True):
            for known in (-.99, -1e-5, 0., 1., 2., 1e4, 1e40, 1e150):
                p = LambertInput((1., 0., 0.), (math.cos(angle), math.sin(angle), 0.), 1., 1., long_way)
                g = prepare_geometry(p).geometry
                lam = (-1 if long_way else 1)*mp.sqrt(1-mp.mpf(g.q)) if g.q < .5 else mp.mpf(g.lambda_)
                time = float(lagrange_reference(known, lam, 160))*g.time_scale_s
                stress.append(LambertInput(p.r1_m, p.r2_m, time, 1., long_way))
    problems += stress
    packed = pack_problems(problems)
    results = {'cpp': cpu.cpu(packed)}
    if gpu:
        with gpu.cuda_batch(len(packed)) as batch:
            batch.upload(packed)
            batch.run()  # First launch is not a performance measurement.
            results['cuda'] = batch.download()
        print('CUDA resources:', gpu.device_info())
    for name, rows in results.items():
        maxima = dict(known_velocity=0., actual_input_velocity=0., gooding_velocity=0.,
                      energy=0., angular_momentum=0., propagated_position=0., propagated_velocity=0.,
                      python_velocity=0., forward_ulps=0.)
        warnings, unknown = 0, 0
        for i, (p, r) in enumerate(zip(problems, rows)):
            py = solve_zero_rev(p)
            if r['status'] != 0 or py.status != 0:
                raise AssertionError(f'{name} full solve failed at {i}: native={r}, python={py}')
            v1, v2 = r['v1'], r['v2']
            speed = max(math.sqrt(p.mu_m3_s2/math.hypot(*p.r1_m)), np.linalg.norm(py.v1_m_s), np.linalg.norm(py.v2_m_s))
            error = max(np.linalg.norm(v1-py.v1_m_s), np.linalg.norm(v2-py.v2_m_s))/speed
            if error > 2e-11:
                raise AssertionError(f'{name}/Python velocity difference: {i}, {error}')
            maxima['python_velocity'] = max(maxima['python_velocity'], error)
            root = r['root']
            # Independently calibrate the forward-roundoff diagnostic on the
            # shape actually carried by each backend. Retain the better
            # resolved member of the redundant lambda/q pair in multiprecision.
            lam = mp.mpf(float(r['lambda']))
            if r['q'] < .5:
                lam = (-1 if lam < 0 else 1)*mp.sqrt(1-mp.mpf(float(r['q'])))
            exact_time = lagrange_reference(float(root['x']), lam, 160)
            forward_ulps = float(abs(mp.mpf(float(root['time']))-exact_time)/math.ulp(float(root['time'])))
            if forward_ulps > 16:
                raise AssertionError(f'{name} forward-roundoff allowance exceeded: {i}, {forward_ulps}')
            maxima['forward_ulps'] = max(maxima['forward_ulps'], forward_ulps)
            if abs(root['relative_residual']) > 5e-13 or root['iterations'] != root['householder']+root['bisection']+root['verification']:
                raise AssertionError(f'{name} full root diagnostics invalid: {i}, {root}')
            warnings += root['precision_limited'] == 1
            unknown += root['precision_limited'] == -1
            if i < len(conics):
                c = conics[i]
                position1, position2 = np.array(p.r1_m)/c.length_scale, np.array(p.r2_m)/c.length_scale
                u1, u2 = v1/c.speed_scale, v2/c.speed_scale
                known1, known2 = np.array(c.v1)/c.speed_scale, np.array(c.v2)/c.speed_scale
                vscale = max(1., np.linalg.norm(known1), np.linalg.norm(known2))
                error = max(np.linalg.norm(u1-known1), np.linalg.norm(u2-known2))/vscale
                if error > 2e-11:
                    raise AssertionError(f'{name} known conic error: {i}, {error}')
                maxima['known_velocity'] = max(maxima['known_velocity'], error)
                h1, h2 = np.cross(position1, u1), np.cross(position2, u2)
                if np.dot(h1, c.normal) <= 0:
                    raise AssertionError(f'{name} direction failure: {i}')
                energy1, energy2 = np.dot(u1, u1)/2-1/np.linalg.norm(position1), np.dot(u2, u2)/2-1/np.linalg.norm(position2)
                escale = max(1., np.dot(u1, u1)/2, np.dot(u2, u2)/2, 1/np.linalg.norm(position1), 1/np.linalg.norm(position2))
                ee, he = abs(energy1-energy2)/escale, np.linalg.norm(h1-h2)/max(1., np.linalg.norm(h1))
                if ee > 2e-12 or he > 2e-12:
                    raise AssertionError(f'{name} invariant failure: {i}, {ee}, {he}')
                maxima['energy'], maxima['angular_momentum'] = max(maxima['energy'], ee), max(maxima['angular_momentum'], he)
                elapsed = p.tof_s/(c.length_scale/c.speed_scale)
                if not c.label.startswith('circle-'):
                    gv1, gv2 = gooding1990(1., position1, position2, elapsed,
                                           prograde=bool(c.normal[2] > 0), maxiter=100, atol=1e-13, rtol=1e-13)
                    error = max(np.linalg.norm(u1-gv1), np.linalg.norm(u2-gv2))/vscale
                    if error > 2e-11:
                        raise AssertionError(f'{name} Gooding discrepancy: {i}, {error}')
                    maxima['gooding_velocity'] = max(maxima['gooding_velocity'], error)
                arrival_r, arrival_v = propagate(position1, u1, elapsed, refined=True)
                re = np.linalg.norm(arrival_r-position2)/max(1., np.linalg.norm(position2))
                ve = np.linalg.norm(arrival_v-u2)/vscale
                if re > 5e-10 or ve > 5e-10:
                    raise AssertionError(f'{name} independent ODE failure: {i}, {re}, {ve}')
                maxima['propagated_position'], maxima['propagated_velocity'] = max(maxima['propagated_position'], re), max(maxima['propagated_velocity'], ve)
            elif i < len(conics)+len(hard):
                expected1, expected2, *_ = physical_reference(p)
                scale = max(mp.sqrt(mp.mpf(p.mu_m3_s2)/norm(tuple(map(mp.mpf, p.r1_m)))), norm(expected1), norm(expected2))
                error = float(max(norm(tuple(mp.mpf(float(a))-b for a, b in zip(v1, expected1))),
                                  norm(tuple(mp.mpf(float(a))-b for a, b in zip(v2, expected2))))/scale)
                if error > 2e-12:
                    raise AssertionError(f'{name} actual-input reference failure: {i}, {error}')
                maxima['actual_input_velocity'] = max(maxima['actual_input_velocity'], error)
        print(f'{name}: full transfers={len(rows)} (conic={len(conics)}, actual-input={len(hard)}, extreme={len(stress)}), '
              f'precision_warnings={warnings}, unknown={unknown}; maxima={maxima}')


def mission_validation(cpu, gpu=None):
    mu = float(re.search(r'GM,\s*km\^3/s\^2\s*=\s*([0-9.E+\-]+)', load('sun')).group(1))*1e9
    earth = states('earth', '399', EPOCHS_UTC)
    mars = states('mars', '499', EPOCHS_UTC)
    spacecraft = states('spacecraft', '-168', EPOCHS_UTC[1:])
    endpoints = [(earth[0], mars[-1]), (spacecraft[1], spacecraft[-2]), (spacecraft[4], spacecraft[5])]
    problems = [LambertInput(tuple(a['r']), tuple(b['r']), float(b['tdb_seconds']-a['tdb_seconds']),
                            mu, bool(np.cross(a['r'], b['r'])[2] < 0)) for a, b in endpoints]
    packed = pack_problems(problems)
    results = {'cpp': cpu.cpu(packed)}
    if gpu:
        with gpu.cuda_batch(len(packed)) as batch:
            batch.upload(packed)
            batch.run()
            results['cuda'] = batch.download()
    for name, rows in results.items():
        worst_gooding = worst_position = worst_velocity = 0.
        for p, row in zip(problems, rows):
            if row['status'] != 0 or row['root']['precision_limited'] != 0:
                raise AssertionError(f'{name} Mars 2020 failure: {row}')
            length = math.hypot(*p.r1_m)
            speed = math.sqrt(mu/length)
            position1, position2 = np.array(p.r1_m)/length, np.array(p.r2_m)/length
            u1, u2 = row['v1']/speed, row['v2']/speed
            elapsed = p.tof_s/(length/speed)
            gv1, gv2 = gooding1990(1., position1, position2, elapsed, prograde=True,
                                   maxiter=100, atol=1e-13, rtol=1e-13)
            difference = max(np.linalg.norm(u1-gv1), np.linalg.norm(u2-gv2))*speed
            arrival_r, arrival_v = propagate(position1, u1, elapsed, refined=True)
            position_error = np.linalg.norm(arrival_r-position2)*length
            velocity_error = np.linalg.norm(arrival_v-u2)*speed
            if difference > 1e-6 or position_error > 10. or velocity_error > 1e-6:
                raise AssertionError(f'{name} Mars 2020 accuracy failure: {difference}, {position_error}, {velocity_error}')
            worst_gooding = max(worst_gooding, difference)
            worst_position = max(worst_position, position_error)
            worst_velocity = max(worst_velocity, velocity_error)
        print(f'{name} Mars 2020: 3 boundary problems; max Gooding difference={worst_gooding:.17g} m/s; '
              f'max propagated closure={worst_position:.17g} m, {worst_velocity:.17g} m/s')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu', action='store_true')
    args = parser.parse_args()
    cpu = NativeEngine()
    scalar_validation(cpu)
    physical_validation(cpu, NativeEngine('cuda') if args.gpu else None)
    mission_validation(cpu, NativeEngine('cuda') if args.gpu else None)
    print('All requested native validation passed. CUDA was '+('executed.' if args.gpu else 'not requested.'))
