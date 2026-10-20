"""Cached Horizons Juno regression for complete Python M>0 transfers.

--refresh retrieves public data sequentially; ordinary runs are offline.
States are Jupiter-centered geometric ICRF vectors in continuous TDB time.
Gooding and multiprecision roots/velocities are independent validation references.
Product left/right velocities are also propagated and checked for orbital winding.
"""
import argparse
import csv
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from decimal import Decimal
import hashlib
import io
import json
import math
from pathlib import Path
import re
import sys
from urllib.parse import urlencode
from urllib.request import urlopen

import mpmath as mp
import numpy as np
from lamberthub import gooding1990

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.lambert_geometry import LambertInput, GeometryStatus, prepare_geometry
from src.lambert_multirev import (
    MinimumStatus, FeasibilityStatus, classify_multirev_time, evaluate_derivatives_multi_rev,
    find_minimum_multi_rev,
)
from src.lambert_multirev_solver import solve_multi_rev_pair
from src.lambert_multirev_root import MultiRootStatus
from src.lambert_solver import LambertStatus
from src.lambert_tof import TofStatus
from validate_multirev import lagrange, minimum_reference, reference_lambda
from validate_multirev_solver import physical_reference, kepler_propagate, propagate_winding

DATA = ROOT/'validation'/'data'/'juno'
API = 'https://ssd.jpl.nasa.gov/api/horizons.api'
START = '2017-02-10 00:00:00'
STOP = '2017-04-15 00:00:00'
VECTOR = dict(COMMAND="'-61'", OBJ_DATA="'YES'", MAKE_EPHEM="'YES'",
              EPHEM_TYPE="'VECTORS'", CENTER="'500@599'", REF_PLANE="'FRAME'",
              REF_SYSTEM="'ICRF'", OUT_UNITS="'KM-S'", VEC_TABLE="'2'",
              VEC_CORR="'NONE'", CSV_FORMAT="'YES'", TIME_TYPE="'TDB'")


def query(name, parameters):
    url = API+'?'+urlencode({'format': 'json', **parameters})
    with urlopen(url, timeout=60) as response:
        payload = json.load(response)
    if ('error' in payload or 'result' not in payload
            or (parameters.get('MAKE_EPHEM') == "'YES'" and '$$SOE' not in payload['result'])):
        raise RuntimeError(f'Horizons query failed: {name}: {payload}')
    text = payload['result']
    snapshot = dict(retrieved_utc=datetime.now(timezone.utc).isoformat(), url=url,
                    parameters=parameters, result_sha256=hashlib.sha256(text.encode()).hexdigest(),
                    response=payload)
    DATA.mkdir(parents=True, exist_ok=True)
    (DATA/f'{name}.json').write_text(json.dumps(snapshot, indent=2)+'\n')


def load(name):
    snapshot = json.loads((DATA/f'{name}.json').read_text())
    text = snapshot['response']['result']
    if hashlib.sha256(text.encode()).hexdigest() != snapshot['result_sha256']:
        raise ValueError(f'Horizons snapshot hash mismatch: {name}')
    return snapshot, text


def states(name):
    snapshot, text = load(name)
    if any(snapshot['parameters'].get(key) != value for key, value in VECTOR.items()):
        raise ValueError(f'Unexpected Horizons vector request: {name}')
    if text.count('$$SOE') != 1 or text.count('$$EOE') != 1:
        raise ValueError('Missing or ambiguous Horizons state table')
    header = text.split('$$SOE')[0]
    for required in ('Target body name: Juno (spacecraft) (-61)', 'Center body name: Jupiter (599)',
                     'Center-site name: BODY CENTER', 'GEOMETRIC cartesian states',
                     'Output units    : KM-S', 'Reference frame : ICRF', 'JDTDB', 'Date (TDB)'):
        if required not in header:
            raise ValueError(f'Unexpected Horizons conventions in {name}: {required}')
    rows = []
    for fields in csv.reader(io.StringIO(text.split('$$SOE')[1].split('$$EOE')[0])):
        if not fields:
            continue
        if len(fields) != 9 or fields[-1].strip():
            raise ValueError(f'Unexpected Horizons columns: {fields}')
        date = datetime.strptime(fields[1].strip(), 'A.D. %Y-%b-%d %H:%M:%S.%f')
        delta = date-datetime(2000, 1, 1, 12)
        seconds = Decimal(delta.days*86400+delta.seconds)+Decimal(delta.microseconds)/1000000
        if abs(Decimal(2451545)+seconds/86400-Decimal(fields[0].strip())) > Decimal('0.0000000006'):
            raise ValueError('Inconsistent redundant TDB date tags')
        position, velocity = np.array(fields[2:5], dtype=float)*1000, np.array(fields[5:8], dtype=float)*1000
        if not np.isfinite([position, velocity]).all():
            raise ValueError('Nonfinite Horizons state')
        rows.append(dict(date=date, tdb_seconds=seconds, r=position, v=velocity))
    if (not rows or any(b['tdb_seconds'] <= a['tdb_seconds'] for a, b in zip(rows, rows[1:]))
            or rows[0]['date'].strftime('%Y-%m-%d %H:%M:%S') != snapshot['parameters']['START_TIME'].strip("'")
            or rows[-1]['date'].strftime('%Y-%m-%d %H:%M:%S') != snapshot['parameters']['STOP_TIME'].strip("'")):
        raise ValueError('Unexpected Horizons time bounds/order')
    expected_step = {'\'1 h\'': 3600, '\'5 m\'': 300}.get(snapshot['parameters']['STEP_SIZE'])
    if expected_step is None or any(b['tdb_seconds']-a['tdb_seconds'] != expected_step
                                    for a, b in zip(rows, rows[1:])):
        raise ValueError('Unexpected Horizons sample cadence')
    return rows


def refresh():
    query('jupiter', dict(COMMAND="'599'", OBJ_DATA="'YES'", MAKE_EPHEM="'NO'"))
    query('orbit-hourly', dict(**VECTOR, START_TIME=f"'{START}'", STOP_TIME=f"'{STOP}'", STEP_SIZE="'1 h'"))
    coarse = states('orbit-hourly')
    closest = min(coarse, key=lambda row: np.linalg.norm(row['r']))['date']
    query('perijove-refined', dict(**VECTOR,
          START_TIME=f"'{closest-timedelta(hours=4):%Y-%m-%d %H:%M:%S}'",
          STOP_TIME=f"'{closest+timedelta(hours=4):%Y-%m-%d %H:%M:%S}'", STEP_SIZE="'5 m'"))


def reference_roots(target, lam, m, q):
    with mp.workdps(100):
        l = reference_lambda(lam, q)
        minimum, _ = minimum_reference(lam, m, q)
        roots = []
        for branch, lo, hi in (('left', mp.mpf(-1), minimum), ('right', minimum, mp.mpf(1))):
            for _ in range(260):
                x = (lo+hi)/2
                above = lagrange(x, l, m) > mp.mpf(target)
                if above == (branch == 'left'):
                    lo = x
                else:
                    hi = x
            roots.append((branch, float((lo+hi)/2)))
        return roots


def orbital_time(r1, v1, r2, v2, mu, m):
    """Independent elapsed-time/revolution check from eccentric anomaly."""
    a = 1/(2/np.linalg.norm(r1)-np.dot(v1, v1)/mu)
    h = np.cross(r1, v1)
    eccentricity = np.linalg.norm(np.cross(v1, h)/mu-r1/np.linalg.norm(r1))
    anomalies = []
    for r, v in ((r1, v1), (r2, v2)):
        cosine = (1-np.linalg.norm(r)/a)/eccentricity
        sine = np.dot(r, v)/(eccentricity*math.sqrt(mu*a))
        anomalies.append(math.atan2(sine, cosine))
    e1, e2 = anomalies
    delta_e = (e2-e1) % (2*math.pi)
    elapsed = math.sqrt(a**3/mu)*(delta_e-eccentricity*(math.sin(e2)-math.sin(e1))+2*math.pi*m)
    return a, elapsed


def validate():
    _, planet = load('jupiter')
    match = re.search(r'GM \(km\^3/s\^2\)\s*=\s*([0-9.E+\-]+)', planet)
    if not match:
        raise ValueError('Jupiter GM missing')
    mu = float(match.group(1))*1e9
    coarse, fine = states('orbit-hourly'), states('perijove-refined')
    if coarse[0]['date'].strftime('%Y-%m-%d %H:%M:%S') != START or coarse[-1]['date'].strftime('%Y-%m-%d %H:%M:%S') != STOP:
        raise ValueError('Unexpected mission arc')
    merged = {row['tdb_seconds']: row for row in coarse}
    for row in fine:
        if row['tdb_seconds'] in merged:
            previous = merged[row['tdb_seconds']]
            if not (np.array_equal(previous['r'], row['r']) and np.array_equal(previous['v'], row['v'])):
                raise ValueError('Overlapping Horizons snapshots disagree')
        merged[row['tdb_seconds']] = row
    rows = [merged[t] for t in sorted(merged)]
    position = np.array([row['r'] for row in rows])
    normal = np.cross(rows[0]['r'], rows[0]['v'])
    normal /= np.linalg.norm(normal)
    radial = rows[0]['r']/np.linalg.norm(rows[0]['r'])
    transverse = np.cross(normal, radial)
    angle = np.unwrap(np.arctan2(position@transverse, position@radial))
    increments = np.diff(angle)
    projection_ratio = np.hypot(position@transverse, position@radial)/np.linalg.norm(position, axis=1)
    if np.any(increments <= 0) or max(increments) >= .5 or min(projection_ratio) < .99:
        raise AssertionError('Observed winding count needs more sampling or a better reference plane')
    sweep = float(angle[-1]-angle[0])
    m = math.floor(sweep/(2*math.pi))
    if m != 1:
        raise AssertionError(f'Expected one complete observed revolution: {sweep}')
    start, stop = rows[0], rows[-1]
    duration = float(stop['tdb_seconds']-start['tdb_seconds'])
    long_way = bool(np.dot(np.cross(start['r'], stop['r']), normal) < 0)
    # The geometry helper is M-independent; the complete product uses its
    # dedicated M>0 interface with explicit branch selection below.
    p = LambertInput(tuple(start['r']), tuple(stop['r']), duration, mu, long_way)
    prepared = prepare_geometry(p)
    if prepared.status != GeometryStatus.OK:
        raise AssertionError(f'Juno geometry failure: {prepared}')
    g = prepared.geometry
    print(f'Juno (-61), center Jupiter (599), geometric ICRF/TDB; GM={mu:.17g} m^3/s^2', flush=True)
    print(f'{START} -> {STOP} TDB: {duration/86400:.9g} days; '
          f'{len(coarse)} hourly and {len(fine)} refined samples, {len(rows)} unique', flush=True)
    print(f'Observed projected sweep={math.degrees(sweep):.12g} deg, M={m}; '
          f'max sample angle={math.degrees(max(increments)):.9g} deg; '
          f'min projection/radius={min(projection_ratio):.12g}', flush=True)
    print(f'lambda={g.lambda_:.17g}, q={g.q:.17g}, target T={g.time_nondim:.17g}, long_way={long_way}', flush=True)
    feasibility = []
    for count in (1, 2, 3):
        minimum = find_minimum_multi_rev(g.lambda_, count, q=g.q)
        if minimum.status != MinimumStatus.OK:
            raise AssertionError(f'Juno minimum failure: {minimum}')
        outcome = classify_multirev_time(g.time_nondim, minimum)
        xr, tr = minimum_reference(g.lambda_, count, g.q)
        if (abs(minimum.x_min-float(xr)) > 2e-14
                or abs(minimum.time_nondim-float(tr)) > minimum.estimated_time_error):
            raise AssertionError('Juno independent minimum mismatch')
        expected = (FeasibilityStatus.TWO_SOLUTIONS if mp.mpf(g.time_nondim) > tr
                    else FeasibilityStatus.NO_SOLUTION)
        if outcome.status != expected:
            raise AssertionError('Juno independent feasibility classification mismatch')
        entry = dict(M=count, status=outcome.status.name, x_min=minimum.x_min,
                     minimum_days=minimum.time_nondim*g.time_scale_s/86400,
                     estimated_minimum_time_error_seconds=minimum.estimated_time_error*g.time_scale_s)
        feasibility.append(entry)
        print('Feasibility:', json.dumps(entry), flush=True)
    if feasibility[0]['status'] != 'TWO_SOLUTIONS':
        raise AssertionError('Observed one-revolution case must be feasible')

    roots = reference_roots(g.time_nondim, g.lambda_, m, g.q)
    length = np.linalg.norm(start['r'])
    speed = math.sqrt(mu/length)
    orbital_normal = np.cross(start['r'], stop['r'])*(-1 if long_way else 1)
    orbital_normal /= np.linalg.norm(orbital_normal)
    reference_a = [g.semiperimeter_m/(2*(1-x)*(1+x)) for _, x in roots]
    gooding = []
    for low in (False, True):
        v1, v2, iterations, _ = gooding1990(1., start['r']/length, stop['r']/length,
            duration/(length/speed), M=m, prograde=bool(orbital_normal[2] > 0),
            low_path=low, maxiter=100, atol=1e-13, rtol=1e-13, full_output=True)
        v1, v2 = v1*speed, v2*speed
        a, elapsed = orbital_time(start['r'], v1, stop['r'], v2, mu, m)
        if iterations >= 100 or abs(elapsed/duration-1) > 1e-10:
            raise AssertionError('Independent Gooding/anomaly time or revolution mismatch')
        if np.dot(np.cross(start['r'], v1), orbital_normal) <= 0:
            raise AssertionError('Independent reference orientation mismatch')
        gooding.append(dict(low_path=low, v1_m_s=v1.tolist(), v2_m_s=v2.tolist(), semimajor_axis_m=a,
                            anomaly_time_error_seconds=abs(elapsed-duration), iterations=int(iterations),
                            departure_ephemeris_velocity_difference_m_s=float(np.linalg.norm(v1-start['v'])),
                            arrival_ephemeris_velocity_difference_m_s=float(np.linalg.norm(v2-stop['v']))))
    # Determine the low/high library mapping from orbital energies, not a label assumption.
    order = min(((0, 1), (1, 0)), key=lambda permutation:
                sum(abs(gooding[i]['semimajor_axis_m']/reference_a[j]-1) for j, i in enumerate(permutation)))
    branches = []
    for j, (branch, x) in enumerate(roots):
        value = evaluate_derivatives_multi_rev(x, g.lambda_, m, q=g.q)
        if (value.status != TofStatus.OK or abs(value.time_nondim/g.time_nondim-1) > 5e-14
                or (value.first < 0) != (branch == 'left')):
            raise AssertionError('Product forward/derivative failure at independent mission root')
        comparison = gooding[order[j]]
        energy_error = abs(comparison['semimajor_axis_m']/reference_a[j]-1)
        if energy_error > 2e-10:
            raise AssertionError('Independent Lagrange and Gooding branch energies disagree')
        entry = dict(branch=branch, reference_x=x, semimajor_axis_relative_difference=energy_error,
                     product_relative_tof_residual=abs(value.time_nondim/g.time_nondim-1), **comparison)
        branches.append(entry)
        print('Reference branch:', json.dumps(entry), flush=True)
    problem = replace(p, revolutions=m)
    products = solve_multi_rev_pair(problem)
    physical = physical_reference(problem)
    product_records = []
    for j, product in enumerate(products):
        if product.status != LambertStatus.OK:
            raise AssertionError(f'Juno complete product failed: {product}')
        comparison = gooding[order[j]]
        gooding_difference = max(np.linalg.norm(np.array(product.v1_m_s)-comparison['v1_m_s']),
                                 np.linalg.norm(np.array(product.v2_m_s)-comparison['v2_m_s']))
        expected1, expected2 = physical[1][j]
        mp_difference = max(np.linalg.norm(np.array(product.v1_m_s)-np.array(list(map(float,expected1)))),
                            np.linalg.norm(np.array(product.v2_m_s)-np.array(list(map(float,expected2)))))
        normalized_r1,normalized_r2 = start['r']/length,stop['r']/length
        u1,u2 = np.array(product.v1_m_s)/speed,np.array(product.v2_m_s)/speed
        elapsed = duration/(length/speed)
        kr,kv,ksweep = kepler_propagate(normalized_r1,u1,elapsed)
        closure_r = np.linalg.norm(np.array(list(map(float,kr)))-normalized_r2)*length
        closure_v = np.linalg.norm(np.array(list(map(float,kv)))-u2)*speed
        ode_r,ode_v,ode_sweep = propagate_winding(normalized_r1,u1,normalized_r2,u2,elapsed,m,
                                                 product.geometry.geometry.transfer_angle_rad)
        expected_sweep = 2*math.pi*m+product.geometry.geometry.transfer_angle_rad
        if (gooding_difference>1e-6 or mp_difference>1e-6 or closure_r>10 or closure_v>1e-6
                or abs(float(ksweep)-expected_sweep)>2e-8):
            raise AssertionError('Juno product velocity or independent propagation discrepancy')
        entry = dict(branch=product.branch.value, x=product.root.x,
                     v1_m_s=product.v1_m_s, v2_m_s=product.v2_m_s,
                     iterations=product.root.iterations, relative_residual=product.root.relative_residual,
                     precision_limited=product.root.x_precision_limited,
                     estimated_x_error=product.root.estimated_x_error,
                     gooding_velocity_difference_m_s=float(gooding_difference),
                     multiprecision_velocity_difference_m_s=float(mp_difference),
                     kepler_position_closure_m=float(closure_r),kepler_velocity_closure_m_s=float(closure_v),
                     ode_position_closure_m=ode_r*length,ode_velocity_closure_m_s=ode_v*speed,
                     propagated_angle_degrees=math.degrees(float(ksweep)),
                     ode_propagated_angle_degrees=math.degrees(ode_sweep))
        product_records.append(entry)
        print('Complete product branch:',json.dumps(entry),flush=True)
    for count in (2,3):
        for product in solve_multi_rev_pair(replace(problem,revolutions=count)):
            if product.root.status!=MultiRootStatus.NO_SOLUTION or product.v1_m_s is not None:
                raise AssertionError('Juno infeasible count fabricated a velocity')
    result = dict(mission='Juno', target=-61, center=599, frame='ICRF', time_scale='TDB',
                  start=START, stop=STOP, duration_days=duration/86400, mu_m3_s2=mu,
                  coarse_samples=len(coarse), refined_samples=len(fine), unique_samples=len(rows),
                  observed_projected_sweep_degrees=math.degrees(sweep), observed_revolutions=m,
                  maximum_sample_angle_degrees=math.degrees(max(increments)),
                  minimum_projection_ratio=float(min(projection_ratio)), feasibility=feasibility,
                  reference_branches=branches, product_branches=product_records,
                  limitation='Differences from spacecraft ephemeris '
                  'velocities include non-Keplerian gravity and maneuvers and are not numerical error tests.')
    print('Juno complete multi-revolution regression passed.', flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--refresh', action='store_true')
    parser.add_argument('--json', type=Path)
    args = parser.parse_args()
    if args.refresh:
        refresh()
    result = validate()
    if args.json:
        args.json.write_text(json.dumps(result, indent=2)+'\n')
