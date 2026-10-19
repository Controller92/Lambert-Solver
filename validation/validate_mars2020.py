"""Offline reproducible Mars 2020 checks against cached JPL Horizons states.

Use --refresh to retrieve public data sequentially from Horizons. Networking,
time conversions and ephemeris parsing stay outside the mathematical runtime.
Sun-only Lambert closure and differences from the real spacecraft ephemeris
are reported separately: the latter includes planetary gravity and maneuvers.
"""
import argparse
import csv
from datetime import datetime, timezone
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

import numpy as np
from lamberthub import gooding1990
from scipy.integrate import solve_ivp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.lambert_geometry import LambertInput
from src.lambert_solver import LambertStatus, solve_zero_rev

DATA = ROOT/'validation'/'data'/'mars2020'
API = 'https://ssd.jpl.nasa.gov/api/horizons.api'
EPOCHS_UTC = ('2020-07-30 11:50:00', '2020-07-30 13:00:00',
              '2020-08-01 00:00:00', '2020-08-15 00:00:00',
              '2020-09-15 00:00:00', '2020-10-15 00:00:00',
              '2020-12-01 00:00:00', '2021-01-15 00:00:00',
              '2021-02-17 00:00:00', '2021-02-18 20:36:50')


def query(name, parameters):
    url = API+'?'+urlencode({'format': 'json', **parameters})
    with urlopen(url, timeout=45) as response:
        payload = json.load(response)
    if 'error' in payload or 'result' not in payload:
        raise RuntimeError(f'Horizons query failed: {name}: {payload}')
    result = payload['result']
    snapshot = dict(retrieved_utc=datetime.now(timezone.utc).isoformat(),
                    url=url, parameters=parameters,
                    result_sha256=hashlib.sha256(result.encode()).hexdigest(),
                    response=payload)
    DATA.mkdir(parents=True, exist_ok=True)
    (DATA/f'{name}.json').write_text(json.dumps(snapshot, indent=2)+'\n')


def refresh():
    query('sun', dict(COMMAND="'10'", OBJ_DATA="'YES'", MAKE_EPHEM="'NO'"))
    # JPL asks clients to submit one request at a time. These are sequential.
    for name, target in (('earth', '399'), ('mars', '499'), ('spacecraft', '-168')):
        epochs = EPOCHS_UTC[1:] if name == 'spacecraft' else EPOCHS_UTC
        query(name, dict(COMMAND=f"'{target}'", OBJ_DATA="'YES'", MAKE_EPHEM="'YES'",
                         EPHEM_TYPE="'VECTORS'", CENTER="'500@10'", REF_PLANE="'FRAME'",
                         REF_SYSTEM="'ICRF'", OUT_UNITS="'KM-S'", VEC_TABLE="'2'",
                         VEC_CORR="'NONE'", CSV_FORMAT="'YES'", VEC_DELTA_T="'YES'",
                         TIME_TYPE="'UT'", TLIST=' '.join(f"'{t}'" for t in epochs)))


def load(name):
    snapshot = json.loads((DATA/f'{name}.json').read_text())
    text = snapshot['response']['result']
    if hashlib.sha256(text.encode()).hexdigest() != snapshot['result_sha256']:
        raise ValueError(f'Cached Horizons text hash mismatch: {name}')
    return text


def states(name, target, expected_epochs):
    text = load(name)
    header = text.split('$$SOE')[0]
    for required in ('Center body name: Sun (10)', 'Reference frame : ICRF',
                     'Output units    : KM-S', 'GEOMETRIC cartesian states', 'JDUT', 'delta-T'):
        if required not in header:
            raise ValueError(f'Unexpected Horizons convention in {name}: {required}')
    target_line = next(line for line in header.splitlines() if line.startswith('Target body name:'))
    if f'({target})' not in target_line:
        raise ValueError(f'Unexpected target: {target_line}')
    rows = []
    for fields in csv.reader(io.StringIO(text.split('$$SOE')[1].split('$$EOE')[0])):
        if not fields:
            continue
        if len(fields) != 10:
            raise ValueError(f'Unexpected vector columns in {name}: {fields}')
        jd_ut = Decimal(fields[0].strip())
        delta_t = Decimal(fields[2].strip())  # Horizons states this is TDB-UT.
        calendar = datetime.strptime(fields[1].strip(), 'A.D. %Y-%b-%d %H:%M:%S.%f')
        elapsed = calendar-datetime(2000, 1, 1, 12)
        utc_seconds = (Decimal(2451545*86400+elapsed.days*86400+elapsed.seconds)
                       + Decimal(elapsed.microseconds)/Decimal(1000000))
        # Use calendar precision: the exported JD has only nine fractional
        # digits. Check that the redundant time tags agree before conversion.
        if abs(utc_seconds/Decimal(86400)-jd_ut) > Decimal('0.0000000006'):
            raise ValueError(f'Inconsistent Horizons time tags in {name}')
        # The fixed 2020-2021 interval has no leap step. Add Horizons' measured
        # TDB-UTC offset to obtain continuous dynamical intervals.
        tdb_seconds = utc_seconds+delta_t
        r = np.array([float(v)*1000 for v in fields[3:6]])
        v = np.array([float(v)*1000 for v in fields[6:9]])
        if not np.isfinite([r, v]).all():
            raise ValueError(f'Nonfinite state in {name}')
        rows.append(dict(tdb_seconds=tdb_seconds, calendar_utc=fields[1].strip(),
                         requested_utc=calendar.strftime('%Y-%m-%d %H:%M:%S'), r=r, v=v))
    if (tuple(row['requested_utc'] for row in rows) != tuple(expected_epochs)
            or any(b['tdb_seconds'] <= a['tdb_seconds'] for a, b in zip(rows, rows[1:]))):
        raise ValueError(f'Unexpected epochs/order in {name}')
    print(f'{name}: {target_line}; rows={len(rows)}')
    return rows


def transfer(label, start, stop, mu, prograde=True):
    r1, r2 = start['r'], stop['r']
    tof = float(stop['tdb_seconds']-start['tdb_seconds'])
    long_way = bool((np.cross(r1, r2)[2] < 0) == prograde)
    result = solve_zero_rev(LambertInput(tuple(r1), tuple(r2), tof, mu, long_way))
    if result.status != LambertStatus.OK or result.root.x_precision_limited is not False:
        raise AssertionError(f'{label} solve/precision failure: {result}')
    length = float(np.linalg.norm(r1))
    speed = math.sqrt(mu/length)
    time_scale = length/speed
    duration = tof/time_scale
    u1, u2 = np.array(result.v1_m_s)/speed, np.array(result.v2_m_s)/speed
    gv1, gv2, iterations, _ = gooding1990(
        1., r1/length, r2/length, duration, prograde=prograde,
        maxiter=100, atol=1e-13, rtol=1e-13, full_output=True)
    difference = max(np.linalg.norm(u1-gv1), np.linalg.norm(u2-gv2))*speed
    if iterations >= 100 or not math.isfinite(difference) or difference > 1e-6:
        raise AssertionError(f'{label} independent solver discrepancy: {difference} m/s')

    def rhs(t, state):
        r = state[:3]
        return np.concatenate((state[3:], -r/np.linalg.norm(r)**3))

    propagated = solve_ivp(rhs, (0., duration), np.concatenate((r1/length, u1)),
                           method='DOP853', rtol=3e-14, atol=3e-15,
                           max_step=duration/256, dense_output=True)
    if not propagated.success:
        raise AssertionError(f'{label} ODE failure: {propagated.message}')
    position_error = np.linalg.norm(propagated.y[:3, -1]*length-r2)
    velocity_error = np.linalg.norm(propagated.y[3:, -1]*speed-np.array(result.v2_m_s))
    if position_error > 10 or velocity_error > 1e-6:
        raise AssertionError(f'{label} two-body closure failed: {position_error} m, {velocity_error} m/s')
    print(f'{label}: tof={tof:.6f} s ({tof/86400:.9f} d), long_way={long_way}, '
          f'root_updates={result.root.iterations}, relative_residual={result.root.relative_residual:.6g}')
    print(f'  Gooding max endpoint velocity difference={difference:.9g} m/s; '
          f'propagation closure={position_error:.9g} m, {velocity_error:.9g} m/s')
    print(f'  departure velocity={result.v1_m_s} m/s; arrival velocity={result.v2_m_s} m/s')
    return result, propagated.sol, length, speed, time_scale


def validate():
    sun = load('sun')
    match = re.search(r'GM,\s*km\^3/s\^2\s*=\s*([0-9.E+\-]+)', sun)
    if not match:
        raise ValueError('Solar GM missing from cached Horizons Sun object data')
    mu = float(match.group(1))*1e9
    print(f'Solar GM from Horizons: {mu:.17g} m^3/s^2')
    earth = states('earth', '399', EPOCHS_UTC)
    mars = states('mars', '499', EPOCHS_UTC)
    spacecraft = states('spacecraft', '-168', EPOCHS_UTC[1:])
    if any(a['tdb_seconds'] != b['tdb_seconds'] for a, b in zip(earth, mars)):
        raise ValueError('Planet epochs are not identical')
    if any(a['tdb_seconds'] != b['tdb_seconds'] for a, b in zip(earth[1:], spacecraft)):
        raise ValueError('Spacecraft epochs are not identical to planet epochs')

    result, _, _, _, _ = transfer('Earth center at launch -> Mars center at entry', earth[0], mars[-1], mu)
    departure_excess = np.linalg.norm(np.array(result.v1_m_s)-earth[0]['v'])/1000
    arrival_excess = np.linalg.norm(np.array(result.v2_m_s)-mars[-1]['v'])/1000
    print(f'  Idealized planet-center v_infinity: departure={departure_excess:.9g} km/s, '
          f'arrival={arrival_excess:.9g} km/s; C3={departure_excess**2:.9g} km^2/s^2')

    # Cruise comparison avoids the launch ascent and atmospheric entry. The
    # selected states are Aug 1 through Feb 17, not Earth/Mars center states.
    cruise = spacecraft[1:-1]
    result, solution, length, speed, time_scale = transfer('Spacecraft cruise Aug 1 -> Feb 17', cruise[0], cruise[-1], mu)
    differences = []
    for state in cruise:
        elapsed = float(state['tdb_seconds']-cruise[0]['tdb_seconds'])/time_scale
        fitted = solution(elapsed)
        position = float(np.linalg.norm(fitted[:3]*length-state['r']))
        velocity = float(np.linalg.norm(fitted[3:]*speed-state['v']))
        differences.append((position, velocity))
        print(f'  {state["calendar_utc"]}: two-body fit minus spacecraft state: '
              f'|dr|={position/1000:.9g} km, |dv|={velocity:.9g} m/s')
    print(f'Cruise sample maximum ephemeris differences: {max(v[0] for v in differences)/1000:.9g} km, '
          f'{max(v[1] for v in differences):.9g} m/s')

    # A shorter interior interval shows the difference in a less strongly
    # perturbed part of cruise, without changing the model to fit observations.
    middle = cruise[3:5]  # Oct 15 -> Dec 1
    result, _, _, _, _ = transfer('Spacecraft mid-cruise Oct 15 -> Dec 1', middle[0], middle[-1], mu)
    print(f'  Mid-cruise endpoint differences from spacecraft velocity: '
          f'{np.linalg.norm(np.array(result.v1_m_s)-middle[0]["v"]):.9g}, '
          f'{np.linalg.norm(np.array(result.v2_m_s)-middle[-1]["v"]):.9g} m/s')
    print('Ephemeris differences are model comparisons, not numerical-error acceptance tests.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--refresh', action='store_true', help='replace cached public Horizons snapshots')
    if parser.parse_args().refresh:
        refresh()
    validate()
