"""Versioned configuration for a complete patched-conic mission search."""
from dataclasses import dataclass, field, asdict
import json
import math
from pathlib import Path


def real(x, name, positive=False):
    if type(x) not in (int, float) or not math.isfinite(x) or (positive and x <= 0):
        raise ValueError(f'{name} must be a finite {"positive " if positive else ""}real number')
    return float(x)


def interval(value, name, positive=False):
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError(f'{name} must contain lower and upper bounds')
    lo, hi = (real(x, name, positive) for x in value)
    if lo > hi:
        raise ValueError(f'{name} bounds are reversed')
    return [lo, hi]


@dataclass
class MissionConfig:
    schema_version: int = 1
    name: str = 'EMC research mission'
    launch_window: list = field(default_factory=lambda: ['2034-01-01', '2038-12-31'])
    visits: list = field(default_factory=list)
    tof_days: list = field(default_factory=list)
    max_duration_days: float = 3652.5
    orientation: str = 'both'
    max_revolutions: int = 2
    model: str = 'ballistic'
    backend: str = 'cpu'
    frame: str = 'ECLIPJ2000'
    time_scale: str = 'TDB'
    kernels: list = field(default_factory=list)
    launch_c3_limit_km2_s2: float | None = None
    minimum_solar_radius_m: float = 29919574140.0  # explicit 0.2 AU search constraint
    speed_atol_m_s: float = 1e-5
    speed_rtol: float = 1e-12
    max_flyby_burn_m_s: float | None = None
    generations: int = 180
    popsize: int = 12
    seeds: list = field(default_factory=lambda: [9587, 2036, 4193])
    refine_starts: int = 12
    refine_iterations: int = 160
    batch_candidates: int = 64
    archive_size: int = 40
    violation_weight: float = 1000.0

    def validate(self):
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError('Expected mission schema_version 1')
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError('A mission name is required')
        if not isinstance(self.launch_window, list) or len(self.launch_window) != 2:
            raise ValueError('launch_window requires two TDB calendar dates')
        from datetime import datetime
        dates = [datetime.fromisoformat(s) for s in self.launch_window]
        if any(d.tzinfo is not None for d in dates) or dates[1] < dates[0]:
            raise ValueError('Use ordered timezone-free TDB launch dates')
        if self.frame not in ('J2000', 'ECLIPJ2000') or self.time_scale != 'TDB':
            raise ValueError('Mission states require J2000/ECLIPJ2000 and TDB')
        if self.backend not in ('cpu', 'cuda') or self.model not in ('ballistic', 'powered'):
            raise ValueError('Select an explicit backend and flyby model')
        if self.orientation not in ('both', 'prograde', 'retrograde', 'short', 'long'):
            raise ValueError('Invalid orientation policy')
        if type(self.max_revolutions) is not int or not 0 <= self.max_revolutions <= 5:
            raise ValueError('Planner max_revolutions must be 0..5')
        for name in ('generations', 'popsize', 'refine_starts', 'refine_iterations',
                     'batch_candidates', 'archive_size'):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f'{name} must be a positive integer')
        if self.popsize < 5 or self.batch_candidates > 4096 or self.archive_size > 1000:
            raise ValueError('Use popsize >= 5, batch_candidates <= 4096, archive_size <= 1000')
        if (not isinstance(self.seeds, list) or not self.seeds or
                any(type(s) is not int or not 0 <= s < 2**32 for s in self.seeds) or
                len(set(self.seeds)) != len(self.seeds)):
            raise ValueError('Use distinct unsigned 32-bit seeds')
        for name in ('max_duration_days', 'minimum_solar_radius_m', 'speed_atol_m_s',
                     'violation_weight'):
            real(getattr(self, name), name, True)
        if real(self.speed_rtol, 'speed_rtol') < 0:
            raise ValueError('Negative relative speed tolerance')
        for name in ('launch_c3_limit_km2_s2', 'max_flyby_burn_m_s'):
            if getattr(self, name) is not None:
                real(getattr(self, name), name, True)
        if not isinstance(self.visits, list) or not 2 <= len(self.visits) <= 10:
            raise ValueError('Use 2..10 mission visits')
        allowed = {'body', 'gm_body', 'kind', 'radius_m', 'parking_altitude_m',
                   'minimum_altitude_m', 'maximum_periapsis_m', 'stay_days', 'body_frame'}
        for i, visit in enumerate(self.visits):
            if not isinstance(visit, dict) or set(visit) - allowed:
                raise ValueError('Unknown visit fields')
            for name in ('body', 'gm_body'):
                if not isinstance(visit.get(name), str) or not visit[name].strip():
                    raise ValueError(f'Visit {i} requires {name}')
            if visit['body'] in [str(n) for n in range(10)]:
                raise ValueError('Use body-centre states, not planetary barycenters')
            kind = visit.get('kind')
            expected = ('departure',) if i == 0 else ('arrival',) if i == len(self.visits)-1 else ('flyby', 'surface')
            if kind not in expected:
                raise ValueError(f'Invalid visit kind at position {i}')
            radius = real(visit.get('radius_m'), 'radius_m', True)
            if kind in ('departure', 'arrival'):
                if real(visit.get('parking_altitude_m'), 'parking_altitude_m') < 0:
                    raise ValueError('Negative parking altitude')
            if kind == 'flyby':
                alt = real(visit.get('minimum_altitude_m'), 'minimum_altitude_m')
                maximum = real(visit.get('maximum_periapsis_m'), 'maximum_periapsis_m', True)
                if alt < 0 or maximum < radius + alt:
                    raise ValueError('Invalid flyby radius limits')
            if kind == 'surface':
                interval(visit.get('stay_days'), 'stay_days', True)
                if not isinstance(visit.get('body_frame'), str) or not visit['body_frame']:
                    raise ValueError('A rotating surface visit requires a body-fixed frame')
        if len(self.tof_days) != len(self.visits)-1:
            raise ValueError('One TOF interval is required per leg')
        for bounds in self.tof_days:
            interval(bounds, 'tof_days', True)
        minimum = sum(b[0] for b in self.tof_days) + sum(v['stay_days'][0] for v in self.visits if v['kind'] == 'surface')
        if minimum > self.max_duration_days:
            raise ValueError('Minimum leg/stay durations exceed the mission duration limit')
        if not isinstance(self.kernels, list) or not self.kernels or any(not isinstance(s, str) for s in self.kernels):
            raise ValueError('Explicit kernel files are required')
        if len(set(self.kernels)) != len(self.kernels):
            raise ValueError('Duplicate kernels')
        return self

    def to_dict(self):
        return asdict(self)


def load_mission(path):
    path = Path(path).expanduser().resolve()
    data = json.loads(path.read_text())
    if not isinstance(data, dict) or set(data) - set(MissionConfig.__dataclass_fields__):
        raise ValueError('Unknown mission configuration fields')
    config = MissionConfig(**data)
    config.kernels = [str((path.parent / p).resolve()) for p in config.kernels]
    return config.validate()
