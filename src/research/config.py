"""Versioned grid specifications, including the original CLI's JSON format."""
from dataclasses import asdict, dataclass, field
from datetime import datetime
from decimal import Decimal, ROUND_FLOOR
import json
import math
from pathlib import Path


@dataclass
class GridConfig:
    dep_body: str = '399'
    arr_body: str = '4'
    dep_start: str = '2035-06-24'
    dep_end: str = '2035-06-26'
    dep_step: float = 1.
    tof_min: float = 180.
    tof_max: float = 210.
    tof_step: float = 1.
    kernels: list[str] = field(default_factory=list)
    frame: str = 'ECLIPJ2000'
    time_scale: str = 'TDB'
    backend: str = 'cuda'
    orientation: str = 'prograde'
    max_revolutions: int = 0
    batch_size: int = 65536
    max_cells: int = 10_000_000
    max_memory_mb: int = 2048
    cmin: float = 10.
    cmax: float = 100.
    tof_contour_step: float = 50.
    plot: bool = True

    def validate(self):
        for name in ('dep_body', 'arr_body', 'dep_start', 'dep_end'):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f'{name} must be a nonempty string')
        for name in ('dep_start', 'dep_end'):
            value = datetime.fromisoformat(getattr(self, name))
            if value.tzinfo is not None:
                raise ValueError('Use timezone-free calendar dates explicitly interpreted as TDB')
        start, end = datetime.fromisoformat(self.dep_start), datetime.fromisoformat(self.dep_end)
        if end < start:
            raise ValueError('Departure end precedes start')
        if self.time_scale != 'TDB':
            raise ValueError('This release uses TDB dates and fixed 86400-second flight days')
        if self.frame not in ('J2000', 'ECLIPJ2000'):
            raise ValueError('Frame must be J2000 or ECLIPJ2000')
        if self.backend not in ('cpu', 'cuda'):
            raise ValueError('Backend must be cpu or cuda')
        if self.orientation not in ('prograde', 'retrograde', 'both', 'short', 'long'):
            raise ValueError('Unknown orientation policy')
        for name in ('dep_step', 'tof_min', 'tof_max', 'tof_step', 'cmin', 'cmax', 'tof_contour_step'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
                raise ValueError(f'{name} must be a finite number')
        if min(self.dep_step, self.tof_min, self.tof_step, self.tof_contour_step) <= 0:
            raise ValueError('Steps and minimum flight time must be positive')
        if self.tof_max < self.tof_min or self.cmax <= self.cmin or self.cmin < 0:
            raise ValueError('Invalid flight-time or C3 range')
        for name in ('max_revolutions', 'batch_size', 'max_cells', 'max_memory_mb'):
            if type(getattr(self, name)) is not int:
                raise ValueError(f'{name} must be an integer')
        if not 0 <= self.max_revolutions <= 100:
            raise ValueError('Research grid max_revolutions must be between 0 and 100')
        if not 1 <= self.batch_size <= 1_048_576 or self.max_cells < 1 or self.max_memory_mb < 1:
            raise ValueError('Invalid batch size or grid limit')
        if type(self.plot) is not bool:
            raise ValueError('plot must be bool')
        if not isinstance(self.kernels, list) or not all(isinstance(p, str) and p for p in self.kernels):
            raise ValueError('kernels must be a list of explicit file paths')
        if len(set(self.kernels)) != len(self.kernels):
            raise ValueError('Duplicate kernel paths would make load ordering ambiguous')
        nd, nt = self.shape
        if nd*nt > self.max_cells:
            raise ValueError(f'Grid has {nd*nt:,} cells, exceeding max_cells={self.max_cells:,}')
        candidates = (1+2*self.max_revolutions)*(2 if self.orientation=='both' else 1)
        estimated_bytes = nd*nt*(160+5*candidates)
        if estimated_bytes > self.max_memory_mb*1024**2:
            raise ValueError('Grid diagnostics exceed max_memory_mb; reduce grid or revolution count')
        return self

    @property
    def shape(self):
        delta = datetime.fromisoformat(self.dep_end)-datetime.fromisoformat(self.dep_start)
        days = Decimal(delta.days)+Decimal(delta.seconds)/86400+Decimal(delta.microseconds)/86400000000
        return axis_count(0, days, self.dep_step), axis_count(self.tof_min, self.tof_max, self.tof_step)

    def to_dict(self):
        return asdict(self)


def axis_count(start, stop, step):
    return int(((Decimal(str(stop))-Decimal(str(start)))/Decimal(str(step))).to_integral_value(rounding=ROUND_FLOOR))+1


def from_dict(data, base=Path('.')):
    if not isinstance(data, dict):
        raise ValueError('Configuration must be a JSON object')
    if 'simulation_type' in data:
        if data['simulation_type'] != 'two_body_grid':
            raise ValueError('Only two_body_grid legacy configurations are supported in this release')
        if not isinstance(data.get('parameters'),dict):
            raise ValueError('Legacy configuration requires a parameters object')
        data = dict(data['parameters'])
        for old, new in (('dep_id','dep_body'), ('arr_id','arr_body')):
            if old in data:
                data[new] = str(data.pop(old))
        for name in ('dep_body_name','arr_body_name'):
            data.pop(name, None)
        for name in ('dep_step','tof_min','tof_max','tof_step'):
            if name in data:
                if isinstance(data[name], bool):raise ValueError(f'{name} must not be bool')
                data[name] = float(data[name])
    unknown = set(data)-set(GridConfig.__dataclass_fields__)
    if unknown:
        raise ValueError(f'Unknown configuration fields: {sorted(unknown)}')
    config = GridConfig(**data).validate()
    config.kernels = [str((base/Path(p).expanduser()).resolve()) for p in config.kernels]
    return config.validate()


def load_config(path):
    path = Path(path).resolve()
    return from_dict(json.loads(path.read_text()), path.parent)


def save_config(config, path):
    config.validate()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config.to_dict(), indent=2, allow_nan=False)+'\n')
