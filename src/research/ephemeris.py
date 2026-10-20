"""Geometric heliocentric SPICE states, normalized once to SI and cached by hash."""
import hashlib
import json
from pathlib import Path
import numpy as np


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            value.update(block)
    return value.hexdigest()


class SpiceEphemeris:
    def __init__(self, kernels, frame, cache_dir):
        import spiceypy as sp
        self.sp, self.frame = sp, frame
        self.cache_dir = Path(cache_dir)
        self.owned, self.provenance = [], []
        self.calls = self.cache_hits = 0
        kernels = [str(Path(p).resolve()) for p in kernels]
        if len(set(kernels)) != len(kernels):
            raise ValueError('Duplicate SPICE kernel paths')
        if not kernels:
            raise ValueError('No kernels configured. Supply explicit LSK, SPK and GM kernel paths.')
        try:
            loaded = {str(Path(sp.kdata(i,'ALL')[0]).resolve()) for i in range(sp.ktotal('ALL'))}
            # A standalone run must not inherit unrecorded kernels or priority order.
            if loaded:
                raise ValueError('SPICE kernels are already loaded; use a clean process to preserve kernel ordering')
            for name in kernels:
                path = Path(name).resolve()
                if path.suffix.lower() not in ('.tls','.bsp','.tpc'):
                    raise ValueError('List direct .tls/.bsp/.tpc kernels; metakernels are not supported')
                self.provenance.append(dict(path=str(path), sha256=sha256(path)))
                if str(path) not in loaded:
                    sp.furnsh(str(path)); self.owned.append(str(path))
                    loaded.add(str(path))
            self.mu = float(sp.bodvrd('SUN','GM',1)[1][0])*1e9
            if not np.isfinite(self.mu) or self.mu <= 0:
                raise ValueError('Invalid solar GM in the configured kernels')
        except Exception as error:
            self.close()
            if isinstance(error,(ValueError,OSError)):raise
            raise ValueError(f'Cannot initialize the configured SPICE kernels: {error}') from error

    def epoch(self, calendar):
        # Explicit label prevents SPICE's default UTC interpretation.
        return float(self.sp.str2et(calendar.replace('T',' ')+' TDB'))

    def states(self, target, epochs):
        epochs = np.asarray(epochs, dtype='f8')
        if epochs.ndim != 1 or not np.isfinite(epochs).all():
            raise ValueError('Expected finite one-dimensional TDB epochs')
        header = dict(schema=1, spiceypy=self.sp.__version__, toolkit=self.sp.tkvrsn('TOOLKIT'),
                      target=str(target), frame=self.frame, center='SUN',
                      aberration='NONE', kernels=self.provenance)
        key = hashlib.sha256(json.dumps(header,sort_keys=True).encode()+epochs.tobytes()).hexdigest()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        path = self.cache_dir/f'{key}.npz'
        if path.exists():
            try:
                with np.load(path,allow_pickle=False) as saved:
                    values = saved['states']
                    if (values.shape != (len(epochs),6) or values.dtype != np.dtype('f8')
                        or not np.isfinite(values).all() or not np.array_equal(saved['et'],epochs)
                        or str(saved['sha256']) != hashlib.sha256(values.tobytes()).hexdigest()):
                        raise ValueError('State checksum or layout mismatch')
                    self.cache_hits += len(epochs)
                    return values.copy()
            except (OSError,KeyError,ValueError) as error:
                raise ValueError(f'Corrupt ephemeris cache: {path}; remove it and rerun') from error
        values = np.empty((len(epochs),6))
        for i, et in enumerate(epochs):
            try:
                values[i] = np.asarray(self.sp.spkezr(str(target),float(et),self.frame,'NONE','SUN')[0])*1000.
            except Exception as error:
                raise ValueError(f'No usable state for body {target} at ET={et:.9f}; check kernel coverage: {error}') from error
        if not np.isfinite(values).all():
            raise ValueError('SPICE returned nonfinite states')
        self.calls += len(epochs)
        import tempfile
        with tempfile.NamedTemporaryFile(dir=self.cache_dir,suffix='.npz',delete=False) as temporary:
            pending = Path(temporary.name)
        try:
            np.savez(pending,et=epochs,states=values,sha256=hashlib.sha256(values.tobytes()).hexdigest())
            pending.replace(path)
        finally:
            pending.unlink(missing_ok=True)
        return values

    def close(self):
        for path in reversed(self.owned):
            self.sp.unload(path)
        self.owned.clear()

    def __enter__(self): return self
    def __exit__(self,*args): self.close()
