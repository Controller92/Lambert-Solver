"""NumPy/ctypes batch interface to the shared C++ and CUDA Izzo engine.

Build explicitly with native/build.py first. CUDA errors raise, with no CPU
fallback. Geometry/root failures remain individual output rows. Persistent
CudaBatch separates allocation, upload, repeated resident solves and download.
This module owns no ephemerides, dates or frame conversions.
"""
import ctypes as ct
from pathlib import Path
from threading import RLock

import numpy as np

from .lambert_geometry import LambertInput
from .lambert_root import RootOptions
from .lambert_multirev import MinimumOptions
from .lambert_multirev_root import RootBranch

INPUT_DTYPE = np.dtype([('r1', 'f8', 3), ('r2', 'f8', 3), ('tof', 'f8'), ('mu', 'f8'),
                        ('long_way', 'i4'), ('revolutions', 'i4'), ('branch', 'i4')], align=True)
ROOT_DTYPE = np.dtype([(n, 'i4') for n in ('status', 'evaluation_status', 'precision_limited', 'fallback_guess')]
                     + [(n, 'f8') for n in ('x', 'y', 'last_x', 'time', 'residual', 'relative_residual',
                                           'last_step', 'lower', 'upper', 'estimated_x_error')]
                     + [(n, 'i8') for n in ('iterations', 'householder', 'bisection', 'verification',
                                           'rejected_householder', 'derivative_failures', 'expansions', 'evaluations')], align=True)
MINIMUM_DTYPE = np.dtype([(n, 'i4') for n in ('status', 'evaluation_status')]
    + [(n, 'f8') for n in ('x', 'time', 'estimated_time_error', 'first', 'lower', 'upper', 'first_lower', 'first_upper')]
    + [(n, 'i8') for n in ('iterations', 'halley', 'bisection', 'zero_derivative', 'expansions', 'evaluations')], align=True)
RESULT_DTYPE = np.dtype([('status', 'i4'), ('geometry_status', 'i4'), ('root', ROOT_DTYPE),
                         ('v1', 'f8', 3), ('v2', 'f8', 3), ('lambda', 'f8'), ('q', 'f8'), ('target', 'f8'), ('minimum', MINIMUM_DTYPE), ('branch', 'i4'), ('revolutions', 'i4')], align=True)
TOF_DTYPE = np.dtype([(n, 'i4') for n in ('status', 'method', 'terms', 'reserved')]
                    + [('time', 'f8'), ('y', 'f8')], align=True)
DERIVATIVE_DTYPE = np.dtype([(n, 'i4') for n in ('status', 'method', 'terms', 'reserved')]
                           + [(n, 'f8') for n in ('time', 'y', 'first', 'second', 'third')], align=True)
JACOBIAN_DTYPE = np.dtype([('status', 'i4'), ('reserved', 'i4'), ('root_condition', 'f8'),
                           ('slope_ratio', 'f8'), ('estimated_relative_slope_error', 'f8'),
                           ('values', 'f8', (6, 7))], align=True)


class _Options(ct.Structure):
    _fields_ = [(n, ct.c_double) for n in ('x_atol', 'x_rtol', 'tof_atol', 'tof_rtol')] + [
        ('max_iterations', ct.c_int32), ('max_bracket_expansions', ct.c_int32)]


class _MinimumOptions(ct.Structure):
    _fields_ = [('x_atol', ct.c_double), ('x_rtol', ct.c_double),
                ('max_iterations', ct.c_int32), ('max_bracket_expansions', ct.c_int32)]


class _Abi(ct.Structure):
    _fields_ = [(n, ct.c_uint32) for n in ('version', 'input_size', 'options_size', 'root_size',
                                         'result_size', 'tof_size', 'derivative_size')]


class _MultiAbi(ct.Structure):
    _fields_ = [(n, ct.c_uint32) for n in ('minimum_size', 'minimum_options_size')]


class _DeviceInfo(ct.Structure):
    _fields_ = [('name', ct.c_char*256)] + [(n, ct.c_int32) for n in (
        'major', 'minor', 'driver', 'runtime', 'multiprocessors', 'registers', 'local_bytes',
        'active_blocks', 'threads_per_sm', 'max_threads_per_block')]


def pack_problems(problems, branches=RootBranch.LEFT):
    """Pack fixed-shape inputs; invalid numerical values retain per-row status.

    Malformed Python types/shapes raise before native code sees a pointer.
    Native flags/revolution counts use signed int32; out-of-range counts raise.
    M>0 branches are explicitly left/right of the minimum, independent of the
    sign of x and long_way. A scalar branch is broadcast; a sequence is per row.
    M=0 ignores the branch selector.
    """
    problems = list(problems)
    if isinstance(branches, (str, RootBranch)):
        branches = [RootBranch(branches)]*len(problems)
    else:
        branches = [RootBranch(b) for b in branches]
    if len(branches) != len(problems):
        raise ValueError('Expected one branch per problem')
    rows = np.empty(len(problems), dtype=INPUT_DTYPE)
    for i, p in enumerate(problems):
        if not isinstance(p, LambertInput) or type(p.long_way) is not bool or type(p.revolutions) is not int:
            raise TypeError('Expected LambertInput with bool long_way and int revolutions')
        if isinstance(p.tof_s, bool) or isinstance(p.mu_m3_s2, bool):
            raise TypeError('Time and mu must be numeric, not bool')
        if not -(2**31) <= p.revolutions < 2**31:
            raise ValueError('Native revolution count must fit int32')
        if np.iscomplexobj(p.r1_m) or np.iscomplexobj(p.r2_m):
            raise TypeError('Position vectors must be real')
        r1, r2 = np.asarray(p.r1_m, dtype='f8'), np.asarray(p.r2_m, dtype='f8')
        if r1.shape != (3,) or r2.shape != (3,):
            raise ValueError('Position vectors must have exactly three components')
        rows[i] = (r1, r2, float(p.tof_s), float(p.mu_m3_s2), int(p.long_way), p.revolutions, int(branches[i] == RootBranch.RIGHT))
    return rows


def pack_multi_rev_pairs(problems):
    """Return interleaved (left,right) rows. Reshape results to (n,2).

    Each branch runs in its own GPU lane and computes its own minimum. This
    avoids coupling branch iteration lifetimes; minimum counters are per row.
    All problems must have a positive revolution count.
    """
    problems = list(problems)
    if any(not isinstance(p, LambertInput) for p in problems):
        raise TypeError('Expected LambertInput problems')
    if any(type(p.revolutions) is not int or p.revolutions < 1 for p in problems):
        raise ValueError('Pairs require positive revolutions')
    return pack_problems([p for p in problems for _ in range(2)],
                         [b for _ in problems for b in RootBranch])


def _minimum_options(options):
    if not isinstance(options, MinimumOptions):
        raise TypeError('Expected MinimumOptions')
    values = [getattr(options, n) for n, _ in _MinimumOptions._fields_]
    if any(isinstance(v, bool) for v in values):
        raise TypeError('Minimum options must not be bool')
    for value in values[2:]:
        if type(value) is not int or not -(2**31) <= value < 2**31:
            raise ValueError('Minimum iteration budgets must be int32 integers')
    return _MinimumOptions(*values)


def _inputs(rows):
    if not isinstance(rows, np.ndarray) or rows.dtype != INPUT_DTYPE or rows.ndim != 1:
        raise TypeError('Expected a one-dimensional INPUT_DTYPE array from pack_problems')
    return np.require(rows, dtype=INPUT_DTYPE, requirements=['C', 'A'])


def _options(options):
    if not isinstance(options, RootOptions):
        raise TypeError('Expected RootOptions')
    for name in ('max_iterations', 'max_bracket_expansions'):
        value = getattr(options, name)
        if type(value) is not int or not -(2**31) <= value < 2**31:
            raise ValueError('Native iteration budgets must be int32 integers')
    values = [getattr(options, n) for n, _ in _Options._fields_]
    if any(isinstance(v, bool) for v in values):
        raise TypeError('Root options must not be bool')
    return _Options(*values)


class NativeEngine:
    """Explicit library loading; binary ABI is checked before every backend use."""
    def __init__(self, backend='cpu', library=None):
        if backend not in ('cpu', 'cuda'):
            raise ValueError('backend must be cpu or cuda')
        path = Path(library) if library else Path(__file__).resolve().parents[1]/'build'/'native'/f'liblambert_{backend}.so'
        if not path.is_file():
            raise FileNotFoundError(f'Native library missing: {path}; run native/build.py')
        self.backend = backend
        self.lib = ct.CDLL(str(path.resolve()))
        self.lib.lb_abi.restype = _Abi
        abi = self.lib.lb_abi()
        expected = (2, INPUT_DTYPE.itemsize, ct.sizeof(_Options), ROOT_DTYPE.itemsize,
                    RESULT_DTYPE.itemsize, TOF_DTYPE.itemsize, DERIVATIVE_DTYPE.itemsize)
        if tuple(getattr(abi, n) for n, _ in _Abi._fields_) != expected:
            raise RuntimeError('Native ABI mismatch; rebuild both libraries')
        self.lib.lb_multi_abi.restype = _MultiAbi
        multi_abi = self.lib.lb_multi_abi()
        if (multi_abi.minimum_size, multi_abi.minimum_options_size) != (MINIMUM_DTYPE.itemsize, ct.sizeof(_MinimumOptions)):
            raise RuntimeError('Native minimum ABI mismatch; rebuild both libraries')
        for name in ('lb_cpu', 'lb_roots', 'lb_multi_roots'):
            function = getattr(self.lib, name)
            args = [ct.c_void_p, ct.c_void_p, ct.c_size_t, ct.POINTER(_Options)]
            if name != 'lb_roots':
                args.append(ct.POINTER(_MinimumOptions))
            function.argtypes, function.restype = args, ct.c_int
        for name in ('lb_tof', 'lb_derivatives', 'lb_multi_tof', 'lb_multi_derivatives'):
            function = getattr(self.lib, name)
            function.argtypes = [ct.c_void_p, ct.c_void_p, ct.c_size_t]
            function.restype = ct.c_int
        if backend == 'cuda':
            error_args = [ct.c_char_p, ct.c_size_t]
            signatures = {
                'lb_cuda_info': [ct.c_int, ct.c_int, ct.c_int, ct.POINTER(_DeviceInfo)],
                'lb_cuda_create': [ct.c_size_t, ct.c_int, ct.c_int, ct.POINTER(ct.c_void_p)],
                'lb_cuda_destroy': [ct.c_void_p],
                'lb_cuda_upload': [ct.c_void_p, ct.c_void_p, ct.c_size_t],
                'lb_cuda_run': [ct.c_void_p, ct.POINTER(_Options), ct.POINTER(_MinimumOptions), ct.c_int, ct.POINTER(ct.c_float)],
                'lb_cuda_download': [ct.c_void_p, ct.c_void_p, ct.c_size_t],
            }
            for name, args in signatures.items():
                function = getattr(self.lib, name)
                function.argtypes, function.restype = args+error_args, ct.c_int

    def cpu(self, rows, options=RootOptions(), out=None, *, minimum_options=MinimumOptions()):
        rows, opt = _inputs(rows), _options(options)
        output = np.empty(len(rows), dtype=RESULT_DTYPE) if out is None else out
        if (not isinstance(output, np.ndarray) or output.dtype != RESULT_DTYPE
                or output.shape != (len(rows),) or not output.flags.c_contiguous
                or not output.flags.aligned or not output.flags.writeable
                or np.shares_memory(rows, output)):
            raise ValueError('Output must be a separate writable, aligned contiguous RESULT_DTYPE array')
        if self.lib.lb_cpu(rows.ctypes.data, output.ctypes.data, len(rows), ct.byref(opt), ct.byref(_minimum_options(minimum_options))):
            raise RuntimeError('Invalid native batch arguments')
        return output

    def cpu_with_jacobian(self, rows, options=RootOptions(), *, minimum_options=MinimumOptions()):
        """Return (solutions, sensitivities) for fixed mu, direction and branch.

        values has rows [v1xyz,v2xyz], columns [r1xyz,r2xyz,tof]; SI units
        are s^-1 for position columns and m/s^2 for time. Check both statuses:
        Jacobian 0=valid, 1=velocity solve failed, 2=root ill-conditioned,
        3=nonfinite derivative arithmetic. Failure matrices contain NaNs.
        This is a local two-body derivative, not a date/mission-cost gradient.
        """
        self._jacobian_abi()
        rows, opt, mopt = _inputs(rows), _options(options), _minimum_options(minimum_options)
        output = np.empty(len(rows), dtype=RESULT_DTYPE)
        jacobian = np.empty(len(rows), dtype=JACOBIAN_DTYPE)
        function = self.lib.lb_cpu_jacobian
        function.argtypes = [ct.c_void_p, ct.c_void_p, ct.c_void_p, ct.c_size_t,
                             ct.POINTER(_Options), ct.POINTER(_MinimumOptions)]
        function.restype = ct.c_int
        if function(rows.ctypes.data, output.ctypes.data, jacobian.ctypes.data, len(rows),
                    ct.byref(opt), ct.byref(mopt)):
            raise RuntimeError('Invalid native Jacobian arguments')
        return output, jacobian

    def _jacobian_abi(self):
        try:
            function = self.lib.lb_jacobian_size
        except AttributeError as exc:
            raise RuntimeError('Native Jacobian unavailable; rebuild native libraries') from exc
        function.argtypes, function.restype = [], ct.c_uint32
        if function() != JACOBIAN_DTYPE.itemsize:
            raise RuntimeError('Native Jacobian ABI mismatch; rebuild native libraries')
        if self.backend == 'cuda':
            error_args = [ct.c_char_p, ct.c_size_t]
            self.lib.lb_cuda_run_jacobian.argtypes = [ct.c_void_p, ct.POINTER(_Options),
                ct.POINTER(_MinimumOptions), ct.c_int, ct.POINTER(ct.c_float)]+error_args
            self.lib.lb_cuda_run_jacobian.restype = ct.c_int
            self.lib.lb_cuda_download_jacobian.argtypes = [ct.c_void_p,ct.c_void_p,ct.c_size_t]+error_args
            self.lib.lb_cuda_download_jacobian.restype = ct.c_int

    def probe(self, kind, triples, options=RootOptions(), *, minimum_options=MinimumOptions()):
        """CPU validation probes. NaN q means infer q from lambda.

        M>0 rows contain [x/target, lambda, q, M, branch], where branch is 0/1.
        multi_roots returns RESULT_DTYPE with root/minimum only, no velocities."""
        types = {'roots': ROOT_DTYPE, 'tof': TOF_DTYPE, 'derivatives': DERIVATIVE_DTYPE,
                 'multi_roots': RESULT_DTYPE, 'multi_tof': TOF_DTYPE, 'multi_derivatives': DERIVATIVE_DTYPE}
        if kind not in types:
            raise ValueError('Unknown validation probe')
        rows = np.require(triples, dtype='f8', requirements=['C', 'A'])
        if rows.ndim != 2 or rows.shape[1] != (5 if kind.startswith('multi_') else 3):
            raise ValueError('Expected [x/target, lambda, q] rows, plus [M, branch] for multi probes')
        output = np.empty(len(rows), dtype=types[kind])
        args = [rows.ctypes.data, output.ctypes.data, len(rows)]
        if kind in ('roots', 'multi_roots'):
            opt = _options(options)
            args.append(ct.byref(opt))
        if kind == 'multi_roots':
            args.append(ct.byref(_minimum_options(minimum_options)))
        if getattr(self.lib, 'lb_'+kind)(*args):
            raise RuntimeError('Invalid native validation arguments')
        return output

    def _cuda_call(self, name, *args):
        if self.backend != 'cuda':
            raise RuntimeError('Load the CUDA backend explicitly')
        error = ct.create_string_buffer(512)
        code = getattr(self.lib, name)(*args, error, len(error))
        if code:
            raise RuntimeError(f'{name}: CUDA error {code}: {error.value.decode()}')

    def device_info(self, device=0, threads=128, *, multi=False):
        info = _DeviceInfo()
        self._cuda_call('lb_cuda_info', device, threads, int(multi), ct.byref(info))
        result = {n: getattr(info, n) for n, _ in info._fields_}
        result['name'] = info.name.decode()
        result['predicted_occupancy'] = info.active_blocks*threads/info.threads_per_sm
        return result

    def cuda_batch(self, capacity, device=0, threads=128):
        return CudaBatch(self, capacity, device, threads)


class CudaBatch:
    """Owned persistent buffers and stream. Use as a context manager.

    Calls on this object are serialized. Other CUDA users are never reset.
    download is legal only after a successful run following the latest upload.
    """
    def __init__(self, engine, capacity, device=0, threads=128):
        if type(capacity) is not int or capacity < 0 or capacity > np.iinfo(np.intp).max//RESULT_DTYPE.itemsize:
            raise ValueError('Invalid batch capacity')
        if type(device) is not int or type(threads) is not int or not 1 <= threads <= 1024:
            raise ValueError('Invalid device or block size')
        self.engine, self.capacity = engine, capacity
        self._handle, self._lock = ct.c_void_p(), RLock()
        self._count, self._ready, self._uploaded = 0, False, False
        self._jacobian_ready = False
        engine._cuda_call('lb_cuda_create', capacity, device, threads, ct.byref(self._handle))

    def _open(self):
        if not self._handle.value:
            raise RuntimeError('CUDA batch is closed')

    def upload(self, rows):
        rows = _inputs(rows)
        with self._lock:
            self._open()
            if len(rows) > self.capacity:
                raise ValueError('Batch exceeds device buffer capacity')
            self._ready = False
            self._jacobian_ready = False
            self._uploaded = False
            self.engine._cuda_call('lb_cuda_upload', self._handle, rows.ctypes.data, len(rows))
            self._count = len(rows)
            self._uploaded = True

    def run(self, options=RootOptions(), repetitions=1, *, minimum_options=MinimumOptions(), jacobian=False):
        """Return CUDA-event milliseconds per batch solve, excluding upload/download.

        A mixed M=0/M>0 solve includes both kernel launches. jacobian=True
        allocates an optional device buffer on first use, then differentiates
        the stored solutions in a separate kernel; timing includes that kernel.
        Allocation and transfers are excluded. Default solves need no Jacobian buffer."""
        opt, mopt = _options(options), _minimum_options(minimum_options)
        if type(jacobian) is not bool:
            raise TypeError('jacobian must be bool')
        if jacobian:
            self.engine._jacobian_abi()
        if type(repetitions) is not int or not 1 <= repetitions < 2**31:
            raise ValueError('repetitions must be a positive int32')
        with self._lock:
            self._open()
            if not self._uploaded:
                raise RuntimeError('Upload inputs before running')
            self._ready = False
            self._jacobian_ready = False
            milliseconds = ct.c_float()
            self.engine._cuda_call('lb_cuda_run_jacobian' if jacobian else 'lb_cuda_run',
                                  self._handle, ct.byref(opt), ct.byref(mopt), repetitions, ct.byref(milliseconds))
            self._ready = True
            self._jacobian_ready = jacobian
            return milliseconds.value

    def download_jacobian(self):
        """Return JACOBIAN_DTYPE rows after the latest run(jacobian=True)."""
        with self._lock:
            self._open()
            if not self._jacobian_ready:
                raise RuntimeError('Run the uploaded inputs with jacobian=True before downloading sensitivities')
            output = np.empty(self._count, dtype=JACOBIAN_DTYPE)
            self.engine._cuda_call('lb_cuda_download_jacobian',self._handle,output.ctypes.data,self._count)
            return output

    def download(self):
        with self._lock:
            self._open()
            if not self._ready:
                raise RuntimeError('Run the uploaded inputs before downloading')
            output = np.empty(self._count, dtype=RESULT_DTYPE)
            self.engine._cuda_call('lb_cuda_download', self._handle, output.ctypes.data, self._count)
            return output

    def close(self):
        with self._lock:
            if self._handle.value:
                handle, self._handle = self._handle, ct.c_void_p()
                self.engine._cuda_call('lb_cuda_destroy', handle)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def __del__(self):
        if getattr(self, '_handle', None) and self._handle.value:
            try:
                self.close()
            except Exception:
                pass  # Destructors cannot propagate; explicit close reports errors.
