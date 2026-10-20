"""Validated NumPy/ctypes interface to the custom flyby CPU/CUDA core.

Independent ABI; explicit backend, no fallback. CUDA contexts hold device memory
across upload/run/download cycles. A context serializes access with an RLock.
"""
import ctypes as ct
from pathlib import Path
from threading import RLock
from numbers import Integral, Real
import numpy as np
from .flyby import Encounter, Mode

INPUT_DTYPE=np.dtype([(n,'f8',3) for n in ('vin','vout','reference')]
    +[(n,'f8') for n in ('mu','rp_min','rp_max','rp','azimuth','speed_atol','speed_rtol')]
    +[('mode','i4'),('reserved','i4')],align=True)
RESULT_DTYPE=np.dtype([(n,'i4') for n in ('status','iterations','precision_limited','reserved')]
    +[(n,'f8',3) for n in ('vout','b','rperi','vperi_in','vperi_out')]
    +[(n,'f8') for n in ('rp','angle','bmag','speed_in','target_speed','speed_error',
                         'delta_v','angle_residual','lower','upper','azimuth')],align=True)
class _Abi(ct.Structure):
    _fields_=[(n,ct.c_uint32) for n in ('version','input_size','result_size')]


class _Info(ct.Structure):
    _fields_=[('name',ct.c_char*256)]+[(n,ct.c_int32) for n in (
        'major','minor','driver','runtime','multiprocessors','registers','local_bytes',
        'active_blocks','threads_per_sm','max_threads_per_block')]


def pack_encounters(encounters):
    encounters=list(encounters)
    rows=np.zeros(len(encounters),dtype=INPUT_DTYPE)
    for i,p in enumerate(encounters):
        if not isinstance(p,Encounter): raise TypeError('Expected Encounter')
        for name in ('vin','vout','reference'):
            v=getattr(p,name)
            if np.iscomplexobj(v): raise TypeError('Vectors must be real')
            if np.asarray(v).dtype.kind not in 'iuf': raise TypeError('Vector components must be real numbers')
            if np.asarray(v).shape!=(3,): raise ValueError('Vectors must have three components')
            rows[name][i]=v
        for name in ('mu','rp_min','rp_max','rp','azimuth','speed_atol','speed_rtol'):
            value=getattr(p,name)
            if isinstance(value,bool) or not isinstance(value,Real): raise TypeError('Scalars must be real numbers')
            rows[name][i]=value
        if isinstance(p.mode,bool) or not isinstance(p.mode,Integral): raise TypeError('Expected an integer Mode')
        rows['mode'][i]=int(Mode(p.mode))
    return rows


def _inputs(rows):
    if not isinstance(rows,np.ndarray) or rows.dtype!=INPUT_DTYPE or rows.ndim!=1:
        raise TypeError('Expected one-dimensional flyby INPUT_DTYPE array')
    return np.require(rows,requirements=['C','A'])


class FlybyEngine:
    def __init__(self,backend='cpu',library=None):
        if backend not in ('cpu','cuda'): raise ValueError('Backend must be cpu or cuda')
        self.backend=backend
        path=Path(library) if library else Path(__file__).resolve().parents[1]/'build/native'/f'libflyby_{backend}.so'
        self.library=path.resolve()
        if not self.library.is_file(): raise FileNotFoundError(f"Missing {self.library}; run native/build.py")
        self.lib=ct.CDLL(str(self.library))
        self.lib.fb_abi.argtypes=[]; self.lib.fb_abi.restype=_Abi
        a=self.lib.fb_abi()
        if (a.version,a.input_size,a.result_size)!=(1,INPUT_DTYPE.itemsize,RESULT_DTYPE.itemsize):
            raise RuntimeError('Flyby ABI mismatch; rebuild native libraries')
        self.lib.fb_cpu.argtypes=[ct.c_void_p,ct.c_void_p,ct.c_size_t]; self.lib.fb_cpu.restype=ct.c_int
        if backend=='cuda':
            specs={'info':[ct.c_int,ct.c_int,ct.POINTER(_Info)],'create':[ct.c_size_t,ct.c_int,ct.c_int,ct.POINTER(ct.c_void_p)],
                   'destroy':[ct.c_void_p], 'upload':[ct.c_void_p,ct.c_void_p,ct.c_size_t],
                   'run':[ct.c_void_p,ct.c_int,ct.POINTER(ct.c_float)],
                   'download':[ct.c_void_p,ct.c_void_p,ct.c_size_t]}
            for name,types in specs.items():
                f=getattr(self.lib,'fb_cuda_'+name)
                f.argtypes=types+[ct.c_char_p,ct.c_size_t]; f.restype=ct.c_int

    def device_info(self,device=0,threads=128):
        if self.backend!='cuda': raise RuntimeError('Load the CUDA library explicitly')
        if type(device)!=int or not 0<=device<2**31 or type(threads)!=int or not 1<=threads<=1024:
            raise ValueError('Invalid device or thread count')
        info=_Info();error=ct.create_string_buffer(512)
        if self.lib.fb_cuda_info(device,threads,ct.byref(info),error,len(error)):
            raise RuntimeError(error.value.decode())
        data={name:getattr(info,name) for name,_ in info._fields_}
        data['name']=info.name.decode()
        data['predicted_occupancy']=info.active_blocks*threads/info.threads_per_sm
        return data

    def cpu(self,rows):
        rows=_inputs(rows); result=np.empty(len(rows),dtype=RESULT_DTYPE)
        if self.lib.fb_cpu(rows.ctypes.data,result.ctypes.data,len(rows)):
            raise RuntimeError('Native flyby CPU call rejected its arguments')
        return result

    def cuda_batch(self,capacity,device=0,threads=128):
        if self.backend!='cuda': raise RuntimeError('Load the CUDA library explicitly')
        return CudaBatch(self,capacity,device,threads)


class CudaBatch:
    def __init__(self,engine,capacity,device,threads):
        if type(capacity)!=int or capacity<0 or capacity>np.iinfo(np.intp).max//RESULT_DTYPE.itemsize:
            raise ValueError('Invalid capacity')
        if type(device)!=int or not 0<=device<2**31 or type(threads)!=int or not 1<=threads<=1024:
            raise ValueError('Invalid device or thread count')
        self.engine=engine; self.capacity=capacity; self.count=0
        self.handle=ct.c_void_p(); self.lock=RLock(); self.ran=False
        self._call('create',capacity,device,threads,ct.byref(self.handle))

    def _call(self,name,*args):
        error=ct.create_string_buffer(512)
        code=getattr(self.engine.lib,'fb_cuda_'+name)(*args,error,len(error))
        if code: raise RuntimeError(f'CUDA flyby {name}: {error.value.decode()} ({code})')

    def _open(self):
        if not self.handle: raise RuntimeError('CUDA flyby context is closed')

    def upload(self,rows):
        rows=_inputs(rows)
        with self.lock:
            self._open()
            if len(rows)>self.capacity: raise ValueError('Batch exceeds capacity')
            self.ran=False; self.count=0
            self._call('upload',self.handle,rows.ctypes.data,len(rows)); self.count=len(rows)

    def run(self,repetitions=1):
        if type(repetitions)!=int or not 1<=repetitions<2**31: raise ValueError('Invalid repetitions')
        with self.lock:
            self._open(); self.ran=False; ms=ct.c_float()
            self._call('run',self.handle,repetitions,ct.byref(ms)); self.ran=True
            return ms.value

    def download(self):
        with self.lock:
            self._open()
            if not self.ran: raise RuntimeError('Run before downloading')
            result=np.empty(self.count,dtype=RESULT_DTYPE)
            self._call('download',self.handle,result.ctypes.data,self.count)
            return result

    def close(self):
        with self.lock:
            if self.handle:
                h=self.handle; self.handle=ct.c_void_p(); self._call('destroy',h)

    def __enter__(self): return self
    def __exit__(self,*args): self.close()
    def __del__(self):
        if getattr(self,'handle',None):
            try: self.close()
            except Exception: pass
