"""Checked NumPy bindings for the independent FP64 C++ N-body library.

Ephemerides are immutable cubic Hermite position/velocity tables, supplied by
host orchestration. Neither SPICE nor a Python callback enters the integrator.
"""
import ctypes as ct
from dataclasses import dataclass, asdict
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
STATUS={0:'SUCCESS',1:'INVALID_INPUT',2:'EPHEMERIS_COVERAGE',3:'COLLISION',4:'STEP_FLOOR',5:'STEP_BUDGET',6:'NONFINITE'}
D=ct.POINTER(ct.c_double)


@dataclass
class Options:
    rtol: float=2e-13
    atol_position: float=1e-4
    atol_velocity: float=1e-8
    atol_stm: float=1e-11
    initial_step: float=10.
    minimum_step: float=1e-7
    maximum_step: float=300.
    maximum_steps: int=1000000

    def validate(self):
        for name,value in asdict(self).items():
            if name=='maximum_steps':
                if type(value) is not int or not 1<=value<2**63:raise ValueError('Invalid maximum_steps')
            elif type(value) not in (int,float) or not np.isfinite(value) or value<=0:
                raise ValueError('Invalid '+name)
        if self.rtol<2e-15 or self.minimum_step>self.maximum_step:raise ValueError('Invalid step/tolerance limits')
        return self


class COptions(ct.Structure):
    _fields_=[(n,ct.c_double) for n in list(Options.__dataclass_fields__)[:-1]]+[('maximum_steps',ct.c_int64)]


class CResult(ct.Structure):
    _fields_=[('state',ct.c_double*6),('stm',ct.c_double*36),('time',ct.c_double),('smallest_distance',ct.c_double),
              ('accepted',ct.c_int64),('rejected',ct.c_int64),('evaluations',ct.c_int64),('status',ct.c_int32)]


def array(value,shape=None):
    a=np.ascontiguousarray(value,dtype='f8')
    if (shape is not None and a.shape!=shape) or not np.isfinite(a).all():raise ValueError('Invalid finite array shape')
    return a


def pointer(a):return a.ctypes.data_as(D)


class NBodyContext:
    def __init__(self,times,states,mu,radii,j2=None):
        self.handle=None
        times=array(times);mu=array(mu);radii=array(radii)
        if times.ndim!=1 or len(times)<2 or (np.diff(times)<=0).any():raise ValueError('Ordered ephemeris knots required')
        if mu.ndim!=1 or not 1<=len(mu)<=128 or (mu<=0).any() or radii.shape!=mu.shape or (radii<0).any():raise ValueError('Invalid body constants')
        states=array(states,(len(times),len(mu),6))
        self.lib=ct.CDLL(str(ROOT/'build/native/libnbody_cpu.so'))
        self.lib.nbody_abi_version.restype=ct.c_int32
        if self.lib.nbody_abi_version()!=2:raise RuntimeError('N-body ABI mismatch')
        if self.lib.nbody_options_size()!=ct.sizeof(COptions) or self.lib.nbody_result_size()!=ct.sizeof(CResult):
            raise RuntimeError('N-body structure layout mismatch')
        self.lib.nbody_create.argtypes=[ct.c_int64,ct.c_int32,D,D,D,D,ct.POINTER(ct.c_int32)]
        self.lib.nbody_create.restype=ct.c_void_p
        self.lib.nbody_destroy.argtypes=[ct.c_void_p]
        self.lib.nbody_force.argtypes=[ct.c_void_p,ct.c_double,D,D,D]
        self.lib.nbody_force.restype=ct.c_int32
        self.lib.nbody_propagate.argtypes=[ct.c_void_p,ct.c_double,ct.c_double,D,ct.POINTER(COptions),ct.POINTER(CResult)]
        self.lib.nbody_propagate_centered.argtypes=[ct.c_void_p,ct.c_double,ct.c_double,D,D,ct.POINTER(COptions),ct.POINTER(CResult)]
        status=ct.c_int32()
        self.handle=self.lib.nbody_create(len(times),len(mu),pointer(times),pointer(states),pointer(mu),pointer(radii),ct.byref(status))
        if not self.handle:raise ValueError('N-body context: '+STATUS.get(status.value,'unknown failure'))

        if j2 is not None:
            body,coefficient,radius,poles=j2
            if type(body) is not int or not 0<=body<len(mu) or not np.isfinite([coefficient,radius]).all() or radius<=0:
                self.close();raise ValueError('Invalid J2 constants')
            poles=array(poles,(len(times),3))
            self.lib.nbody_set_j2.argtypes=[ct.c_void_p,ct.c_int32,ct.c_double,ct.c_double,D]
            self.lib.nbody_set_j2.restype=ct.c_int32
            status=self.lib.nbody_set_j2(self.handle,body,float(coefficient),float(radius),pointer(poles))
            if status:self.close();raise ValueError('N-body J2: '+STATUS[status])

    def _open(self):
        if not self.handle:raise RuntimeError('N-body context is closed')

    def force(self,time,position):
        self._open();r=array(position,(3,));acc=np.empty(3);G=np.empty((3,3))
        status=self.lib.nbody_force(self.handle,float(time),pointer(r),pointer(acc),pointer(G))
        if status:raise ValueError('N-body force: '+STATUS[status])
        return acc,G

    def propagate(self,start,end,state,options=None,check=True,origin=None):
        self._open();state=array(state,(6,));o=(options or Options()).validate();result=CResult()
        arguments=(ct.byref(COptions(**asdict(o))),ct.byref(result))
        if origin is None:self.lib.nbody_propagate(self.handle,float(start),float(end),pointer(state),*arguments)
        else:
            origin=array(origin,(3,))
            self.lib.nbody_propagate_centered(self.handle,float(start),float(end),pointer(state),pointer(origin),*arguments)
        out={n:getattr(result,n) for n in ['status','time','smallest_distance','accepted','rejected','evaluations']}
        out.update(state=np.array(result.state),stm=np.array(result.stm).reshape(6,6))
        if check and result.status:raise RuntimeError('N-body propagation: '+STATUS[result.status])
        return out

    def close(self):
        if self.handle:self.lib.nbody_destroy(self.handle);self.handle=None
    def __enter__(self):self._open();return self
    def __exit__(self,*args):self.close()
    def __del__(self):self.close()
