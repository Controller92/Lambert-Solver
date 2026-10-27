"""Independent terminal Jacobian checks; no production derivative reference.

Use DOP853 integration of the Newtonian flow and variational equations, plus
100-digit Lagrange-anomaly inversion and central differences of full velocities.
CUDA opt-in runs the same physical corpus on the actual device.
"""
import argparse
from dataclasses import replace
import json
import math
from pathlib import Path
import sys
import time
import mpmath as mp
import numpy as np
from scipy.integrate import solve_ivp

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.lambert_geometry import LambertInput
from src.lambert_native import NativeEngine,pack_problems


def lagrange(x,l,m):
    if x==1: return mp.mpf(2)/3*(1-l**3)
    d=abs(1-x*x)
    # Independent anomaly equation, increasing precision near its cancellation.
    with mp.workdps(mp.mp.dps+max(0,int(-mp.log10(d)))+20):
        if x<1:
            a=2*mp.acos(x);b=2*mp.asin(l*mp.sqrt(d))
            return ((a-mp.sin(a))-(b-mp.sin(b))+2*m*mp.pi)/(2*d**mp.mpf('1.5'))
        a=2*mp.acosh(x);b=2*mp.asinh(l*mp.sqrt(d))
        return ((mp.sinh(a)-a)-(mp.sinh(b)-b))/(2*d**mp.mpf('1.5'))


def geometry(z):
    a,b=mp.matrix(z[:3]),mp.matrix(z[3:6])
    r1,r2=mp.norm(a),mp.norm(b);c=mp.norm(b-a);s=(r1+r2+c)/2
    return a,b,r1,r2,c,s


def cross(a,b): return mp.matrix((a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]))


def reference_velocity(z,mu,long_way,m,branch):
    a,b,r1,r2,c,s=geometry(z)
    l=mp.sqrt(1-c/s)*(-1 if long_way else 1);target=z[6]*mp.sqrt(2*mu/s**3)
    if m:
        # Independently locate the minimum using the anomaly formula derivative.
        # Bisection brackets are independent of the native root/minimum values.
        lo,hi=mp.mpf(0),mp.mpf('.999999999999999999999999')
        for _ in range(260):
            mid=(lo+hi)/2
            if mp.diff(lambda x:lagrange(x,l,m),mid)<0:lo=mid
            else:hi=mid
        minimum=(lo+hi)/2
        lo,hi=(mp.mpf(-1),minimum) if branch=='left' else (minimum,mp.mpf(1))
    else:
        lo,hi=mp.mpf(-1),mp.mpf(2)
        while lagrange(hi,l,0)>target:hi=2*hi+1
    decreasing=not m or branch=='left'
    for _ in range(280):
        mid=(lo+hi)/2
        if (lagrange(mid,l,m)>target)==decreasing:lo=mid
        else:hi=mid
    x=(lo+hi)/2;y=mp.sqrt(1-l*l*(1-x*x))
    rho=(r1-r2)/c;sigma=mp.sqrt(1-rho*rho);gamma=mp.sqrt(mu*s/2)
    h=cross(a,b);normal=h/mp.norm(h)*(-1 if long_way else 1)
    u1,u2=a/r1,b/r2;t1,t2=cross(normal,u1),cross(normal,u2)
    vr1=gamma/r1*((l*y-x)-rho*(l*y+x));vr2=-gamma/r2*((l*y-x)+rho*(l*y+x))
    transverse=gamma*sigma*(y+l*x)
    v1=u1*vr1+t1*(transverse/r1);v2=u2*vr2+t2*(transverse/r2)
    return mp.matrix(list(v1)+list(v2))


def multiprecision_jacobian(p,branch,step='1e-25',digits=100):
    with mp.workdps(digits):
        z=list(map(mp.mpf,(*p.r1_m,*p.r2_m,p.tof_s)))
        rscale=max(mp.norm(mp.matrix(z[:3])),mp.norm(mp.matrix(z[3:6])))
        J=np.empty((6,7));hbase=mp.mpf(step)
        for k in range(7):
            h=hbase*(z[6] if k==6 else rscale)
            za,zb=z.copy(),z.copy();za[k]+=h;zb[k]-=h
            va=reference_velocity(za,mp.mpf(p.mu_m3_s2),p.long_way,p.revolutions,branch)
            vb=reference_velocity(zb,mp.mpf(p.mu_m3_s2),p.long_way,p.revolutions,branch)
            J[:,k]=[float(v) for v in (va-vb)/(2*h)]
        return J


def flow_jacobian(p,r):
    """Independent two-body STM integration, then solve endpoint linear system."""
    radius=np.linalg.norm(p.r1_m);tau=math.sqrt(radius**3/p.mu_m3_s2);speed=radius/tau
    a=np.asarray(p.r1_m)/radius;v=r['v1']/speed;target=p.tof_s/tau
    def rhs(t,y):
        x=y[:3];rho=np.linalg.norm(x);acc=-x/rho**3
        grad=3*np.outer(x,x)/rho**5-np.eye(3)/rho**3
        A=np.zeros((6,6));A[:3,3:]=np.eye(3);A[3:,:3]=grad
        return np.r_[y[3:6],acc,(A@y[6:].reshape(6,6)).ravel()]
    initial=np.r_[a,v,np.eye(6).ravel()]
    sol=solve_ivp(rhs,(0,target),initial,method='DOP853',rtol=2.3e-14,atol=2e-14)
    if not sol.success: raise AssertionError(sol.message)
    end=sol.y[:6,-1];phi=sol.y[6:,-1].reshape(6,6)
    A,B,C,D=phi[:3,:3],phi[:3,3:],phi[3:,:3],phi[3:,3:]
    B_inv=np.linalg.solve(B,np.eye(3));first=np.column_stack((-B_inv@A,B_inv,-B_inv@end[3:]))
    acc=-end[:3]/np.linalg.norm(end[:3])**3
    second=np.column_stack((C,np.zeros((3,3)),acc))+D@first
    J=np.vstack((first,second))*speed/np.array([radius]*6+[tau])
    endpoint=max(np.linalg.norm(end[:3]-np.asarray(p.r2_m)/radius),np.linalg.norm(end[3:]-r['v2']/speed))
    return J,float(endpoint)


def cases():
    p=LambertInput((1.,.2,-.1),(-.3,1.1,.4),3.,1.)
    for long_way in (False,True):
        for tof in (.15,1.,3.,30.): yield f'zero-{long_way}-{tof}',replace(p,tof_s=tof,long_way=long_way),'left','flow'
        for m in (1,2,5):
            for branch in ('left','right'):yield f'multi-{long_way}-{m}-{branch}',replace(p,tof_s=3.+7*m,revolutions=m,long_way=long_way),branch,'flow'
    rng=np.random.default_rng(958718)
    for i in range(12):
        Q,_=np.linalg.qr(rng.normal(size=(3,3)))
        angle=rng.uniform(.4,2.7);radius=rng.uniform(.8,1.3);m=i%4
        r1=Q@np.array((1.,0.,0.));r2=Q@np.array((radius*math.cos(angle),radius*math.sin(angle),0.))
        random=LambertInput(tuple(r1),tuple(r2),float((m+2)*8),1.,bool(i%2),m)
        yield f'random-{i}',random,'left' if i%3 else 'right','flow'
    # Axis-aligned tiny arcs let the full Jacobian vary all 7 coordinates; no
    # finite-difference step is allowed to cross the degenerate-plane boundary.
    for angle in (1e-4,1e-8,1e-12,2e-14):
        yield f'tiny-{angle}',LambertInput((1.,0.,0.),(math.cos(angle),math.sin(angle),0.),angle,1.),'left','mp'
    # Arbitrary planes and nearly antiparallel endpoints exercise normal terms.
    u=np.array((.3,-.4,.5));u/=np.linalg.norm(u)
    w=np.cross(u,(.7,.1,-.2));w/=np.linalg.norm(w)
    for angle in (.005,.02,.03,1e-8,1e-12,math.pi-1e-6,math.pi-1e-8):
        yield f'plane-{angle}',LambertInput(tuple(u),tuple(math.cos(angle)*u+math.sin(angle)*w),angle,1.),'left','mp'
    for m,long_way,branch in ((0,True,'left'),(1,False,'left'),(1,False,'right'),
                               (1,True,'left'),(1,True,'right')):
        angle=1e-8
        elapsed=2*math.pi*(m+int(long_way))+(-angle if long_way else angle)
        yield f'near-full-circle-{m}-{long_way}-{branch}',LambertInput(tuple(u),tuple(math.cos(angle)*u+math.sin(angle)*w),elapsed,1.,long_way,m),branch,'mp'
    # Construct a parabolic time from an independent geometric formula; rounded
    # received time can lie on either side of the parabolic point.
    with mp.workdps(100):
        _,_,_,_,c,s=geometry(list(map(mp.mpf,(*p.r1_m,*p.r2_m,p.tof_s))))
        for long_way in (False,True):
            l=mp.sqrt(1-c/s)*(-1 if long_way else 1)
            tof=float(mp.mpf(2)/3*(1-l**3)*mp.sqrt(s**3/2))
            for offset in (-1e-9,0.,1e-9):
                yield f'parabolic-{long_way}-{offset}',replace(p,long_way=long_way,tof_s=tof*(1+offset)),'left','mp'
    for branch in ('left','right'):
        yield f'mp-multi-{branch}',replace(p,revolutions=2,tof_s=17.),branch,'mp'
    with mp.workdps(100):
        _,_,_,_,c,s=geometry(list(map(mp.mpf,(*p.r1_m,*p.r2_m,p.tof_s))))
        l=mp.sqrt(1-c/s)
        xmin=mp.findroot(lambda x:mp.diff(lambda y:lagrange(y,l,1),x),(.1,.2))
        minimum=lagrange(xmin,l,1)*mp.sqrt(s**3/2)
        for gap in ('1e-6','1e-8'):
            for branch in ('left','right'):
                yield f'near-minimum-{gap}-{branch}',replace(p,revolutions=1,tof_s=float(minimum*(1+mp.mpf(gap)))),branch,'mp-fold'


def validate(gpu=False):
    cpu=NativeEngine();corpus=list(cases());rows=pack_problems([p for _,p,_,_ in corpus],[b for _,_,b,_ in corpus])
    solutions,jac=cpu.cpu_with_jacobian(rows)
    device=None
    if gpu:
        engine=NativeEngine('cuda')
        with engine.cuda_batch(len(rows)) as batch:
            batch.upload(rows);batch.run(jacobian=True)
            gr,gj=batch.download(),batch.download_jacobian()
        np.testing.assert_array_equal(gr['status'],solutions['status']);np.testing.assert_array_equal(gj['status'],jac['status'])
        device=dict(name=engine.device_info()['name'])
    evidence=[]
    for i,((name,p,branch,kind),r,j) in enumerate(zip(corpus,solutions,jac)):
        if r['status'] or j['status']: raise AssertionError((name,r['status'],j))
        if kind=='flow':reference,endpoint=flow_jacobian(p,r)
        else:reference=multiprecision_jacobian(p,branch);endpoint=None
        # Entrywise absolute+relative scaling is reported separately from a
        # matrix norm, so large normal sensitivities cannot hide small entries.
        radius=np.linalg.norm(p.r1_m);speed=math.sqrt(p.mu_m3_s2/radius)
        units=np.array([speed/radius]*6+[speed/p.tof_s])
        error=float(np.max(abs(j['values']-reference)/(units+abs(reference))))
        bound=3e-10 if kind=='flow' else 2e-11
        if kind=='mp-fold':bound=max(bound,16*float(j['estimated_relative_slope_error']))
        record=dict(name=name,reference=kind,scaled_max_error=error,bound=bound,flow_endpoint_scaled_error=endpoint)
        record['estimated_relative_slope_error']=float(j['estimated_relative_slope_error'])
        if gpu:
            record['cuda_scaled_max_error']=float(np.max(abs(gj['values'][i]-reference)/(units+abs(reference))))
        evidence.append(record);print(json.dumps(record),flush=True)
        if error>bound or (gpu and record['cuda_scaled_max_error']>bound):raise AssertionError(record)
        if endpoint is not None and endpoint>2e-9:raise AssertionError(record)
    # Reference convergence at different step/precision, including a tiny arc
    # and both multi-revolution branches. This does not reuse native partials.
    convergence=[]
    for index in (33,len(corpus)-6,len(corpus)-5):
        name,p,branch,_=corpus[index]
        a=multiprecision_jacobian(p,branch)
        b=multiprecision_jacobian(p,branch,step='1e-28',digits=130)
        error=float(np.max(abs(a-b)/(1+abs(b))))
        if error>1e-12:raise AssertionError((name,'reference convergence',error))
        convergence.append(dict(name=name,scaled_change=error))
    return dict(cases=evidence,reference_convergence=convergence,cuda=device)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--gpu',action='store_true');parser.add_argument('--output',type=Path)
    args=parser.parse_args();started=time.perf_counter();result=validate(args.gpu);result['elapsed_s']=time.perf_counter()-started
    if args.output:args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2)+'\n')
    print('All independent Jacobian checks passed.',flush=True)
