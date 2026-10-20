"""Scientific projections of saved heliocentric patched-conic trajectories."""
import json
from pathlib import Path
import numpy as np
from scipy.integrate import solve_ivp
from .mission_search import digest

AU=149597870700.
DAY=86400.


def plot_candidate(path, output=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    path=Path(path);c=json.loads(path.read_text())
    validation=path.with_suffix('.validation.json')
    status='independent validation required'
    if validation.exists():
        report=json.loads(validation.read_text())
        if report['candidate_sha256']==digest(path):status=report['status']
    fig,axes=plt.subplots(1,2,figsize=(12,5.6),layout='constrained')
    colors=plt.get_cmap('tab10').colors
    names={'399':'Earth','499':'Mars','20000001':'Ceres'}
    mu=c['solar_mu_m3_s2']*DAY**2/AU**3
    for i,leg in enumerate(c['legs']):
        duration=(leg['arrival_et']-leg['departure_et'])/DAY
        def ode(t,y):return np.r_[y[3:],-mu*y[:3]/np.linalg.norm(y[:3])**3]
        sol=solve_ivp(ode,(0,duration),np.r_[np.array(leg['departure_state_si'][:3])/AU,np.array(leg['v1_m_s'])*DAY/AU],
            t_eval=np.linspace(0,duration,1500),method='DOP853',rtol=3e-13,atol=2e-15)
        if not sol.success:raise RuntimeError(sol.message)
        label=f'{i+1}: {names.get(leg["origin"],leg["origin"])}–{names.get(leg["destination"],leg["destination"])} (M={leg["policy"]["revolutions"]})'
        for ax,vertical in zip(axes,(1,2)):
            ax.plot(sol.y[0],sol.y[vertical],color=colors[i%10],lw=1.3,label=label)
            a=np.array(leg['departure_state_si'][:3])/AU;b=np.array(leg['arrival_state_si'][:3])/AU
            ax.scatter([a[0],b[0]],[a[vertical],b[vertical]],s=22,color=colors[i%10],zorder=4)
    for ax,vertical in zip(axes,('Y','Z')):
        ax.scatter([0],[0],color='#df9b16',s=65,marker='*',zorder=5)
        ax.axhline(0,color='.85',lw=.6);ax.axvline(0,color='.85',lw=.6)
        ax.set(xlabel='Heliocentric X (AU)',ylabel=f'Heliocentric {vertical} (AU)')
        ax.set_aspect('equal',adjustable='datalim');ax.grid(alpha=.2)
    axes[0].legend(fontsize=8,loc='best')
    fig.suptitle(f'{c["search_model"].capitalize()} Mars encounters | {c["duration_days"]/365.25:.2f} years | total Δv {c["total_delta_v_m_s"]/1000:.2f} km/s\n{status}; {c["frame"]}; impulsive patched conics',fontsize=11)
    destination=Path(output) if output else path.with_suffix('.png')
    fig.savefig(destination,dpi=180);plt.close(fig)
    return destination
