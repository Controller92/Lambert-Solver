"""Saved finite-encounter projections from independent direct-SPICE dynamics."""
import numpy as np
from scipy.integrate import solve_ivp


def plot_local(ephem,seed,result,path):
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 P=np.asarray(seed['incoming_relative_periapsis_state'])[:3];P=P/np.linalg.norm(P)
 Q=np.asarray(seed['incoming_relative_periapsis_state'])[3:];Q=Q/np.linalg.norm(Q)
 ti,tm,to=[result[k] for k in ['entry_time','encounter_time','exit_time']]
 state=np.asarray(result['entry_state_shifted_si'])
 def ode(t,y):return np.r_[y[3:],ephem.exact_force(t,y)]
 segments=[]
 for a,b in [(ti,tm),(tm,to)]:
  times=np.linspace(a,b,800)
  sol=solve_ivp(ode,(a,b),state,t_eval=times,method='DOP853',rtol=3e-14,atol=1e-7,max_step=30.)
  if not sol.success:raise RuntimeError(sol.message)
  relative=sol.y[:3].T-np.array([ephem.body_state('499',t)[:3] for t in times])
  segments.append((times,relative));state=sol.y[:,-1].copy()
  state[3:]+=np.asarray(result['node_after_shifted_si'])[3:]-np.asarray(result['node_before_shifted_si'])[3:]
 fig,axes=plt.subplots(1,2,figsize=(12,6.8))
 fig.subplots_adjust(left=.085,right=.98,bottom=.14,top=.80,wspace=.28)
 for ax in axes:
  ax.add_patch(plt.Circle((0,0),3396.2,color='#bd725a',alpha=.65,label='Mars sphere'))
  for (times,r),label,color in zip(segments,['Incoming arc','Outgoing arc'],['#25669b','#c37a18']):
   ax.plot(r@P/1000,r@Q/1000,color=color,lw=1.6,label=label)
  ax.set_aspect('equal');ax.set_xlabel('Seed periapsis direction (km)');ax.set_ylabel('Seed tangent direction (km)');ax.grid(alpha=.2)
 axes[0].set_xlim(-105000,105000);axes[0].set_ylim(-105000,105000)
 axes[0].set_xticks([-100000,-50000,0,50000,100000])
 axes[0].tick_params(axis='x',labelsize=8)
 axes[0].legend(fontsize=8,loc='upper left')
 axes[1].set_xlim(-8000,10000);axes[1].set_ylim(-9000,9000);axes[1].set_title('Closest approach detail')
 axes[1].set_xticks([-6000,-2000,2000,6000,10000])
 fig.suptitle(f'{seed["model"].capitalize()} finite Mars encounter | Newtonian point masses\nLocal correction only; full mission remains to be reattached',fontsize=11,y=.97)
 fig.savefig(path,dpi=170);plt.close(fig)
