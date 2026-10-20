"""Finite full-chain projections from independent direct-SPICE propagation."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plot_global(ephem,seed,result,solutions,path):
 AU=149597870700.;fig,axes=plt.subplots(1,2,figsize=(12,6));colors=['#0072B2','#009E73','#D55E00','#CC79A7']
 for record in solutions:
  times=np.linspace(record['start'],record['end'],1001)
  y=record['sol'].sol(times)[:3].T+record['origin']
  sun=np.array([ephem.body_state('10',t)[:3] for t in times]);r=(y-sun)/AU
  for ax,j in zip(axes,[1,2]):ax.plot(r[:,0],r[:,j],color=colors[record['leg']],lw=1.4)
 for i,(body,name) in enumerate([('399','Earth'),('499','Mars'),('20000001','Ceres')]):
  times=np.linspace(result['nodes']['times'][0],result['nodes']['times'][-1],1600)
  r=np.array([(ephem.body_state(body,t)[:3]-ephem.body_state('10',t)[:3])/AU for t in times])
  for ax,j in zip(axes,[1,2]):ax.plot(r[:,0],r[:,j],color='0.65',lw=.6,alpha=.7)
 for ax,j in zip(axes,[1,2]):
  ax.scatter([0],[0],c='#E69F00',s=35,label='Sun');ax.set_xlabel('J2000 X [AU]');ax.set_ylabel('J2000 '+('Y' if j==1 else 'Z')+' [AU]');ax.grid(alpha=.2);ax.set_aspect('equal',adjustable='datalim')
 for i,color in enumerate(colors):axes[0].plot([],[],c=color,label=['Earth → Mars','Mars → Ceres','Ceres → Mars','Mars → Earth'][i])
 axes[0].legend(loc='upper right',fontsize=8)
 cost=result['nodes']['total_delta_v_m_s']/1000;years=(result['nodes']['times'][-1]-result['nodes']['times'][0])/(86400*365.25)
 fig.suptitle(f"{seed['model'].capitalize()} finite N-body + Mars J2 correction: {years:.3f} years, {cost:.3f} km/s",fontsize=12)
 fig.text(.5,.025,'Spherical impulsive parking and surface operations; numerical correction, no flight-readiness or optimality claim.',ha='center',fontsize=8)
 fig.subplots_adjust(left=.08,right=.98,bottom=.14,top=.89,wspace=.28);fig.savefig(path,dpi=170);plt.close(fig)
