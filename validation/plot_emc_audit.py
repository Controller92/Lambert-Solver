"""Scientific figures from preserved EMC audit results; no recomputation of ephemerides."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from audit_emc import MARS_RADIUS,iso


def plot(folder,outdir):
    outdir.mkdir(parents=True,exist_ok=True)
    r=json.loads((folder/'nominal.json').read_text())
    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':160})
    first,second=r['scenarios'];selection=[]
    for s,index in [(first,0),(second,1),(first,1)]:selection.append(next(v for v in s['mars_encounters'] if v['prograde_pair'] and v['encounter_index']==index))
    labels=['Outbound\n250-day Ceres leg','Return\nCalendar dates','Return\nStated durations']
    x=np.arange(3);width=.35
    fig,axes=plt.subplots(1,2,figsize=(12,4.7),layout='constrained')
    for k,key,color,label in [(0,'vinf_in_m_s','#22658c','Incoming'),(1,'vinf_out_m_s','#c95e32','Required outgoing')]:
        values=[v[key]/1000 for v in selection]
        bars=axes[0].bar(x+(k-.5)*width,values,width,label=label,color=color)
        axes[0].bar_label(bars,fmt='%.2f',padding=4,fontsize=10)
    axes[0].set_ylabel('Mars-relative excess speed (km/s)');axes[0].set_ylim(0,25)
    axes[0].set_title('Unpowered encounter requires equal speed');axes[0].legend(frameon=False)
    for k,key,color,label in [(0,'maximum_unpowered_turn_deg','#22658c','Available maximum'),(1,'required_turn_deg','#c95e32','Required turn')]:
        bars=axes[1].bar(x+(k-.5)*width,[v[key] for v in selection],width,label=label,color=color)
        axes[1].bar_label(bars,fmt='%.1f°',padding=4,fontsize=10)
    axes[1].set_ylabel('Planet-relative turn (degrees)');axes[1].set_ylim(0,185)
    axes[1].set_title('Turn limit at assumed 200 km altitude');axes[1].legend(frameon=False,loc='upper left')
    for ax in axes:ax.set_xticks(x,labels);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    fig.suptitle('EMC nominal Mars encounters: both unpowered constraints fail',fontsize=15)
    fig.savefig(outdir/'nominal-encounters.png');fig.savefig(outdir/'nominal-encounters.svg');plt.close(fig)
    with np.load(folder/'search-populations.npz') as f:
        incoming=f['incoming'];outgoing=f['outgoing'];mars=f['mars_states'];launch=f['launch_et'];enc=f['mars_et']
        # Multi-revolution candidates are all infeasible in this population;
        # choose the sole prograde M=0 orientation independently per grid cell.
        for pop in (incoming,outgoing):
            assert all(np.all(pop[...,k]['status']!=0) for k,p in enumerate(f['policies']) if p[1]>0)
        ci=np.cross(mars[:,None,:3],incoming[:,:,0]['v2'])[:,:,2]>0
        co=np.cross(mars[:,None,:3],outgoing[:,:,0]['v1'])[:,:,2]>0
        vi=np.where(ci[:,:,None],incoming[:,:,0]['v2'],incoming[:,:,5]['v2'])-mars[:,None,3:]
        vo=np.where(co[:,:,None],outgoing[:,:,0]['v1'],outgoing[:,:,5]['v1'])-mars[:,None,3:]
        field=np.empty(ci.shape)
        for im in range(len(mars)):
            a=vi[im];b=vo[im];A=np.linalg.norm(a,axis=1);B=np.linalg.norm(b,axis=1)
            delta=np.arctan2(np.linalg.norm(np.cross(a[:,None,:],b[None,:,:]),axis=2),a@b.T)
            maximum=2*np.arcsin(1/(1+(MARS_RADIUS+200000)*A*A/r['mars_mu_m3_s2']))
            gap=np.maximum(0,delta-maximum[:,None])
            residual=np.hypot(B[None,:]-A[:,None],2*np.sqrt(A[:,None]*B[None,:])*np.sin(gap/2))
            field[im]=np.min(residual,axis=1)/1000
        expected=json.loads((folder/'neighborhood.json').read_text())['best_sampled_asymptotic_correction']['prograde']['optimistic_postflyby_asymptotic_correction_m_s']/1000
        assert abs(field.min()-expected)<1e-9
        xx=[datetime.fromisoformat(iso(t)) for t in launch];yy=[datetime.fromisoformat(iso(t)) for t in enc]
    fig,ax=plt.subplots(figsize=(10,6.5),layout='constrained')
    mesh=ax.pcolormesh(xx,yy,field,cmap='viridis',vmin=0,vmax=15,shading='auto',rasterized=True)
    color=fig.colorbar(mesh,ax=ax,extend='max');color.set_label('Minimum residual asymptotic velocity (km/s)')
    ax.scatter([datetime(2035,7,1)],[datetime(2036,1,17)],marker='*',s=170,c='white',edgecolors='black',label='Paper nominal dates',zorder=5)
    best=json.loads((folder/'continuous-search.json').read_text())['minimum']
    ax.scatter([datetime.fromisoformat(best['launch_tdb'])],[datetime.fromisoformat(best['mars_tdb'])],marker='x',s=85,c='#ff6666',linewidths=2,label='Best continuous search point',zorder=6)
    ax.set_xlabel('Earth departure date (2035, TDB)');ax.set_ylabel('Mars encounter date (2035–2036, TDB)')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'));ax.yaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    ax.legend(loc='upper left',framealpha=.95)
    ax.set_title('Nearby prograde search: minimum over 160–340 day Ceres legs\nAn unpowered velocity match would have zero residual',pad=14)
    fig.savefig(outdir/'nearby-search.png');fig.savefig(outdir/'nearby-search.svg');plt.close(fig)
    np.savez_compressed(outdir/'nearby-search-figure-data.npz',launch_et=launch,mars_et=enc,minimum_residual_km_s=field)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('folder',type=Path);p.add_argument('--outdir',type=Path,required=True)
    a=p.parse_args();plot(a.folder,a.outdir)
