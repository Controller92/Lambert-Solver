"""Regenerate all EMC survey figures from saved numerical outputs."""
from datetime import datetime
import json
from pathlib import Path
import numpy as np
from .mission_sweep import sample_arc
from .mission_dynamics import AU


def plot_sweep(run):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm,Normalize
    import matplotlib.dates as dates
    run=Path(run);summary=json.loads((run/'summary.json').read_text())
    config=json.loads((run/'config.json').read_text())
    data=np.load(run/'atlas.npz',allow_pickle=False)
    figures=[];epoch=dates.date2num(datetime(2000,1,1,12))
    names=('Earth → Mars','Mars → Ceres','Ceres → Mars','Mars → Earth')
    fig,axes=plt.subplots(2,2,figsize=(14,10),layout='constrained')
    for leg,ax in enumerate(axes.flat,1):
        with np.load(run/f'leg-{leg}-grid.npz') as grid:
            x=epoch+grid['dep_et']/86400;y=grid['tof_days'];c3=grid['C3']
            finite=c3[np.isfinite(c3)&(c3>0)]
            lo=max(.01,float(np.percentile(finite,1)));hi=max(lo*1.01,float(np.percentile(finite,90)))
            im=ax.pcolormesh(x,y,c3.T,cmap='turbo',norm=LogNorm(lo,hi),shading='auto',rasterized=True)
            fig.colorbar(im,ax=ax,label='Departure C3 (km²/s²), log scale',extend='both')
            ax.set(title=names[leg-1],ylabel='Flight time (days)',xlabel='Departure date (TDB)')
            ax.xaxis.set_major_locator(dates.AutoDateLocator(minticks=3,maxticks=6));ax.xaxis.set_major_formatter(dates.DateFormatter('%Y-%m'))
    fig.suptitle(f'Independent leg porkchops | minimum departure C3 over {config["orientation"]} directions, M=0..{config["max_revolutions"]}\nThese minima are not preselected for the joined mission search',fontsize=13)
    path=run/'four-leg-porkchops.png';fig.savefig(path,dpi=160);plt.close(fig);figures.append(path)
    for model in ('ballistic','powered'):
        fig,axes=plt.subplots(2,2,figsize=(13,10),layout='constrained')
        for row,node in enumerate((1,3)):
            key=f'{model}-{node}';title='Earth → Mars → Ceres' if node==1 else 'Ceres → Mars → Earth'
            for col,vertical in enumerate((1,2)):
                ax=axes[row,col];hk=f'track_{key}-{vertical}'
                if hk in data:
                    hist=data[hk];im=ax.pcolormesh(data['track_edges'],data['track_edges'],np.ma.masked_equal(hist.T,0),
                        norm=LogNorm(1,max(2,hist.max())),cmap='magma',shading='flat',rasterized=True)
                    fig.colorbar(im,ax=ax,label='Conic-locus sample count (log)')
                else:ax.text(.5,.5,'No accepted local connections',transform=ax.transAxes,ha='center')
                ax.scatter([0],[0],marker='*',s=70,color='#ffa600')
                ax.set(xlabel='Heliocentric X (AU)',ylabel=f'Heliocentric {"Y" if vertical==1 else "Z"} (AU)',title=f'{title} | {summary["counts"].get(key,0):,} connections')
                ax.set_aspect('equal');ax.set_xlim(-8,8);ax.set_ylim((-8,8) if vertical==1 else (-4,4));ax.grid(alpha=.15)
        fig.suptitle(f'{model.capitalize()} Mars encounters | all accepted sampled connection tracks\nUniform true-anomaly sampling, not residence-time density; local feasibility does not certify a full mission',fontsize=12)
        path=run/f'{model}-all-trajectory-coverage.png';fig.savefig(path,dpi=160);plt.close(fig);figures.append(path)
        fig,axes=plt.subplots(2,2,figsize=(13,10),layout='constrained')
        for row,node in enumerate((1,3)):
            key=f'{model}-{node}';ax=axes[row,0];hk=f'metric_{key}'
            if hk in data:
                hist=data[hk];im=ax.pcolormesh(data['metric_plane_edges'],data['metric_gain_edges'],np.ma.masked_equal(hist.T,0),
                    cmap='viridis',norm=LogNorm(1,max(2,hist.max())),rasterized=True)
                fig.colorbar(im,ax=ax,label='All accepted connection counts (log)')
            ax.axhline(0,color='.5',lw=.7);ax.set(xlabel='Heliocentric orbital-plane rotation (deg)',ylabel='Heliocentric speed change (km/s)',title=('Outbound' if node==1 else 'Return')+' Mars encounter')
            ax=axes[row,1];source=run/f'highlights-{key}.npz'
            if source.exists():
                with np.load(source) as sample:
                    im=ax.scatter(sample['ceres_plane_after_deg'],sample['heliocentric_speed_gain_km_s'],c=sample['burn_km_s'],s=7,alpha=.6,cmap='plasma')
                    fig.colorbar(im,ax=ax,label='Periapsis burn (km/s)')
            ax.axhline(0,color='.5',lw=.7);ax.set(xlabel='Post-flyby misalignment with Ceres orbital plane (deg)',ylabel='Heliocentric speed change (km/s)',title='Bounded unbiased highlight sample; all rows saved')
        fig.suptitle(f'{model.capitalize()} | orbital-plane change and heliocentric speed gain\nPlane rotation differs from asymptote turn; powered gains include the explicitly charged periapsis impulse',fontsize=12)
        path=run/f'{model}-plane-change-speed-gain.png';fig.savefig(path,dpi=160);plt.close(fig);figures.append(path)
        fig=plt.figure(figsize=(13,6.5),layout='constrained')
        for col,node in enumerate((1,3),1):
            ax=fig.add_subplot(1,2,col,projection='3d');source=run/f'highlights-{model}-{node}.npz'
            if source.exists():
                with np.load(source) as s:
                    selected=np.linspace(0,len(s['rank'])-1,min(50,len(s['rank'])),dtype=int)
                    for index in selected:
                        color=plt.get_cmap('viridis')(s['plane_change_deg'][index]/180)
                        for leg in (node-1,node):
                            points=sample_arc(s[f'r{leg}'][index:index+1],s[f'v{leg}'][index:index+1],s[f'end{leg}'][index:index+1],
                                s[f'm{leg}'][index:index+1],summary['solar_mu_m3_s2'],128)[0]/AU
                            ax.plot(*points.T,color=color,alpha=.5,lw=.6)
            ax.scatter([0],[0],[0],color='orange',marker='*',s=70)
            ax.set(xlabel='X (AU)',ylabel='Y (AU)',zlabel='Z (AU)',title='Outbound' if node==1 else 'Return')
        fig.suptitle(f'{model.capitalize()} | up to 50 connection highlights per direction\nFull sampled trajectory coverage is in the separate density figures',fontsize=12)
        path=run/f'{model}-3d-highlights.png';fig.savefig(path,dpi=160);plt.close(fig);figures.append(path)
    timing=json.loads((run/'timings.json').read_text());bench=timing['benchmarks']['records']
    fig,axes=plt.subplots(1,2,figsize=(12,4.8),layout='constrained')
    cpu=[r for r in bench if r['backend']=='cpu']
    axes[0].plot([r['threads'] for r in cpu],[r['rows_per_second']/1e6 for r in cpu],'o-')
    axes[0].set(xlabel='CPU worker threads',ylabel='Million branch attempts/s',title='Same native workload; wall timing');axes[0].grid(alpha=.25)
    gpu=[r for r in bench if r['backend']=='cuda_resident']
    labels=['CPU 1 thread','CPU best threads','CUDA resident'];values=[cpu[0]['rows_per_second'],max(r['rows_per_second'] for r in cpu),gpu[0]['rows_per_second']]
    axes[1].bar(labels,np.array(values)/1e6,color=['#536878','#7a9e7e','#e6a23c']);axes[1].set(ylabel='Million branch attempts/s',title='CUDA excludes allocation / transfers')
    fig.suptitle('Measured native Lambert throughput | unsuccessful branch attempts included')
    path=run/'cpu-gpu-throughput.png';fig.savefig(path,dpi=160);plt.close(fig);figures.append(path)
    return figures
