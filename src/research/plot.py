"""Regenerate a labelled porkchop from saved numerical data, without SPICE/CUDA."""
from datetime import datetime
import json
from pathlib import Path
import numpy as np
from .ephemeris import sha256


NAMES = {'399':'Earth','4':'Mars barycenter','499':'Mars','20000001':'Ceres'}


def read_run(path):
    path = Path(path)
    metadata = json.loads(path.with_name('run.json').read_text())
    if metadata.get('schema') != 1 or metadata['grid_sha256'] != sha256(path):
        raise ValueError('Grid provenance/hash mismatch')
    return metadata


def plot_grid(path, destination=None, metadata=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.dates as dates
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    path = Path(path)
    metadata = metadata or read_run(path)
    metadata['matplotlib'] = matplotlib.__version__
    config = metadata['config']
    with np.load(path,allow_pickle=False) as saved:
        dep = dates.date2num(datetime(2000,1,1,12))+saved['dep_et']/86400.
        tof = saved['tof_days']
        c3 = saved['C3']
        if c3.shape != (len(dep),len(tof)) or not np.isfinite(dep).all() or not np.isfinite(tof).all():
            raise ValueError('Invalid saved grid axes')
    destination = Path(destination) if destination else path.with_name('porkchop.png')
    if destination.resolve() in (path.resolve(),path.with_name('run.json').resolve(),path.with_name('config.json').resolve()):
        raise ValueError('Plot output must not overwrite saved research data')
    fig,ax = plt.subplots(figsize=(10,6),layout='constrained')
    try:
        norm = Normalize(config['cmin'],config['cmax'])
        display = np.ma.masked_where(~np.isfinite(c3)|(c3<config['cmin'])|(c3>config['cmax']),c3)
        if min(c3.shape)>=2:
            x = np.broadcast_to(dep[:,None],c3.shape)
            y = dep[:,None]+tof[None,:]
            if display.count():
                ax.contourf(x,y,display,levels=np.linspace(config['cmin'],config['cmax'],60),cmap='turbo',norm=norm)
            else:
                ax.text(.5,.5,'No solutions within the displayed C3 range',transform=ax.transAxes,ha='center')
            levels = np.arange(config['tof_contour_step'],tof[-1]+config['tof_contour_step']/2,config['tof_contour_step'])
            levels = levels[(levels>tof[0])&(levels<tof[-1])]
            if len(levels):
                curves = ax.contour(x,y,np.broadcast_to(tof[None,:],c3.shape),levels=levels,colors='0.25',linewidths=.55)
                ax.clabel(curves,fmt=lambda v:f'{v:g} d',fontsize=7)
        else:
            ax.scatter(np.repeat(dep,len(tof)),(dep[:,None]+tof[None,:]).ravel(),
                       c=c3.ravel(),norm=norm,cmap='turbo')
        finite = np.isfinite(c3)
        valid_rows = finite.any(axis=1)
        argmin = np.argmin(np.where(finite,c3,np.inf),axis=1)
        valley = np.where(valid_rows,dep+tof[argmin],np.nan)
        ax.scatter(dep[valid_rows],valley[valid_rows],c='k',s=3,zorder=3,
                   label='Minimum departure C3 at sampled dates')
        fig.colorbar(ScalarMappable(norm=norm,cmap='turbo'),ax=ax,label='Departure C3 (km²/s²)')
        ax.set_xlabel('Departure date (TDB)');ax.set_ylabel('Arrival date (TDB)')
        ax.xaxis.set_major_formatter(dates.DateFormatter('%Y-%m-%d\n%H:%M' if np.ptp(dep)<7 else '%Y-%m-%d'))
        ax.yaxis.set_major_formatter(dates.DateFormatter('%Y-%m-%d\n%H:%M' if np.ptp(dep)+np.ptp(tof)<7 else '%Y-%m-%d'))
        for axis in (ax.xaxis,ax.yaxis):
            locator = dates.AutoDateLocator(minticks=3,maxticks=7)
            locator.intervald[dates.MONTHLY] = [1,2,3,4,6,12]
            axis.set_major_locator(locator)
        plt.setp(ax.get_xticklabels(),rotation=25,ha='right')
        dep_name = NAMES.get(str(config['dep_body']),str(config['dep_body']))
        arr_name = NAMES.get(str(config['arr_body']),str(config['arr_body']))
        revs = 'M=0' if config['max_revolutions']==0 else f'M=0..{config["max_revolutions"]}'
        ax.set_title(f'{dep_name} → {arr_name} | {config["orientation"]}, {revs}')
        ax.legend(loc='best',fontsize=8)
        fig.savefig(destination,dpi=160)
    finally:
        plt.close(fig)
    return destination
