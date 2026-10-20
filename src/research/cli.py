"""Menu and scripted CLI adapted from Simple-Lambert-Solver's Flyby-Solver UI.

Original interface copyright 2025 Controller92, MIT; see third_party/.
Only validated research workflows are offered. Flyby migration is a later stage.
"""
import argparse
import json
from pathlib import Path
import sys
import time
import uuid

from .config import GridConfig, from_dict, load_config, save_config

ROOT = Path(__file__).resolve().parents[2]


def grid_parser(sub):
    p = sub.add_parser('transfer-grid',help='Compute and save a two-body porkchop grid')
    p.add_argument('--config',type=Path,help='New or legacy two_body_grid JSON')
    for name in ('dep-body','arr-body','dep-start','dep-end','backend','frame','time-scale','orientation'):
        p.add_argument('--'+name)
    for name in ('dep-step','tof-min','tof-max','tof-step','cmin','cmax','tof-contour-step'):
        p.add_argument('--'+name,type=float)
    for name in ('max-revolutions','batch-size','max-cells','max-memory-mb'):
        p.add_argument('--'+name,type=int)
    p.add_argument('--kernels','--kernel',dest='kernels',action='append',help='Explicit kernel path; repeat for every kernel')
    p.add_argument('--outdir',default='results',type=Path)
    p.add_argument('--cache-dir',default='.research-cache',type=Path)
    p.add_argument('--save-config',type=Path)
    p.add_argument('--no-plot',action='store_true')
    p.add_argument('--dry-run',action='store_true',help='Validate and show grid size without running SPICE/CUDA')


def parser():
    p = argparse.ArgumentParser(description='LambertLab research CLI: custom C++/CUDA Lambert engine')
    sub = p.add_subparsers(dest='command')
    grid_parser(sub)
    plot = sub.add_parser('plot',help='Regenerate a figure from grid.npz without recomputing')
    plot.add_argument('grid',type=Path);plot.add_argument('--output',type=Path)
    configs = sub.add_parser('configs',help='List saved grid configurations')
    configs.add_argument('directory',type=Path,nargs='?',default=ROOT/'configs')
    batch = sub.add_parser('batch',help='Run a list of saved grid configurations')
    batch.add_argument('--config',dest='configs',type=Path,action='append',required=True)
    batch.add_argument('--backend',choices=('cpu','cuda'))
    batch.add_argument('--outdir',type=Path,default='results')
    batch.add_argument('--cache-dir',type=Path,default='.research-cache')
    batch.add_argument('--dry-run',action='store_true')
    sub.add_parser('interactive',help='Open the interactive menu')
    return p


def describe(config):
    from .grid import policies
    nd,nt = config.shape
    print(f'{config.dep_body} → {config.arr_body} | {config.backend} | {config.orientation} | M=0..{config.max_revolutions}')
    print(f'{nd:,} departures × {nt:,} flight times = {nd*nt:,} cells; {nd*nt*len(policies(config)):,} candidate solves')
    print('Dates: TDB. States: heliocentric geometric. C3: km²/s².')


def execute(config,outdir='results',cache_dir='.research-cache'):
    from .grid import run_grid
    describe(config)
    previous = -1
    def progress(done,total):
        nonlocal previous
        percent = int(100*done/total)
        if percent//10 != previous//10:
            print(f'Grid: {percent}% ({done:,}/{total:,} cells)',flush=True)
            previous = percent
    path,metadata = run_grid(config,outdir,cache_dir,progress)
    print(f'Saved results: {path.resolve()}')
    print(f'Valid cells: {metadata["valid_cells"]:,}/{metadata["total_cells"]:,}; '
          f'precision warnings: {metadata["warning_cells"]:,}; unknown: {metadata["unknown_precision_cells"]:,}')
    print(f'Total {metadata["total_seconds"]:.3f} s; states {metadata["state_seconds"]:.3f} s; '
          f'solve stage {metadata["solve_seconds"]:.3f} s')
    if metadata['minimum']: print('Minimum:',json.dumps(metadata['minimum']))
    return path


def run_batch(configs,outdir,cache_dir='.research-cache',backend=None,dry_run=False):
    configs = [(Path(path),load_config(path)) for path in configs]
    for _,config in configs:
        if backend:config.backend = backend
        config.validate()
    if dry_run:
        for name,config in configs:
            print(name);describe(config)
        return None
    started = time.perf_counter()
    runs = []
    for name,config in configs:
        path = execute(config,Path(outdir)/name.stem,cache_dir)
        runs.append(dict(config=str(name.resolve()),run=str(path.resolve())))
    summary = dict(runs=runs,total_seconds=time.perf_counter()-started)
    destination = Path(outdir)/('batch-'+uuid.uuid4().hex[:8]+'.json')
    destination.write_text(json.dumps(summary,indent=2)+'\n')
    print(f'Batch complete: {summary["total_seconds"]:.3f} s; summary {destination.resolve()}')
    return destination


def prompt(label,default):
    value = input(f'{label} [{default}]: ').strip()
    return value or str(default)


def saved_configs(directory=ROOT/'configs'):
    return [p for p in sorted(Path(directory).glob('*.json')) if p.name!='kernel-sources.json']


def choose_config():
    files = saved_configs()
    if not files:
        print('No saved configurations');return None
    for i,path in enumerate(files,1):print(f'{i}. {path.name}')
    index = int(prompt('Configuration number',1))-1
    if not 0<=index<len(files): raise ValueError('Invalid configuration selection')
    return load_config(files[index])


def interactive():
    print('Welcome to LambertLab Interactive Simulation Runner!')
    print('Custom Lambert engine | CPU or CUDA | saved research configurations')
    while True:
        print('\n1. Two Body Grid (Porkchop Plot)\n2. Run Saved Configuration\n3. Run EMC Paper Presets\n4. Regenerate Saved Plot\n5. View Saved Configurations\n6. Exit')
        try:
            choice = prompt('Select an option',6)
            if choice=='6':return 0
            if choice=='5':
                for path in saved_configs():print(path.name)
                continue
            if choice=='4':
                from .plot import plot_grid
                print(plot_grid(Path(prompt('Path to grid.npz','results/grid.npz'))))
                continue
            if choice=='3':
                run_batch([ROOT/'configs'/f'emc-{name}.json' for name in ('earth-mars','mars-ceres','ceres-mars','mars-earth')],
                          prompt('Output directory','results'))
                continue
            if choice=='2':
                config = choose_config()
                if config is None:continue
            elif choice=='1':
                defaults = GridConfig()
                data = defaults.to_dict()
                print('Calendar dates below are interpreted as TDB.')
                for name,label in (('dep_body','Departure body (NAIF ID/name)'),('arr_body','Arrival body (NAIF ID/name)'),
                                   ('dep_start','Departure start date'),('dep_end','Departure end date')):
                    data[name] = prompt(label,data[name])
                for name,label in (('dep_step','Departure step days'),('tof_min','Minimum TOF days'),
                                   ('tof_max','Maximum TOF days'),('tof_step','TOF step days')):
                    data[name] = float(prompt(label,data[name]))
                data['orientation'] = prompt('Orientation: prograde/retrograde/both/short/long',data['orientation'])
                data['max_revolutions'] = int(prompt('Maximum complete revolutions',0))
                data['backend'] = prompt('Backend: cpu/cuda','cuda')
                names = ['naif0012.tls','de440s.bsp','gm_de440.tpc']
                if (ROOT/'data/kernels/20000001.bsp').exists():names.append('20000001.bsp')
                defaults = ';'.join(str(ROOT/'data/kernels'/n) for n in names)
                data['kernels'] = [p.strip() for p in prompt('Kernel paths separated by semicolons',defaults).split(';') if p.strip()]
                config = from_dict(data)
                destination = Path(prompt('Save configuration',str(ROOT/'configs'/'custom-grid.json')))
                if destination.exists():
                    print('Choose a new filename; an existing configuration will not be overwritten.')
                    continue
                save_config(config,destination)
                print(f'Saved configuration: {destination}')
            else:
                print('Invalid choice');continue
            execute(config,prompt('Output directory','results'))
        except (ValueError,OSError,RuntimeError,ImportError) as error:
            print(f'Error: {error}',file=sys.stderr)


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        if args.command in (None,'interactive'):return interactive()
        if args.command=='batch':
            run_batch(args.configs,args.outdir,args.cache_dir,args.backend,args.dry_run);return 0
        if args.command=='configs':
            for path in saved_configs(args.directory):print(path)
            return 0
        if args.command=='plot':
            from .plot import plot_grid
            print(plot_grid(args.grid,args.output));return 0
        config = load_config(args.config) if args.config else GridConfig()
        for name in GridConfig.__dataclass_fields__:
            if hasattr(args,name) and getattr(args,name) is not None:
                setattr(config,name,getattr(args,name))
        if args.kernels:
            config.kernels = [str(Path(p).expanduser().resolve()) for p in args.kernels]
        if args.no_plot:config.plot = False
        config.validate()
        if args.save_config:
            if args.save_config.exists():raise ValueError('Refusing to overwrite existing configuration')
            save_config(config,args.save_config)
        if args.dry_run:describe(config);return 0
        execute(config,args.outdir,args.cache_dir)
        return 0
    except (ValueError,OSError,RuntimeError,ImportError) as error:
        print(f'Error: {error}',file=sys.stderr)
        return 1
    except (KeyboardInterrupt,EOFError):
        print('\nCancelled',file=sys.stderr);return 130


if __name__=='__main__':sys.exit(main())
