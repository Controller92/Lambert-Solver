"""Menu and scripted CLI adapted from Simple-Lambert-Solver's Flyby-Solver UI.

Original interface copyright 2025 Controller92, MIT; see third_party/.
Only validated research workflows are offered.
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
    p = argparse.ArgumentParser(description='LambertLab research CLI: custom C++/CUDA Lambert and flyby engines')
    sub = p.add_subparsers(dest='command')
    grid_parser(sub)
    flyby = sub.add_parser('flyby',help='Solve saved unpowered or tangential powered encounters')
    flyby.add_argument('--config',type=Path,required=True)
    flyby.add_argument('--backend',choices=('cpu','cuda'),default='cpu')
    flyby.add_argument('--batch-size',type=int,default=65536)
    flyby.add_argument('--outdir',type=Path,default='results')
    mission = sub.add_parser('mission-search',help='Jointly search complete encounter chains')
    mission.add_argument('--config',type=Path,required=True)
    mission.add_argument('--model',choices=('ballistic','powered','both'))
    mission.add_argument('--backend',choices=('cpu','cuda'))
    mission.add_argument('--outdir',type=Path,default='results')
    mission.add_argument('--resume',type=Path,help='Warm continuation of one saved run')
    mission.add_argument('--dry-run',action='store_true')
    sweep=sub.add_parser('mission-sweep',help='Bounded EMC survey, flyby atlas and CPU/GPU stress test')
    sweep.add_argument('--config',type=Path,required=True)
    sweep.add_argument('--outdir',type=Path,default=Path('results'))
    sweep.add_argument('--seconds',type=float,default=900.)
    sweep.add_argument('--max-candidates',type=int,default=1048576)
    sweep.add_argument('--grid-step',type=float,default=2.)
    sweep.add_argument('--batch-size',type=int,default=1024)
    validate = sub.add_parser('mission-validate',help='Independently check an exported mission candidate')
    validate.add_argument('candidate',type=Path)
    validate.add_argument('--config',type=Path)
    validate.add_argument('--output',type=Path)
    horizons = sub.add_parser('mission-horizons',help='Compare candidate endpoint states against cached JPL Horizons')
    horizons.add_argument('candidate',type=Path)
    horizons.add_argument('--refresh',action='store_true',help='Explicitly fetch fresh public JPL snapshots')
    mission_plot = sub.add_parser('mission-plot',help='Plot an exported mission in heliocentric projections')
    mission_plot.add_argument('candidate',type=Path)
    mission_plot.add_argument('--output',type=Path)
    evaluate = sub.add_parser('mission-evaluate',help='Evaluate one complete set of event times')
    evaluate.add_argument('--config',type=Path,required=True)
    evaluate.add_argument('--days',type=float,nargs='+',required=True,
        help='Launch offset, leg durations, then surface stays, all in days')
    evaluate.add_argument('--output',type=Path,required=True)
    global_mission = sub.add_parser('nbody-mission',help='Jointly correct and independently validate a complete finite EMC mission')
    global_mission.add_argument('candidate',type=Path)
    global_mission.add_argument('--config',type=Path)
    global_mission.add_argument('--max-evaluations',type=int,default=160)
    global_mission.add_argument('--outdir',type=Path,default=Path('results'))
    nbody = sub.add_parser('nbody-flyby',help='Correct and independently validate one finite Mars encounter')
    nbody.add_argument('candidate',type=Path)
    nbody.add_argument('--config',type=Path,help='Source mission configuration; inferred by default')
    nbody.add_argument('--node',type=int,default=1)
    nbody.add_argument('--boundary-radius',type=float,default=1e8,help='Finite encounter boundary radius in metres')
    nbody.add_argument('--design-margin',type=float,default=10000.,help='Explicit seed periapsis margin added in metres')
    nbody.add_argument('--ephemeris-step',type=float,default=60.,help='SPICE table spacing in seconds')
    nbody.add_argument('--gmat-console',type=Path,help='Installed GmatConsole executable; never downloaded implicitly')
    nbody.add_argument('--outdir',type=Path,default=Path('results'))
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
    print('Custom Lambert and flyby engines | CPU or CUDA | saved research configurations')
    while True:
        print('\n1. Two Body Grid (Porkchop Plot)\n2. Run Saved Configuration\n3. Run EMC Paper Presets\n4. Regenerate Saved Plot\n5. View Saved Configurations\n6. Exit\n7. Flyby Encounters\n8. Complete Mission Search')
        try:
            choice = prompt('Select an option',6)
            if choice=='6':return 0
            if choice=='8':
                from .mission_config import load_mission
                config = load_mission(prompt('Mission configuration',str(ROOT/'examples/missions/emc-ten-year.json')))
                model = prompt('Mars encounters: ballistic/powered/both','both')
                execute_missions(config,prompt('Output directory','results'),model)
                continue
            if choice=='7':
                from .flyby import run_encounters
                run_encounters(prompt('Encounter configuration',str(ROOT/'examples/flyby-earth.json')),
                               prompt('Output directory','results'),prompt('Backend: cpu/cuda','cuda'))
                continue
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
        except (ValueError,KeyError,TypeError,OSError,RuntimeError,ImportError) as error:
            print(f'Error: {error}',file=sys.stderr)


def execute_missions(config,outdir,model=None,resume=None):
    from copy import deepcopy
    from .mission_search import run_mission_search
    mode = model or config.model
    if mode not in ('both','ballistic','powered'): raise ValueError('Invalid mission model')
    if mode == 'both' and resume: raise ValueError('Resume one explicitly selected model at a time')
    for selected in (('ballistic','powered') if mode == 'both' else (mode,)):
        current = deepcopy(config);current.model = selected
        run,summary = run_mission_search(current,outdir,resume,progress=lambda x:print(json.dumps(x),flush=True))
        print(json.dumps(dict(run=str(run.resolve()),**summary),indent=2))


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        if args.command in (None,'interactive'):return interactive()
        if args.command=='nbody-mission':
            from .nbody_mission import correct_mission
            run,summary=correct_mission(args.candidate,args.outdir,args.config,args.max_evaluations,
                progress=lambda s:print(json.dumps(s),flush=True))
            print(json.dumps(dict(run=str(run.resolve()),**summary),indent=2))
            return 0 if summary['status']=='passed_global_nbody_numerical_checks' else 2
        if args.command=='nbody-flyby':
            from .nbody_workflow import correct_encounter
            run,summary=correct_encounter(args.candidate,args.outdir,args.config,args.node,args.boundary_radius,
                args.design_margin,args.ephemeris_step,args.gmat_console,progress=lambda s:print(s,flush=True))
            print(json.dumps(dict(run=str(run.resolve()),**summary),indent=2))
            return 0 if summary['status']=='passed_local_nbody_validation' else 2
        if args.command=='mission-plot':
            from .mission_plot import plot_candidate
            print(plot_candidate(args.candidate,args.output));return 0
        if args.command=='mission-horizons':
            from .mission_horizons import crosscheck_candidate
            report = crosscheck_candidate(args.candidate,args.refresh)
            print(report['status']);return 0 if report['status']=='consistent_ephemerides' else 2
        if args.command=='mission-validate':
            from .mission_validation import validate_candidate
            report = validate_candidate(args.candidate,args.config,args.output)
            print(report['status'])
            for check in report['checks']:
                if not check['passed']: print(json.dumps(check))
            return 0 if report['status']=='passed_patched_conic' else 2
        if args.command=='mission-sweep':
            from .mission_config import load_mission
            from .mission_sweep import run_sweep
            path,summary=run_sweep(load_mission(args.config),args.outdir,args.seconds,args.max_candidates,args.grid_step,args.batch_size)
            print(json.dumps(dict(run=str(path.resolve()),**summary),indent=2));return 0
        if args.command in ('mission-search','mission-evaluate'):
            from .mission_config import load_mission
            config = load_mission(args.config)
            if args.command=='mission-evaluate':
                from .mission_dynamics import MissionEvaluator
                from .mission_search import write_json,fingerprint
                with MissionEvaluator(config,args.output.parent/'state-cache') as evaluator:
                    candidate = evaluator.candidate(args.days,False)
                    candidate['fingerprint'] = fingerprint(config,evaluator)
                    write_json(args.output,candidate)
                print(f'Saved {args.output}; feasible={candidate["feasible"]}; independent validation required')
                return 0
            if args.backend:config.backend = args.backend
            config.validate()
            if args.dry_run:
                mode = args.model or config.model
                print(json.dumps(dict(models=['ballistic','powered'] if mode=='both' else [mode],
                                      configuration=config.to_dict()),indent=2));return 0
            execute_missions(config,args.outdir,args.model,args.resume);return 0
        if args.command=='flyby':
            from .flyby import run_encounters
            run_encounters(args.config,args.outdir,args.backend,args.batch_size);return 0
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
    except (ValueError,KeyError,TypeError,OSError,RuntimeError,ImportError) as error:
        print(f'Error: {error}',file=sys.stderr)
        return 1
    except (KeyboardInterrupt,EOFError):
        print('\nCancelled',file=sys.stderr);return 130


if __name__=='__main__':sys.exit(main())
