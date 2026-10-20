"""Reproducible complete-chain search, constrained refinement and checkpoints."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import platform
import scipy
import time
import uuid
import numpy as np
from scipy.optimize import differential_evolution, minimize
from .mission_dynamics import MissionEvaluator

ROOT = Path(__file__).resolve().parents[2]


def safe(value):
    if isinstance(value, dict): return {str(k):safe(v) for k,v in value.items()}
    if isinstance(value, (list,tuple)): return [safe(v) for v in value]
    if isinstance(value, np.ndarray): return safe(value.tolist())
    if isinstance(value, np.generic): return safe(value.item())
    if isinstance(value, float) and not np.isfinite(value): return None
    return value


def write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(safe(value),indent=2,allow_nan=False)+'\n')
    temporary.replace(path)


def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def configuration_fingerprint(config):
    return hashlib.sha256(json.dumps(config.to_dict(),sort_keys=True).encode()).hexdigest()


def fingerprint(config, evaluator):
    libraries = ['liblambert_cpu.so','libflyby_cpu.so']
    if config.backend == 'cuda': libraries += ['liblambert_cuda.so','libflyby_cuda.so']
    sources = [Path('src/research')/name for name in ('mission_config.py','mission_local.py','mission_dynamics.py','mission_search.py','ephemeris.py')]
    sources += [Path('src')/name for name in ('lambert_native.py','flyby_native.py','flyby.py')]
    return dict(config_sha256=configuration_fingerprint(config),kernels=evaluator.e.provenance,
        libraries={name:digest(ROOT/'build/native'/name) for name in libraries},
        modules={str(name):digest(ROOT/name) for name in sources},
        versions=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,
                      spiceypy=evaluator.e.sp.__version__,spice=evaluator.e.sp.tkvrsn('TOOLKIT')))


def refine(config, point, selected_policies, cache_dir, ephemeris=None):
    """Freeze a discrete branch sequence; enforce physical constraints directly.

    Search penalties never substitute for equality/inequality feasibility here.
    CPU refinement is explicit, separate from the configured population backend.
    """
    cfg = deepcopy(config); cfg.backend = 'cpu'
    cache = {}
    with MissionEvaluator(cfg,cache_dir,selected_policies,ephemeris=ephemeris) as e:
        def evaluate(x):
            key = np.asarray(x,dtype='f8').tobytes()
            if key not in cache:
                cache.clear(); cache[key] = e.block(np.asarray(x)[None,:])
            return cache[key]
        def objective(x):
            b = evaluate(x)
            cost = b['start_cost'][0,0]+b['end_cost'][0,0]
            cost += sum(node['cost'][0,0,0] for node in b['nodes'])
            return float(cost/1000) if np.isfinite(cost) else 1e12
        def equalities(x):
            b = evaluate(x)
            values = [node['speed_difference'][0,0,0]/1000 for j,node in enumerate(b['nodes'],1)
                      if cfg.visits[j]['kind']=='flyby' and cfg.model=='ballistic']
            return np.nan_to_num(values,nan=1e6,posinf=1e6,neginf=-1e6)
        def inequalities(x):
            b = evaluate(x)
            values = [(cfg.max_duration_days-b['duration_days'][0])/365.25,
                      b['c3_margin'][0,0]/100]
            values.extend(b['solar_margin'][0,:,0])
            for j,node in enumerate(b['nodes'],1):
                if cfg.visits[j]['kind']=='flyby':
                    values += [node['turn_margin'][0,0,0],node['upper_radius_turn_margin'][0,0,0],node['burn_margin'][0,0,0]/1000]
                else: values += [node['site_margin'][0,0,0],node['direction_sine'][0,0,0]-1e-10]
            if (b['legs']['status']!=0).any(): values.append(-1e6)
            else: values.append(1.)
            return np.nan_to_num(values,nan=-1e6,posinf=1e6,neginf=-1e6)
        constraints = [dict(type='ineq',fun=inequalities)]
        if any(v['kind']=='flyby' for v in cfg.visits) and cfg.model=='ballistic':
            constraints.append(dict(type='eq',fun=equalities))
        result = minimize(objective,np.asarray(point,float),method='SLSQP',bounds=e.bounds,
            constraints=constraints,options=dict(maxiter=cfg.refine_iterations,ftol=1e-11,eps=1e-5))
        candidate = e.candidate(result.x)
        record = dict(success=bool(result.success),message=str(result.message),iterations=int(result.nit),
            decisions_days=result.x,physical_candidate_found=candidate is not None,
            equality_residuals_km_s=equalities(result.x),inequality_margins=inequalities(result.x))
        if candidate is not None:
            candidate['refinement'] = record
        return candidate,record


def run_mission_search(config, outdir='results', resume=None, progress=None):
    """Search one explicitly named flyby model. Resume is a warm continuation.

    Checkpoints preserve population, RNG state, seed progress and all input/code
    hashes. SciPy re-evaluates a resumed initial population; bitwise continuity
    with an uninterrupted run is not promised. Concurrent writes are locked.
    """
    config.validate(); started = time.perf_counter()
    if resume is None:
        run = Path(outdir)/('mission-'+config.model+'-'+uuid.uuid4().hex[:12])
        run.mkdir(parents=True,exist_ok=False)
        write_json(run/'config.json',config.to_dict())
    else:
        run = Path(resume).resolve()
        saved = json.loads((run/'config.json').read_text())
        if saved != config.to_dict(): raise ValueError('Resume configuration differs')
    import fcntl
    with (run/'.lock').open('a+') as lock:
        try: fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError: raise RuntimeError('This mission run is already active')
        with MissionEvaluator(config,run/'state-cache') as e:
            fp = fingerprint(config,e)
            if resume is not None and json.loads((run/'fingerprint.json').read_text()) != fp:
                raise ValueError('Resume code, libraries, or kernels differ; start a fresh run')
            write_json(run/'fingerprint.json',fp)
            if resume is None:
                for relative in fp['modules']:
                    destination = run/'sources'/relative
                    destination.parent.mkdir(parents=True,exist_ok=True)
                    destination.write_bytes((ROOT/relative).read_bytes())
            state = json.loads((run/'checkpoint.json').read_text()) if (run/'checkpoint.json').exists() else dict(seed_index=0,generation=0,archive=[],completed_seeds=[],refinement_done=False)
            if resume is not None and state['refinement_done'] and (run/'summary.json').exists():
                return run,json.loads((run/'summary.json').read_text())
            archive = state['archive']
            def remember(population):
                for lo in range(0,len(population),config.batch_candidates):
                    block = e.block(population[lo:lo+config.batch_candidates])
                    for i,point in enumerate(block['decisions']):
                        path = block['path'][i]
                        score = float(block['score'][i])
                        if not np.isfinite(score): continue
                        record = dict(decisions_days=point.tolist(),score=score,
                            policies=[e.policy_list[int(k)] for k in path])
                        archive.append(record)
                unique = {}
                for record in sorted(archive,key=lambda r:r['score']):
                    key = (tuple(np.round(record['decisions_days'],5)),json.dumps(record['policies'],sort_keys=True))
                    if key not in unique: unique[key] = record
                ordered = list(unique.values())
                # Keep good local starts AND other time/branch basins. A single
                # converged population must not erase every alternative seed.
                selected = ordered[:max(1,config.archive_size//2)]
                scale = np.maximum(1.,e.bounds[:,1]-e.bounds[:,0])
                for record in ordered[len(selected):]:
                    distinct = all(record['policies'] != prior['policies'] or
                        np.max(np.abs(np.asarray(record['decisions_days'])-prior['decisions_days'])/scale) >= .03
                        for prior in selected)
                    if distinct: selected.append(record)
                    if len(selected) >= config.archive_size: break
                for record in ordered:
                    if len(selected) >= config.archive_size: break
                    if not any(record is prior for prior in selected): selected.append(record)
                archive[:] = selected
            for seed_index in range(state['seed_index'],len(config.seeds)):
                seed = config.seeds[seed_index]; rng = np.random.default_rng(seed)
                continuing = seed_index == state['seed_index'] and state['generation'] > 0 and (run/'population.npz').exists()
                init = 'latinhypercube'
                generation = state['generation'] if continuing else 0
                if continuing:
                    if digest(run/'population.npz') != state.get('population_sha256'):
                        raise ValueError('Checkpoint population checksum differs')
                    with np.load(run/'population.npz',allow_pickle=False) as f: init = f['population']
                    rng.bit_generator.state = state['rng_state']
                def callback(intermediate_result):
                    nonlocal generation
                    generation += 1
                    if generation % 10 == 0 or generation == 1:
                        remember(intermediate_result.population)
                        temporary = run/'population.tmp.npz'
                        np.savez_compressed(temporary,population=intermediate_result.population)
                        temporary.replace(run/'population.npz')
                        state.update(seed_index=seed_index,generation=generation,archive=archive,rng_state=rng.bit_generator.state,
                                     population_sha256=digest(run/'population.npz'))
                        write_json(run/'checkpoint.json',state)
                        update = dict(seed=seed,generation=generation,best_score_km_s=float(intermediate_result.fun))
                        with (run/'history.jsonl').open('a') as f: f.write(json.dumps(update)+'\n')
                        if progress: progress(update)
                    return False
                result = differential_evolution(lambda x:e.fitness(np.asarray(x).T),e.bounds,
                    maxiter=max(0,config.generations-generation),popsize=config.popsize,rng=rng,
                    callback=callback,init=init,polish=False,vectorized=True,updating='deferred',
                    workers=1,tol=1e-7,atol=1e-7)
                remember(result.population)
                state['completed_seeds'].append(dict(seed=seed,success=bool(result.success),message=str(result.message),
                    generations=generation,best_score_km_s=float(result.fun)))
                state.update(seed_index=seed_index+1,generation=0,archive=archive)
                write_json(run/'checkpoint.json',state)
            candidates = []; refinements = []
            for record in archive:
                candidate = e.candidate(record['decisions_days'])
                if candidate is not None: candidates.append(candidate)
            # Search archive holds complete chains, including policies. Refine
            # several distinct starts rather than merely the minimum-C3 leg.
            for index,record in enumerate(archive[:config.refine_starts]):
                candidate,metadata = refine(config,record['decisions_days'],record['policies'],run/'state-cache',ephemeris=e.e)
                refinements.append(metadata)
                if candidate is not None: candidates.append(candidate)
                if progress: progress(dict(refinement=index+1,of=min(config.refine_starts,len(archive)),feasible_candidates=len(candidates)))
            unique = {}
            for candidate in sorted(candidates,key=lambda r:r['total_delta_v_m_s']):
                key = tuple(np.round(candidate['decisions_days'],4))
                if key not in unique: unique[key] = candidate
            candidates = list(unique.values())[:config.archive_size]
            (run/'candidates').mkdir(exist_ok=True)
            for index,candidate in enumerate(candidates):
                candidate['candidate_id'] = f'candidate-{index:03d}'
                candidate['fingerprint'] = fp
                write_json(run/'candidates'/f'{candidate["candidate_id"]}.json',candidate)
            if candidates: write_json(run/'best.json',candidates[0])
            elif archive:
                diagnostic = e.candidate(archive[0]['decisions_days'],False)
                write_json(run/'best-infeasible.json',diagnostic)
            write_json(run/'refinements.json',refinements)
            state['refinement_done'] = True; write_json(run/'checkpoint.json',state)
            summary = dict(status='feasible_candidates_found' if candidates else 'no_feasible_candidate_found',
                model=config.model,feasible_candidate_count=len(candidates),
                best_delta_v_m_s=candidates[0]['total_delta_v_m_s'] if candidates else None,
                best_duration_days=candidates[0]['duration_days'] if candidates else None,
                validation_status='independent_validation_required',completed_seeds=state['completed_seeds'],
                wall_seconds=time.perf_counter()-started,refinement_backend='cpu',population_backend=config.backend,
                model_scope='impulsive patched conics, spherical rotating surface visit, free-plane endpoint parking',
                search_limitation='finite stochastic search; no global optimality or impossibility certificate',
                checkpoint_semantics='warm continuation, not bitwise uninterrupted replay')
            write_json(run/'summary.json',summary)
            return run,summary
