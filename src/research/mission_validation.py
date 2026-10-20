"""Independent acceptance checks for complete saved patched-conic missions.

No native Lambert/flyby calls: Gooding (lamberthub), DOP853 integration,
orbital-element asymptotes and fresh SPICE queries check the exported states.
This does not certify N-body dynamics, finite burns, terrain, or launch sites.
"""
import json
from importlib.metadata import version
from pathlib import Path
import numpy as np
from scipy.integrate import solve_ivp
from lamberthub import gooding1990
from .ephemeris import SpiceEphemeris
from .mission_config import load_mission
from .mission_search import digest, write_json, configuration_fingerprint

AU = 149597870700.
DAY = 86400.


def norm(x): return float(np.linalg.norm(x))
def vec(x): return np.asarray(x,float)


def asymptote(r, v, mu, incoming):
    """Recover asymptotic velocity from an arbitrary hyperbolic local state."""
    r,v = vec(r),vec(v)
    h = np.cross(r,v); energy = np.dot(v,v)-2*mu/norm(r)
    eccentricity = np.cross(v,h)/mu-r/norm(r); e = norm(eccentricity)
    if e <= 1 or energy <= 0 or norm(h) == 0:
        raise ValueError('Local state is not a nonradial hyperbola')
    P = eccentricity/e; Q = np.cross(h/norm(h),P)
    return np.sqrt(energy)*((1 if incoming else -1)*P/e+np.sqrt(e*e-1)/e*Q)


def validate_candidate(path, config_path=None, output=None):
    path = Path(path).resolve()
    run = path.parent.parent if path.parent.name == 'candidates' else path.parent
    config_path = Path(config_path) if config_path else run/'config.json'
    config = load_mission(config_path)
    candidate = json.loads(path.read_text())
    checks = []; leg_records = []; encounter_records = []; budget = []
    def check(name, value, limit, unit=''):
        value,limit = float(value),float(limit)
        checks.append(dict(name=name,value=value,limit=limit,unit=unit,
                           passed=bool(np.isfinite(value) and value <= limit)))
    def difference(name,a,b,tolerance,unit=''):
        check(name,norm(vec(a)-vec(b)),tolerance,unit)
    def condition(name,passed): check(name,0 if passed else 1,0)
    condition('candidate explicitly feasible',candidate.get('feasible') is True)
    condition('schema',candidate.get('schema_version') == 1)
    condition('configuration fingerprint',candidate.get('fingerprint',{}).get('config_sha256') == configuration_fingerprint(config))
    condition('model matches configuration',candidate.get('search_model') == config.model)
    condition('frame and time scale',candidate.get('frame') == config.frame and candidate.get('time_scale') == 'TDB')
    legs = candidate['legs']; visits = candidate['visits']
    if len(legs) != len(config.tof_days) or len(visits) != len(config.visits):
        raise ValueError('Candidate chain length differs from configuration')
    with SpiceEphemeris(config.kernels,config.frame,run/'validation-state-cache') as ephem:
        fp = candidate.get('fingerprint',{})
        condition('recorded kernels match',fp.get('kernels') == ephem.provenance)
        difference('solar GM',candidate['solar_mu_m3_s2'],ephem.mu,ephem.mu*1e-14)
        states = []
        for i,leg in enumerate(legs):
            name = f'leg {i}'
            condition(name+' body sequence',leg['origin'] == config.visits[i]['body'] and leg['destination'] == config.visits[i+1]['body'])
            condition(name+' native statuses',leg['status'] == 0 and leg['root_status'] == 0)
            condition(name+' precision',leg['precision_limited'] == 0)
            t1,t2 = float(leg['departure_et']),float(leg['arrival_et']); duration = t2-t1
            check(name+' TOF below minimum',config.tof_days[i][0]-duration/DAY,1e-8,'days')
            check(name+' TOF above maximum',duration/DAY-config.tof_days[i][1],1e-8,'days')
            a = ephem.states(leg['origin'],[t1])[0]; b = ephem.states(leg['destination'],[t2])[0]
            difference(name+' departure body state',a,leg['departure_state_si'],1e-4,'SI')
            difference(name+' arrival body state',b,leg['arrival_state_si'],1e-4,'SI')
            difference(name+' departure date label',ephem.epoch(leg['departure_tdb']),t1,2e-6,'s')
            difference(name+' arrival date label',ephem.epoch(leg['arrival_tdb']),t2,2e-6,'s')
            v1,v2 = vec(leg['v1_m_s']),vec(leg['v2_m_s'])
            refs = []; m = leg['policy']['revolutions']
            condition(name+' configured revolutions',isinstance(m,int) and 0 <= m <= config.max_revolutions)
            orientation = leg['policy']['orientation']
            allowed = ('short','long') if config.orientation == 'both' else (config.orientation,)
            condition(name+' configured orientation',orientation in allowed)
            expected_long = (np.cross(a[:3],b[:3])[2]<0 if orientation=='prograde' else
                             np.cross(a[:3],b[:3])[2]>0 if orientation=='retrograde' else orientation=='long')
            condition(name+' orientation identity',leg['long_way'] == bool(expected_long))
            condition(name+' branch index',leg['policy']['branch'] in ((0,) if m==0 else (0,1)))
            prograde = bool((np.cross(a[:3],b[:3])[2] > 0) != leg['long_way'])
            for low in ((False,) if m == 0 else (False,True)):
                try:
                    p,q = gooding1990(ephem.mu,a[:3],b[:3],duration,M=m,prograde=prograde,
                        low_path=low,maxiter=100,atol=1e-13,rtol=1e-13)
                    refs.append(max(norm(p-v1),norm(q-v2)))
                except ValueError: pass
            gooding_error = min(refs,default=float('inf'))
            check(name+' independent Gooding velocities',gooding_error,1e-4,'m/s')
            h = np.cross(a[:3],v1); hhat = h/norm(h); xhat = a[:3]/norm(a[:3]); yhat = np.cross(hhat,xhat)
            gm = ephem.mu*DAY**2/AU**3
            def ode(t,y): return np.r_[y[3:],-gm*y[:3]/np.linalg.norm(y[:3])**3]
            sol = solve_ivp(ode,(0,duration/DAY),np.r_[a[:3]/AU,v1*DAY/AU],method='DOP853',
                rtol=3e-13,atol=2e-15,max_step=max(.01,duration/DAY/600))
            condition(name+' DOP853 termination',sol.success)
            position_error = norm(sol.y[:3,-1]*AU-b[:3]); velocity_error = norm(sol.y[3:,-1]*AU/DAY-v2)
            check(name+' independent propagation position',position_error,500.,'m')
            check(name+' independent propagation velocity',velocity_error,.001,'m/s')
            angles = np.unwrap(np.arctan2(yhat@sol.y[:3],xhat@sol.y[:3]))
            winding = int(np.floor((angles[-1]+1e-10)/(2*np.pi)))
            condition(name+' revolution count from propagation',winding == m)
            eccentricity = norm(np.cross(v1,h)/ephem.mu-a[:3]/norm(a[:3]))
            perihelion = norm(h)**2/(ephem.mu*(1+eccentricity))
            check(name+' solar clearance',config.minimum_solar_radius_m-perihelion,1.,'m')
            difference(name+' recorded perihelion',perihelion,leg['solar_perihelion_m'],1.,'m')
            states.append((a,b,v1-a[3:],v2-b[3:]))
            leg_records.append(dict(index=i,gooding_velocity_error_m_s=gooding_error,
                ode_position_error_m=position_error,ode_velocity_error_m_s=velocity_error,
                propagated_revolutions=winding,solar_perihelion_m=perihelion))
        launch = legs[0]['departure_et']; finish = legs[-1]['arrival_et']
        check('launch before window',ephem.epoch(config.launch_window[0])-launch,2e-6,'s')
        check('launch after window',launch-ephem.epoch(config.launch_window[1]),2e-6,'s')
        duration = (finish-launch)/DAY
        check('maximum mission duration',duration-config.max_duration_days,1e-8,'days')
        difference('recorded duration',duration,candidate['duration_days'],1e-8,'days')
        c3 = norm(states[0][2])**2/1e6
        difference('recorded launch C3',c3,candidate['launch_C3_km2_s2'],1e-8,'km2/s2')
        difference('recorded arrival speed',norm(states[-1][3]),candidate['arrival_vinf_m_s'],1e-6,'m/s')
        if config.launch_c3_limit_km2_s2 is not None:
            check('launch C3 limit',c3-config.launch_c3_limit_km2_s2,1e-8,'km2/s2')
        decisions = vec(candidate['decisions_days'])
        expected = [(launch-ephem.epoch(config.launch_window[0]))/DAY]
        expected += [(leg['arrival_et']-leg['departure_et'])/DAY for leg in legs]
        for j,v in enumerate(visits):
            spec = config.visits[j]; name = f'visit {j}'
            condition(name+' identity',v['index'] == j and v['body'] == spec['body'] and v['kind'] == spec['kind'])
            mu = float(ephem.sp.bodvrd(spec['gm_body'],'GM',1)[1][0])*1e9
            difference(name+' GM',mu,v['mu_m3_s2'],mu*1e-14)
            if spec['kind'] in ('departure','arrival'):
                incoming = spec['kind'] == 'arrival'
                r = vec(v['rperi_m']); hyper = vec(v['hyperbolic_velocity_m_s']); circular = vec(v['circular_velocity_m_s'])
                target = states[-1][3] if incoming else states[0][2]
                difference(name+' parking radius',norm(r),spec['radius_m']+spec['parking_altitude_m'],1e-5,'m')
                difference(name+' circular speed squared',np.dot(circular,circular),mu/norm(r),1e-4,'m2/s2')
                check(name+' circular tangency',abs(np.dot(r,circular))/(norm(r)*norm(circular)),1e-10)
                difference(name+' local asymptote',asymptote(r,hyper,mu,incoming),target,1e-4,'m/s')
                burn = circular-hyper if incoming else hyper-circular
                difference(name+' burn vector',burn,v['burn_vector_m_s'],1e-6,'m/s')
                difference(name+' burn magnitude',norm(burn),v['delta_v_m_s'],1e-6,'m/s')
                budget.append(norm(burn)); continue
            tin,tout = legs[j-1]['arrival_et'],legs[j]['departure_et']
            difference(name+' arrival epoch',v['arrival_et'],tin,1e-6,'s')
            difference(name+' departure epoch',v['departure_et'],tout,1e-6,'s')
            vi,vo = states[j-1][3],states[j][2]
            difference(name+' required incoming asymptote',v['vinf_in_m_s'],vi,1e-6,'m/s')
            difference(name+' required outgoing asymptote',v['vinf_out_m_s'],vo,1e-6,'m/s')
            if spec['kind'] == 'flyby':
                condition(name+' flyby model',v['model'] == config.model)
                difference(name+' shared epoch',tin,tout,1e-6,'s')
                difference(name+' shared body state',states[j-1][1],states[j][0],1e-4,'SI')
                condition(name+' native status',v['status'] == 0)
                r = vec(v['rperi']); pi = vec(v['vperi_in']); po = vec(v['vperi_out'])
                rp = norm(r)
                check(name+' minimum radius',spec['radius_m']+spec['minimum_altitude_m']-rp,1e-4,'m')
                check(name+' maximum radius',rp-spec['maximum_periapsis_m'],1e-4,'m')
                difference(name+' recorded radius',rp,v['rp'],1e-4,'m')
                difference(name+' incoming conic',asymptote(r,pi,mu,True),vi,1e-4,'m/s')
                difference(name+' outgoing conic',asymptote(r,po,mu,False),vo,1e-4,'m/s')
                check(name+' incoming periapsis tangency',abs(np.dot(r,pi))/(rp*norm(pi)),1e-10)
                check(name+' outgoing periapsis tangency',abs(np.dot(r,po))/(rp*norm(po)),1e-10)
                check(name+' tangential powered geometry',norm(np.cross(pi,po))/(norm(pi)*norm(po)),1e-10)
                burn = norm(po-pi)
                difference(name+' impulse magnitude',burn,v['delta_v'],1e-5,'m/s')
                if config.model == 'ballistic':
                    check(name+' unpowered velocity continuity',burn,1e-5,'m/s')
                    check(name+' unpowered speed matching',abs(norm(vi)-norm(vo)),config.speed_atol_m_s+config.speed_rtol*norm(vi),'m/s')
                if config.max_flyby_burn_m_s is not None: check(name+' burn cap',burn-config.max_flyby_burn_m_s,1e-5,'m/s')
                budget.append(burn)
                encounter_records.append(dict(index=j,kind='flyby',periapsis_altitude_m=rp-spec['radius_m'],delta_v_m_s=burn,
                    vinf_speed_difference_m_s=norm(vo)-norm(vi)))
            else:
                condition(name+' body frame',v['body_frame'] == spec['body_frame'])
                difference(name+' recorded surface radius',v['radius_m'],spec['radius_m'],1e-6,'m')
                stay = (tout-tin)/DAY; expected.append(stay)
                check(name+' minimum stay',spec['stay_days'][0]-stay,1e-8,'days')
                check(name+' maximum stay',stay-spec['stay_days'][1],1e-8,'days')
                fi = ephem.sp.sxform(spec['body_frame'],config.frame,tin)
                fo = ephem.sp.sxform(spec['body_frame'],config.frame,tout)
                difference(name+' incoming rotation',fi,v['transform_in'],1e-12)
                difference(name+' outgoing rotation',fo,v['transform_out'],1e-12)
                site = vec(v['site_body_fixed_m']); ri,ro = vec(v['rperi_in_m']),vec(v['rperi_out_m'])
                pi,po = vec(v['vperi_in_m_s']),vec(v['vperi_out_m_s'])
                gi,go = fi[3:,:3]@site,fo[3:,:3]@site
                difference(name+' spherical surface radius',norm(site),spec['radius_m'],1e-4,'m')
                difference(name+' same site at arrival',fi[:3,:3]@site,ri,1e-4,'m')
                difference(name+' same site at departure',fo[:3,:3]@site,ro,1e-4,'m')
                difference(name+' ground arrival rotation velocity',gi,v['ground_in_m_s'],1e-7,'m/s')
                difference(name+' ground departure rotation velocity',go,v['ground_out_m_s'],1e-7,'m/s')
                difference(name+' incoming conic',asymptote(ri,pi,mu,True),vi,1e-4,'m/s')
                difference(name+' outgoing conic',asymptote(ro,po,mu,False),vo,1e-4,'m/s')
                check(name+' landing tangency',abs(np.dot(ri,pi))/(norm(ri)*norm(pi)),1e-10)
                check(name+' departure tangency',abs(np.dot(ro,po))/(norm(ro)*norm(po)),1e-10)
                difference(name+' landing burn',gi-pi,v['landing_velocity_change_m_s'],1e-6,'m/s')
                difference(name+' departure burn',po-go,v['departure_velocity_change_m_s'],1e-6,'m/s')
                budget.extend([norm(gi-pi),norm(po-go)])
                encounter_records.append(dict(index=j,kind='surface',stay_days=stay,landing_delta_v_m_s=norm(gi-pi),launch_delta_v_m_s=norm(po-go)))
        difference('decision vector reconstructs timeline',expected,decisions,1e-8,'days')
        difference('full maneuver budget',sum(budget),candidate['total_delta_v_m_s'],1e-4,'m/s')
        recorded = [item['delta_v_m_s'] for item in candidate['cost_items']]
        difference('each maneuver budget item',budget,recorded,1e-4,'m/s')
    report = dict(schema_version=1,status='passed_patched_conic' if all(c['passed'] for c in checks) else 'failed',
        candidate_sha256=digest(path),config_sha256=digest(config_path),checks=checks,legs=leg_records,encounters=encounter_records,
        validator_sha256=digest(__file__),
        versions={name:version(name) for name in ('numpy','scipy','lamberthub','spiceypy')},
        maximum_ode_position_error_m=max(r['ode_position_error_m'] for r in leg_records),
        maximum_gooding_velocity_error_m_s=max(r['gooding_velocity_error_m_s'] for r in leg_records),
        scope='Independent two-body leg propagation and local conic/rotation/budget checks. Not N-body/finite-burn/terrain/flight validation.')
    destination = Path(output) if output else path.with_suffix('.validation.json')
    write_json(destination,report)
    return report
