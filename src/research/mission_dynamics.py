"""Complete chain evaluation, preserving Lambert branches through encounters.

For fixed event times, dynamic programming finds the least-cost branch sequence
using pair costs at EVERY interior visit. No minimum-C3 leg is selected first.
SPICE stays on the host; only the Lambert/flyby batches use the chosen native
backend. Penalized search diagnostics are never labelled feasible trajectories.
"""
from collections import OrderedDict
from datetime import datetime, timedelta
import math
from pathlib import Path
import numpy as np
from .ephemeris import SpiceEphemeris
from .mission_local import length, surface_pair, parking
from ..lambert_native import NativeEngine, INPUT_DTYPE
from ..flyby_native import FlybyEngine, INPUT_DTYPE as FLYBY_INPUT
from ..flyby import Mode

DAY = 86400.
AU = 149597870700.


def calendar(et):
    return (datetime(2000, 1, 1, 12)+timedelta(seconds=float(et))).isoformat()


def policies(config):
    ways = ('short', 'long') if config.orientation == 'both' else (config.orientation,)
    return [dict(orientation=w, revolutions=m, branch=b)
            for w in ways for m in range(config.max_revolutions+1)
            for b in ((0,) if m == 0 else (0, 1))]


def dynamic_path(start, links, end):
    """Batched exact minimum of an additive chain with pair-dependent costs."""
    dp = start.copy(); parents = []
    for link in links:
        proposed = dp[:, :, None]+link
        parent = np.argmin(proposed, axis=1)
        dp = np.take_along_axis(proposed, parent[:, None, :], axis=1)[:, 0, :]
        parents.append(parent)
    final = dp+end
    last = np.argmin(final, axis=1)
    score = final[np.arange(len(final)), last]
    path = [last]
    for parent in reversed(parents):
        last = parent[np.arange(len(final)), last]
        path.append(last)
    return score, np.stack(path[::-1], axis=1)


class MissionEvaluator:
    def __init__(self, config, cache_dir, fixed_policies=None, ephemeris=None):
        self.config = config.validate()
        self.owns_ephemeris = ephemeris is None
        if ephemeris is not None:
            if config.frame != ephemeris.frame or [str(Path(k).resolve()) for k in config.kernels] != [p['path'] for p in ephemeris.provenance]:
                raise ValueError('Shared ephemeris must have identical frame and ordered kernels')
        self.e = ephemeris if ephemeris is not None else SpiceEphemeris(config.kernels, config.frame, cache_dir)
        self.lb_batch = self.fb_batch = None
        self.states_cache = OrderedDict(); self.frames_cache = OrderedDict()
        self.fixed = fixed_policies
        self.policy_list = policies(config)
        self.K = 1 if fixed_policies is not None else len(self.policy_list)
        self.L = len(config.visits)-1
        if fixed_policies is not None and len(fixed_policies) != self.L:
            self.close(); raise ValueError('One fixed policy is required per leg')
        try:
            if any(0 <= self.e.sp.bods2c(v['body']) <= 9 for v in config.visits):
                raise ValueError('Use body-centre states, not named or numeric planetary barycenters')
            self.mu = [float(self.e.sp.bodvrd(v['gm_body'], 'GM', 1)[1][0])*1e9 for v in config.visits]
            if not all(math.isfinite(m) and m > 0 for m in self.mu):
                raise ValueError('Invalid body GM')
            self.launch_origin = self.e.epoch(config.launch_window[0])
            self.bounds = [[0., (self.e.epoch(config.launch_window[1])-self.launch_origin)/DAY]]+config.tof_days
            self.surface_indices = [i for i,v in enumerate(config.visits) if v['kind'] == 'surface']
            self.stay_columns = {i:1+self.L+j for j,i in enumerate(self.surface_indices)}
            self.bounds += [config.visits[i]['stay_days'] for i in self.surface_indices]
            self.bounds = np.asarray(self.bounds, float)
            self.lb = NativeEngine(config.backend)
            self.fb = FlybyEngine(config.backend)
            if config.backend == 'cuda':
                self.lb_batch = self.lb.cuda_batch(config.batch_candidates*self.L*self.K)
                self.fb_batch = self.fb.cuda_batch(config.batch_candidates*self.K*self.K)
            # Check extreme-date coverage and surface-frame availability before
            # a long search. No silent clamping/extrapolation of ephemerides.
            lo, hi = self.timeline(np.asarray([self.bounds[:,0], self.bounds[:,1]]))
            for i,v in enumerate(config.visits):
                self.states(v['body'], np.r_[lo[:,i], hi[:,i]])
                if v['kind'] == 'surface':
                    self.transforms(v['body_frame'], np.r_[lo[:,i], hi[:,i]])
        except Exception:
            self.close(); raise

    def close(self):
        if self.lb_batch is not None: self.lb_batch.close(); self.lb_batch = None
        if self.fb_batch is not None: self.fb_batch.close(); self.fb_batch = None
        if self.owns_ephemeris and getattr(self, 'e', None) is not None: self.e.close()

    def __enter__(self): return self
    def __exit__(self, *args): self.close()

    def timeline(self, x):
        x = np.atleast_2d(x)
        arrival = np.empty((len(x), self.L+1)); departure = np.empty_like(arrival)
        arrival[:,0] = departure[:,0] = self.launch_origin+x[:,0]*DAY
        for i in range(self.L):
            arrival[:,i+1] = departure[:,i]+x[:,1+i]*DAY
            departure[:,i+1] = arrival[:,i+1]
            if i+1 in self.stay_columns:
                departure[:,i+1] += x[:,self.stay_columns[i+1]]*DAY
        return arrival, departure

    def _lookup(self, cache, key, function):
        if key in cache:
            cache.move_to_end(key); return cache[key]
        value = function(); cache[key] = value
        if len(cache) > 65536: cache.popitem(last=False)
        return value

    def states(self, body, epochs):
        return np.asarray([self._lookup(self.states_cache, (body,float(t)),
            lambda t=t: np.asarray(self.e.sp.spkezr(body,float(t),self.config.frame,'NONE','SUN')[0])*1000.) for t in epochs])

    def transforms(self, frame, epochs):
        return np.asarray([self._lookup(self.frames_cache, (frame,float(t)),
            lambda t=t: np.asarray(self.e.sp.sxform(frame,self.config.frame,float(t)))) for t in epochs])

    def _lambert(self, rows):
        if self.config.backend == 'cpu': return self.lb.cpu(rows)
        self.lb_batch.upload(rows); self.lb_batch.run(); return self.lb_batch.download()

    def _flyby(self, rows):
        if self.config.backend == 'cpu': return self.fb.cpu(rows)
        self.fb_batch.upload(rows); self.fb_batch.run(); return self.fb_batch.download()

    def block(self, x):
        c = self.config; x = np.atleast_2d(np.asarray(x,float)); N = len(x)
        if x.shape[1] != len(self.bounds) or not np.isfinite(x).all():
            raise ValueError('Invalid mission decision array')
        if np.any(x < self.bounds[:,0]-1e-9) or np.any(x > self.bounds[:,1]+1e-9):
            raise ValueError('Decisions outside configured bounds')
        arrivals, departures = self.timeline(x)
        incoming_states = np.stack([self.states(v['body'], arrivals[:,i]) for i,v in enumerate(c.visits)], axis=1)
        outgoing_states = incoming_states.copy()
        frames = {}
        for i in self.surface_indices:
            outgoing_states[:,i] = self.states(c.visits[i]['body'], departures[:,i])
            frames[i] = (self.transforms(c.visits[i]['body_frame'], arrivals[:,i]),
                         self.transforms(c.visits[i]['body_frame'], departures[:,i]))
        rows = np.zeros((N,self.L,self.K), dtype=INPUT_DTYPE)
        rows['r1'] = outgoing_states[:,:-1,None,:3]
        rows['r2'] = incoming_states[:,1:,None,:3]
        rows['tof'] = (arrivals[:,1:]-departures[:,:-1])[:,:,None]
        rows['mu'] = self.e.mu
        hz = np.cross(rows['r1'],rows['r2'])[...,2]
        for leg in range(self.L):
            choices = [self.fixed[leg]] if self.fixed is not None else self.policy_list
            for k,p in enumerate(choices):
                way = p['orientation']
                rows['long_way'][:,leg,k] = (hz[:,leg,k] < 0 if way == 'prograde' else
                    hz[:,leg,k] > 0 if way == 'retrograde' else way == 'long')
                rows['revolutions'][:,leg,k] = p['revolutions']
                rows['branch'][:,leg,k] = p['branch']
        legs = self._lambert(rows.ravel()).reshape(N,self.L,self.K)
        valid_leg = legs['status'] == 0
        with np.errstate(invalid='ignore', divide='ignore'):
            h = np.cross(rows['r1'],legs['v1'])
            eccentricity = length(np.cross(legs['v1'],h)/self.e.mu-rows['r1']/length(rows['r1'])[...,None])
            perihelion = length(h)**2/(self.e.mu*(1+eccentricity))
            solar_margin = (perihelion-c.minimum_solar_radius_m)/AU
            valid_leg &= solar_margin >= -1e-12
            inf_depart = legs['v1']-outgoing_states[:,:-1,None,3:]
            inf_arrive = legs['v2']-incoming_states[:,1:,None,3:]
            speed_start = length(inf_depart[:,0]); speed_end = length(inf_arrive[:,-1])
            rp0 = c.visits[0]['radius_m']+c.visits[0]['parking_altitude_m']
            rp1 = c.visits[-1]['radius_m']+c.visits[-1]['parking_altitude_m']
            start_cost = np.sqrt(speed_start**2+2*self.mu[0]/rp0)-math.sqrt(self.mu[0]/rp0)
            end_cost = np.sqrt(speed_end**2+2*self.mu[-1]/rp1)-math.sqrt(self.mu[-1]/rp1)
            c3_margin = np.ones_like(speed_start)
            if c.launch_c3_limit_km2_s2 is not None:
                c3_margin = c.launch_c3_limit_km2_s2-speed_start**2/1e6
        start_cost = np.where(valid_leg[:,0],start_cost,np.inf)
        end_cost = np.where(valid_leg[:,-1],end_cost,np.inf)
        start_hard = np.where(c3_margin >= -1e-10,start_cost,np.inf)
        start_score = start_cost+c.violation_weight*np.maximum(0,-c3_margin)*1000
        links = []; hard_links = []; nodes = []
        for node in range(1,self.L):
            v = c.visits[node]
            vi, vo = inf_arrive[:,node-1], inf_depart[:,node]
            pair_valid = valid_leg[:,node-1,:,None] & valid_leg[:,node,None,:]
            with np.errstate(invalid='ignore', divide='ignore', over='ignore'):
                if v['kind'] == 'surface':
                    info = surface_pair(vi,vo,self.mu[node],v['radius_m'],*frames[node])
                else:
                    A = length(vi)[:,:,None]; B = length(vo)[:,None,:]
                    vvi = np.broadcast_to(vi[:,:,None,:],(N,self.K,self.K,3))
                    vvo = np.broadcast_to(vo[:,None,:,:],vvi.shape)
                    required = np.arctan2(length(np.cross(vvi,vvo)),np.sum(vvi*vvo,axis=-1))
                    rp_min = v['radius_m']+v['minimum_altitude_m']; rp_max = v['maximum_periapsis_m']
                    a = np.arcsin(1/(1+rp_min*A*A/self.mu[node]))
                    amin = np.arcsin(1/(1+rp_max*A*A/self.mu[node]))
                    if c.model == 'powered':
                        maximum = a+np.arcsin(1/(1+rp_min*B*B/self.mu[node]))
                        minimum = amin+np.arcsin(1/(1+rp_max*B*B/self.mu[node]))
                    else: maximum,minimum = 2*a,2*amin
                    angular_gap = np.maximum(0,np.maximum(required-maximum,minimum-required))
                    violation = 2*np.sqrt(A*B)*np.sin(angular_gap/2)
                    if c.model == 'ballistic': violation = np.hypot(violation,A-B)
                    packed = np.zeros((N,self.K,self.K),dtype=FLYBY_INPUT)
                    packed['vin'] = vvi; packed['vout'] = vvo
                    packed['reference'] = [0.,0.,1.]
                    pole = np.abs(vvi[...,2]) > .95*A
                    packed['reference'][pole] = [1.,0.,0.]
                    packed['mu'] = self.mu[node]; packed['rp_min'] = rp_min; packed['rp_max'] = rp_max
                    packed['speed_atol'] = c.speed_atol_m_s; packed['speed_rtol'] = c.speed_rtol
                    packed['mode'] = int(Mode.MATCH if c.model == 'ballistic' else Mode.POWERED)
                    solved = self._flyby(packed.ravel()).reshape(N,self.K,self.K)
                    cost = np.zeros_like(violation)
                    if c.model == 'powered':
                        # Lower-cost surrogate on infeasible rows is ONLY a
                        # search aid. Actual exports require native SUCCESS.
                        vp_a = np.sqrt(A*A+2*self.mu[node]/rp_min)
                        vp_b = np.sqrt(B*B+2*self.mu[node]/rp_min)
                        cost = np.where(solved['status']==0,solved['delta_v'],np.abs(vp_a-vp_b))
                    burn_margin = np.ones_like(cost)
                    if c.max_flyby_burn_m_s is not None:
                        burn_margin = c.max_flyby_burn_m_s-cost
                        violation = np.hypot(violation,np.maximum(0,-burn_margin))
                    info = dict(cost=cost,valid=(solved['status']==0)&(burn_margin>=0),violation=violation,
                        native=solved,speed_difference=(B-A),turn_margin=maximum-required,
                        upper_radius_turn_margin=required-minimum,burn_margin=burn_margin)
                    numerical = ~np.isin(solved['status'],[0,4,5,6])
                    pair_valid &= ~numerical
            physical = info['valid'] & pair_valid
            score = info['cost']+c.violation_weight*info['violation']
            links.append(np.where(pair_valid & np.isfinite(score),score,np.inf))
            hard_links.append(np.where(physical,info['cost'],np.inf))
            nodes.append(info)
        score,path = dynamic_path(start_score,links,end_cost)
        hard_score,hard_path = dynamic_path(start_hard,hard_links,end_cost)
        duration = (arrivals[:,-1]-departures[:,0])/DAY
        duration_gap = np.maximum(0,duration-c.max_duration_days)
        score += duration_gap*c.violation_weight*1000
        hard_score = np.where(duration_gap <= 1e-9,hard_score,np.inf)
        return dict(score=score,path=path,feasible_score=hard_score,feasible_path=hard_path,
                    decisions=x,arrivals=arrivals,departures=departures,arrival_states=incoming_states,
                    departure_states=outgoing_states,frames=frames,legs=legs,inputs=rows,nodes=nodes,
                    start_cost=start_cost,end_cost=end_cost,c3_margin=c3_margin,
                    solar_margin=solar_margin,duration_days=duration)

    def fitness(self, x):
        x = np.atleast_2d(np.asarray(x,float)); answer = []
        for lo in range(0,len(x),self.config.batch_candidates):
            block = self.block(x[lo:lo+self.config.batch_candidates])
            answer.extend(np.minimum(block['score']/1000.,1e12))
        return np.asarray(answer)

    def candidate(self, x, require_feasible=True):
        b = self.block(np.asarray(x)[None,:])
        feasible = bool(np.isfinite(b['feasible_score'][0]))
        if require_feasible and not feasible: return None
        path = b['feasible_path'][0] if feasible else b['path'][0]
        return self.describe(b,path,feasible)

    def describe(self, b, path, feasible):
        c = self.config
        legs = []; visits = []; cost_items = []
        for l,k in enumerate(path):
            r = b['legs'][0,l,k]; inp = b['inputs'][0,l,k]
            p = self.fixed[l] if self.fixed is not None else self.policy_list[int(k)]
            legs.append(dict(origin=c.visits[l]['body'],destination=c.visits[l+1]['body'],
                departure_et=float(b['departures'][0,l]),arrival_et=float(b['arrivals'][0,l+1]),
                departure_tdb=calendar(b['departures'][0,l]),arrival_tdb=calendar(b['arrivals'][0,l+1]),
                departure_state_si=b['departure_states'][0,l],arrival_state_si=b['arrival_states'][0,l+1],
                v1_m_s=r['v1'],v2_m_s=r['v2'],policy=p,long_way=bool(inp['long_way']),
                status=int(r['status']),root_status=int(r['root']['status']),
                precision_limited=int(r['root']['precision_limited']),
                solar_perihelion_m=float(c.minimum_solar_radius_m+b['solar_margin'][0,l,k]*AU)))
        first_inf = legs[0]['v1_m_s']-legs[0]['departure_state_si'][3:]
        final_inf = legs[-1]['v2_m_s']-legs[-1]['arrival_state_si'][3:]
        departure = parking(first_inf,self.mu[0],c.visits[0]['radius_m']+c.visits[0]['parking_altitude_m'],False)
        arrival = parking(final_inf,self.mu[-1],c.visits[-1]['radius_m']+c.visits[-1]['parking_altitude_m'],True)
        visits.append(dict(index=0,kind='departure',body=c.visits[0]['body'],mu_m3_s2=self.mu[0],**departure))
        cost_items.append(dict(name='departure parking injection',delta_v_m_s=departure['delta_v_m_s']))
        for j in range(1,self.L):
            v = c.visits[j]; left,right = int(path[j-1]),int(path[j])
            vi = legs[j-1]['v2_m_s']-b['arrival_states'][0,j,3:]
            vo = legs[j]['v1_m_s']-b['departure_states'][0,j,3:]
            info = dict(index=j,kind=v['kind'],body=v['body'],mu_m3_s2=self.mu[j],
                arrival_et=float(b['arrivals'][0,j]),departure_et=float(b['departures'][0,j]),
                vinf_in_m_s=vi,vinf_out_m_s=vo)
            if v['kind'] == 'surface':
                fi,fo = b['frames'][j]
                local = surface_pair(vi[None,None,:],vo[None,None,:],self.mu[j],v['radius_m'],fi,fo,True)
                for key,value in local.items(): info[key] = value[0,0,0]
                info.update(body_frame=v['body_frame'],transform_in=fi[0],transform_out=fo[0],radius_m=v['radius_m'])
                landing = float(length(info['landing_velocity_change_m_s']))
                launch = float(length(info['departure_velocity_change_m_s']))
                cost_items.extend([dict(name=f'{v["body"]} landing',delta_v_m_s=landing),
                                   dict(name=f'{v["body"]} departure',delta_v_m_s=launch)])
            else:
                native = b['nodes'][j-1]['native'][0,left,right]
                for name in native.dtype.names: info[name] = native[name]
                info.update(model=c.model,minimum_radius_m=v['radius_m']+v['minimum_altitude_m'],
                            maximum_radius_m=v['maximum_periapsis_m'])
                cost_items.append(dict(name=f'flyby {j} impulse',delta_v_m_s=float(native['delta_v'])))
            visits.append(info)
        visits.append(dict(index=self.L,kind='arrival',body=c.visits[-1]['body'],mu_m3_s2=self.mu[-1],**arrival))
        cost_items.append(dict(name='arrival parking capture',delta_v_m_s=arrival['delta_v_m_s']))
        total = sum(item['delta_v_m_s'] for item in cost_items)
        return dict(schema_version=1,model='impulsive patched conics; rotating spherical surface stop',
            feasible=feasible,validation_status='unvalidated',search_model=c.model,
            decisions_days=b['decisions'][0],branch_path=list(map(int,path)),
            duration_days=float(b['duration_days'][0]),total_delta_v_m_s=float(total),
            penalized_search_score_km_s=float(b['score'][0]/1000),
            launch_C3_km2_s2=float(np.dot(first_inf,first_inf)/1e6),
            arrival_vinf_m_s=float(length(final_inf)),legs=legs,visits=visits,cost_items=cost_items,
            frame=c.frame,time_scale='TDB',solar_mu_m3_s2=self.e.mu,
            solar_clearance_constraint_m=c.minimum_solar_radius_m,
            endpoint_model='free-plane circular parking at departure and arrival',
            surface_model='spherical rotating body; instantaneous propulsive landing and launch at the same body-fixed site')
