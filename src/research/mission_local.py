"""Vectorized local patched-conic maneuvers, including a rotating surface stay.

The spherical surface and impulsive burns are explicit model assumptions. A
surface visit uses the SAME body-fixed point at both ends of the stay. Ground
velocities come from the full SPICE state transformation, including rotation.
"""
import numpy as np


def length(v):
    return np.linalg.norm(v, axis=-1)


def half_turn(radius, speed, mu):
    u = radius * speed**2 / mu
    s = 1 / (1 + u)
    c = np.sqrt(u * (u + 2)) / (1 + u)
    return s, c


def surface_pair(vin, vout, mu, radius, transform_in, transform_out, details=False):
    """Arrays vin/vout: (N,K,3). Output matrices: (N,K,K).

    At incoming periapsis Sin·rhat = sin(alpha_in); at outgoing periapsis
    Sout·rhat_out = -sin(alpha_out). Transform the second constraint back to
    arrival time and intersect both planes with the unit sphere. There are
    at most two sites; retain the one with the smaller landing+launch burn.
    """
    a = length(vin)[:, :, None]
    b = length(vout)[:, None, :]
    A = vin[:, :, None, :] / np.maximum(a[..., None], 1e-300)
    B = vout[:, None, :, :] / np.maximum(b[..., None], 1e-300)
    fi, fo = transform_in[:, :3, :3], transform_out[:, :3, :3]
    gi, go = transform_in[:, 3:, :3], transform_out[:, 3:, :3]
    rotation = fo @ fi.transpose(0, 2, 1)
    Q = np.einsum('nji,nabj->nabi', rotation, B)
    cosine = np.sum(A * Q, axis=-1)
    transverse = Q - cosine[..., None] * A
    sine = length(transverse)
    T = transverse / np.maximum(sine[..., None], 1e-30)
    H = np.cross(A, T)
    si, ci = half_turn(radius, a, mu)
    so, co = half_turn(radius, b, mu)
    beta = (-so - cosine * si) / np.maximum(sine, 1e-30)
    squared = si**2 + beta**2
    base = si[..., None] * A + beta[..., None] * T
    height = np.sqrt(np.maximum(0., 1 - squared))
    vp_in = np.sqrt(a*a + 2*mu/radius)
    vp_out = np.sqrt(b*b + 2*mu/radius)
    candidates = []
    for sign in (1., -1.):
        er = base + sign*height[..., None]*H
        site = np.einsum('nji,nabj->nabi', fi, er) * radius
        ro = np.einsum('nij,nabj->nabi', fo, site)
        eri = er
        ero = ro / radius
        ti = (A - si[..., None]*eri) / np.maximum(ci[..., None], 1e-30)
        to = (B + so[..., None]*ero) / np.maximum(co[..., None], 1e-30)
        velocity_in = vp_in[..., None]*ti
        velocity_out = vp_out[..., None]*to
        ground_in = np.einsum('nij,nabj->nabi', gi, site)
        ground_out = np.einsum('nij,nabj->nabi', go, site)
        landing = ground_in-velocity_in
        departure = velocity_out-ground_out
        cost = length(landing)+length(departure)
        candidates.append((cost, site, eri*radius, ro, velocity_in, velocity_out,
                           ground_in, ground_out, landing, departure))
    choose = candidates[1][0] < candidates[0][0]
    cost = np.where(choose, candidates[1][0], candidates[0][0])
    valid = ((squared <= 1+1e-12) & (sine > 1e-12) & (a > 0) & (b > 0)
             & np.isfinite(cost))
    violation = np.sqrt(np.maximum(0, squared-1))*(vp_in+vp_out)
    violation = np.where(sine > 1e-12, violation, 1e9)
    cost = np.where(valid, cost, vp_in+vp_out)
    result = dict(cost=cost, valid=valid, violation=violation,
                  site_margin=1-squared, direction_sine=sine)
    if details:
        names = ('cost', 'site_body_fixed_m', 'rperi_in_m', 'rperi_out_m',
                 'vperi_in_m_s', 'vperi_out_m_s', 'ground_in_m_s', 'ground_out_m_s',
                 'landing_velocity_change_m_s', 'departure_velocity_change_m_s')
        for index, name in enumerate(names[1:], 1):
            result[name] = np.where(choose[..., None], candidates[1][index], candidates[0][index])
    return result


def parking(vinf, mu, radius, incoming):
    """Free-plane circular parking orbit and a tangential injection/capture.

    The endpoint parking plane is a design variable, not a fixed launch-site
    constraint. Return the complete relative states for independent validation.
    """
    v = np.asarray(vinf, float)
    speed = float(length(v)); S = v/speed
    ref = np.eye(3)[np.argmin(np.abs(S))]
    B = ref-np.dot(ref, S)*S; B /= length(B)
    s, c = half_turn(radius, speed, mu)
    radial = s*S+c*B if incoming else -s*S+c*B
    tangent = c*S-s*B if incoming else c*S+s*B
    hyperbolic = np.sqrt(speed*speed+2*mu/radius)*tangent
    circular = np.sqrt(mu/radius)*tangent
    change = circular-hyperbolic if incoming else hyperbolic-circular
    return dict(rperi_m=radius*radial, hyperbolic_velocity_m_s=hyperbolic,
                circular_velocity_m_s=circular, burn_vector_m_s=change,
                delta_v_m_s=float(length(change)), parking_plane_free=True)
