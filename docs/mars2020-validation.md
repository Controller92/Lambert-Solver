# Mars 2020: real Horizons inputs and spacecraft comparison

This validation uses NASA/JPL Horizons for the actual Mars 2020/Perseverance
mission, as requested. Public responses are cached in
`validation/data/mars2020/`, with exact request parameters/URLs, retrieval UTC,
API signature and SHA-256 of each returned text. The default validation runs
offline; `--refresh` replaces the snapshots through four sequential API calls.

## Sources and conventions

- [Horizons API documentation](https://ssd-api.jpl.nasa.gov/doc/horizons.html)
  and [system manual](https://ssd.jpl.nasa.gov/horizons/manual.html).
- [NASA trajectory reconstruction report](https://ntrs.nasa.gov/citations/20220000775):
  launch July 30, 2020 at 11:50:00 UTC; atmospheric entry February 18, 2021
  at 20:36:50 spacecraft-event UTC. The latter is not Earth-received landing
  confirmation time.
- [JPL mission overview](https://www.jpl.nasa.gov/news/press_kits/mars_2020/landing/mission/)
  describes cruise trajectory corrections. Their precise effects are not
  reconstructed by this validation.

Targets are Earth center 399, Mars center 499, spacecraft -168 and Sun 10.
The returned sources are DE441, mar099 and Mars2020_merged, respectively.
Horizons identifies its spacecraft data as post-launch fits to tracking data
assembled from several operational trajectory segments. These are stored
mission ephemerides, not an uncertainty-free navigation truth.

All vectors have Sun-center origin (`500@10`), ICRF axes (`REF_PLANE=FRAME`,
`REF_SYSTEM=ICRF`), geometric states (`VEC_CORR=NONE`), and km/km/s output.
The validator checks headers and converts to m/m/s. Solar GM is retrieved from
the Sun object data, giving binary64 1.3271244004193937e20 m^3/s^2.

The vector tables use UTC tags and provide TDB-UTC in seconds. The validator
combines the precise calendar tags with those offsets to obtain dynamical
elapsed times. The redundant exported JDUTC is checked against the calendar
tag; it is not used to introduce its nine-digit Julian-date quantization into
the elapsed time. The fixed 2020-2021 interval has no intervening leap second.
Horizons offsets are printed to microseconds. Calendar, target, origin, frame,
units, output type, time tags, ordering and row counts are checked explicitly.

## Numerical checks of three boundary problems

Each case uses the custom CPU solver, an independently solved Gooding problem
with the same Sun-only model, and DOP853 integration of our departure state.
Both direction flags are mapped to the same prograde physical branch. The
integration uses normalized units, rtol=3e-14, atol=3e-15, max_step=TOF/256.

| Boundary problem | Dynamical duration | Max Gooding velocity difference | Propagated endpoint position error |
| --- | ---: | ---: | ---: |
| Earth at launch to Mars at entry | 203.365856504 days | 1.977e-11 m/s | 0.247 mm |
| Spacecraft Aug 1 to Feb 17 | 200.000000022 days | 1.430e-11 m/s | 0.333 mm |
| Spacecraft Oct 15 to Dec 1 | 47.000000008 days | 1.128e-11 m/s | 0.107 mm |

All use short arcs, converge in three completed updates, have computed relative
TOF residual at most 1.818e-16 and no root precision warning. Maximum propagated
arrival velocity mismatch is 2.988e-11 m/s. Numerical acceptance bounds are
1e-6 m/s for Gooding and propagated velocity comparisons, and 10 m for position
closure. No unexpected failures occurred.

The idealized planet-center transfer gives departure v-infinity 3.81713511 km/s,
arrival v-infinity 2.55203041 km/s and C3 14.5705205 km^2/s^2. These describe
the computed planet-center two-body transfer. They are not a reconstruction of
the launch vehicle's actual injection or a prescribed entry B-plane target.

## Comparison with the actual spacecraft states

A Lambert fit between the spacecraft's Aug 1 and Feb 17 positions is propagated
through intermediate Horizons epochs. Its endpoints are fitted by construction;
interior positions and all spacecraft velocities are independent comparisons.

| UTC sample | Position difference | Velocity difference |
| --- | ---: | ---: |
| 2020-Aug-01 | 0 km | 174.363253 m/s |
| 2020-Aug-15 | 47,014.3802 km | 11.7934357 m/s |
| 2020-Sep-15 | 53,979.3180 km | 6.67578068 m/s |
| 2020-Oct-15 | 50,703.6827 km | 4.79131355 m/s |
| 2020-Dec-01 | 39,891.9961 km | 4.53853351 m/s |
| 2021-Jan-15 | 23,046.2653 km | 5.85944283 m/s |
| 2021-Feb-17 | 0.000000333 km | 37.7821375 m/s |

On the shorter Oct 15 to Dec 1 fit, endpoint velocity differences from the
spacecraft ephemeris are 0.439817643 and 0.187872493 m/s.

These differences do not meet a millimetre-accuracy claim for the real mission,
and no such claim is made. The Sun-only boundary-value problem is verified
numerically to the tolerances above; the stored mission trajectory follows
a different dynamical history. Planetary perturbations, motion of the Sun and
trajectory corrections are absent from this core. Their individual contributions
have not been isolated. The long-cruise mismatch establishes that a two-body fit
alone does not reproduce the mission. Perturbed propagation and differential
correction remain separate downstream work.

## Reproduce

```bash
.venv-validation/bin/python validation/validate_mars2020.py
# Explicitly refresh the cached public ephemerides when wanted:
.venv-validation/bin/python validation/validate_mars2020.py --refresh
```

There is no network access, ephemeris parsing or date conversion inside the
custom mathematical solver. The Horizons fixture and script are validation-only.
