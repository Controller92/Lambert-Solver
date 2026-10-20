"""Complete scalar M>0 two-body transfers with explicit left/right branches.

Positions in metres, positive seconds, mu in m^3/s^2 in one central-body inertial
frame. long_way sets geometric orientation; branch selects the root relative to
x_min. Geometry and stable velocity reconstruction are shared with M=0.
"""
from dataclasses import dataclass, replace

from .lambert_geometry import GeometryResult, GeometryStatus, LambertInput, Vector3, prepare_geometry
from .lambert_multirev import MAX_REVOLUTIONS, MinimumOptions
from .lambert_multirev_root import (
    MultiRootResult, MultiRootStatus, RootBranch, solve_x_multi_rev, solve_x_multi_rev_pair,
)
from .lambert_root import RootOptions
from .lambert_solver import LambertStatus, _reconstruct_velocities


@dataclass(frozen=True)
class MultiLambertResult:
    """Velocities exist only on OK; all failure/conditioning details stay nested.

    root.minimum contains the shared minimum diagnostics. A successful root's
    True/None precision flag is not a velocity error certificate. The native
    C++/CUDA batch ABI still supports M=0 only in this Python increment.
    """
    status: LambertStatus
    geometry: GeometryResult
    branch: RootBranch | None = None
    revolutions: int | None = None
    root: MultiRootResult | None = None
    v1_m_s: Vector3 | None = None
    v2_m_s: Vector3 | None = None


def _geometry(problem):
    if (not isinstance(problem, LambertInput) or type(problem.revolutions) is not int
            or not 1 <= problem.revolutions <= MAX_REVOLUTIONS):
        return GeometryResult(GeometryStatus.INVALID_INPUT)
    # The shape/basis calculation is independent of M. Preserve the original
    # public geometry and M=0 contracts rather than removing their input checks.
    return prepare_geometry(replace(problem, revolutions=0))


def _finish(problem, geometry, root):
    if root.status != MultiRootStatus.OK:
        return MultiLambertResult(LambertStatus.ROOT_FAILURE, geometry, root.branch,
                                  problem.revolutions, root)
    try:
        v1, v2 = _reconstruct_velocities(problem, geometry.geometry, root)
    except (OverflowError, ValueError, ZeroDivisionError):
        return MultiLambertResult(LambertStatus.NUMERICAL_RANGE, geometry, root.branch,
                                  problem.revolutions, root)
    return MultiLambertResult(LambertStatus.OK, geometry, root.branch, problem.revolutions, root, v1, v2)


def solve_multi_rev(problem: LambertInput, branch: RootBranch | str,
                    options: RootOptions = RootOptions(), *,
                    minimum_options: MinimumOptions = MinimumOptions()) -> MultiLambertResult:
    """Return one M>=1 solution, with no fallback to M=0 or the other branch."""
    geometry = _geometry(problem)
    if geometry.status != GeometryStatus.OK:
        return MultiLambertResult(LambertStatus.GEOMETRY_FAILURE, geometry)
    g = geometry.geometry
    root = solve_x_multi_rev(g.time_nondim, g.lambda_, problem.revolutions, branch,
                            options, q=g.q, minimum_options=minimum_options)
    return _finish(problem, geometry, root)


def solve_multi_rev_pair(problem: LambertInput, options: RootOptions = RootOptions(), *,
                         minimum_options: MinimumOptions = MinimumOptions()
                         ) -> tuple[MultiLambertResult, MultiLambertResult]:
    """Return (left,right), reusing geometry and the minimum once per pair.

    Each slot has its own status, residual, iterations, precision flag and
    optional velocities. Infeasible/uncertain requests retain both failure slots.
    """
    geometry = _geometry(problem)
    if geometry.status != GeometryStatus.OK:
        return tuple(MultiLambertResult(LambertStatus.GEOMETRY_FAILURE, geometry, branch)
                     for branch in RootBranch)
    g = geometry.geometry
    roots = solve_x_multi_rev_pair(g.time_nondim, g.lambda_, problem.revolutions,
                                  options, q=g.q, minimum_options=minimum_options)
    return tuple(_finish(problem, geometry, root) for root in roots)
