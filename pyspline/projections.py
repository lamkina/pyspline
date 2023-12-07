# Standard Python modules
from typing import Optional, Tuple

# External modules
import numpy as np

# Local modules
from . import libspline
from .customTypes import CURVETYPE, SURFTYPE


def pointCurve(
    points: np.ndarray,
    curve: CURVETYPE,
    nIter: int = 10,
    nIterLS: int = 3,
    tol: float = 1e-6,
    rho: float = 0.5,
    alpha0: float = 1.0,
    wolfe: float = 1e-3,
    printLevel: int = 0,
    u: Optional[np.ndarray] = None,
    lb: Optional[np.ndarray] = None,
    ub: Optional[np.ndarray] = None,
) -> Tuple[float, float]:
    points = np.atleast_2d(points)

    # Handle the lower bounds
    if lb is None:
        lb = np.zeros(len(points))
    else:
        lb = np.atleast_1d(lb)

    # Handle the upper bounds
    if ub is None:
        ub = np.ones(len(points))
    else:
        ub = np.atleast_1d(ub)

    # Handle the initial guesses for u
    if u is not None:
        u = np.atleast_2d(u)
    else:
        u = -np.ones(points.shape[0])

    # If necessary brute force the starting point
    if np.any(u < 0) or np.any(u > 1):
        curve.computeData(mult=20)
        u = libspline.pointcurvestart(points.T, curve.uData, curve.data.T)

    D = np.zeros_like(points)
    ctrlPnts = curve.ctrlPntsW if curve.rational else curve.ctrlPnts

    u, D = libspline.pointcurve(
        points,
        u,
        curve.knotVec,
        curve.degree,
        ctrlPnts.T,
        curve.rational,
        lb,
        ub,
        tol,
        nIter,
        nIterLS,
        alpha0,
        rho,
        wolfe,
        printLevel,
    )

    return u.squeeze(), D.squeeze()


def pointSurface(point: np.ndarray, surface: SURFTYPE, nIter, tol: float) -> Tuple[float, float]:
    pass


def pointVolume():
    pass


def curveCurve(
    curve1: CURVETYPE,
    curve2: CURVETYPE,
    nIter: int = 10,
    nIterLS: int = 3,
    tol: float = 1e-6,
    rho: float = 0.5,
    alpha0: float = 1.0,
    wolfe: float = 1e-3,
    printLevel: int = 0,
    u: Optional[np.ndarray] = None,
    lb: Optional[np.ndarray] = None,
    ub: Optional[np.ndarray] = None,
) -> float:
    # Handle the lower bounds
    if lb is None:
        lb = np.zeros(2)
    else:
        lb = np.atleast_1d(lb)

    # Handle the upper bounds
    if ub is None:
        ub = np.ones(2)
    else:
        ub = np.atleast_1d(ub)

    # Handle the initial guesses for u
    if u is not None:
        u = np.atleast_2d(u)
    else:
        u = -np.ones(2)

    if np.any(u < 0) or np.any(u > 1):
        curve1.computeData(mult=20)
        curve2.computeData(mult=20)
        u1, u2 = libspline.curvecurvestart(curve1.data.T, curve1.uData, curve2.data.T, curve2.uData)

    u0 = np.array([u1, u2])

    u, d = libspline.curvecurve(
        u0,
        curve1.knotVec,
        curve1.degree,
        curve1.ctrlPnts.T,
        curve2.knotVec,
        curve2.degree,
        curve2.ctrlPnts.T,
        curve1.rational,
        curve2.rational,
        lb,
        ub,
        tol,
        nIter,
        nIterLS,
        alpha0,
        rho,
        wolfe,
        printLevel,
    )

    return u[0], u[1], d


def curveSurface(curve: CURVETYPE, surface: SURFTYPE, nIter, tol: float):
    pass
