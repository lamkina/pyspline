# Standard Python modules
from typing import Optional, Tuple

# External modules
import numpy as np

# Local modules
from . import libspline
from .customTypes import CURVETYPE, SURFTYPE


def pointCurve(
    points: np.ndarray, curve: CURVETYPE, nIter: int = 10, tol: float = 1e-6, u: Optional[np.ndarray] = None
) -> Tuple[float, float]:
    points = np.atleast_2d(points)

    if u is not None:
        u = np.atleast_2d(u)
    else:
        if points.shape[0] == 1:
            u = -1 * np.ones(len(points.flatten()))
        else:
            u = -1 * np.ones(len(points))

    # If necessary brute force the starting point
    if np.any(u < 0) or np.any(u > 1):
        curve.computeData()
        u = libspline.pointcurvestart(points, curve.uData, curve.data.reshape((curve.nDim, len(curve.data))))

    if points.shape[0] == 1:
        points = points.flatten()
    D = np.zeros_like(points)
    for i, point in enumerate(points):
        ctrlPnts = curve.ctrlPntsW if curve.rational else curve.ctrlPnts
        u[i], D[i] = libspline.pointcurve(
            np.array(point),
            curve.knotVec,
            curve.degree,
            ctrlPnts.reshape((curve.nDim, curve.nCtl)),
            nIter,
            tol,
            u[i],
            curve.rational,
        )

    return u.squeeze(), D.squeeze()


def pointSurface(point: np.ndarray, surface: SURFTYPE, nIter, tol: float) -> Tuple[float, float]:
    pass


def pointVolume():
    pass


def curveCurve(curve1: CURVETYPE, curve2: CURVETYPE, nIter, tol: float, u: float = -1, t: float = -1) -> float:
    """
    Find the minimum distance between this curve (self) and a
    second curve passed in (inCurve)

    Parameters
    ----------
    curve1 : BSplineCurve or NURBSCurve
        Base curve for the projection
    curve2 : BSplineCurve or NURBSCurve
        Second curve to use
    nIter : int
        Maximum number of Newton iterations to perform.
    tol : float
        Desired parameter tolerance.
    u : float
        Initial guess for curve1 (this curve class)
    t : float
        Initial guess for inCurve (curve passed in )

    Returns
    -------
    float
        Parametric position on curve1 (this class)
    float
        Parametric position on curve2 (inCurve)
    float
        Minimum distance between this curve and inCurve. It
        is equivalent to ||self(s) - inCurve(t)||_2.
    """
    if u < 0 or u > 1 or t < 0 or t > 1:
        curve1.computeData()
        curve2.computeData()
        u, t = libspline.curvecurvestart(curve1.data.T, curve1.uData, curve2.data.T, curve2.uData)

    u, t, d = libspline.curvecurve(
        curve1.knotVec,
        curve1.degree,
        curve1.ctrlPnts.T,
        curve2.knotVec,
        curve2.degree,
        curve2.ctrlPnts.T,
        nIter,
        tol,
        u,
        t,
    )

    return u, t, d


def curveSurface(curve: CURVETYPE, surface: SURFTYPE, nIter, tol: float):
    pass
