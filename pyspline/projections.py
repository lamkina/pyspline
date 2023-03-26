# Standard Python modules
from typing import Tuple, Optional

# External modules
from .customTypes import CURVETYPE, SURFTYPE

import numpy as np

# Local modules
from . import libspline


def pointCurve(points: np.ndarray, curve: CURVETYPE, nIter, tol: float, u: Optional[np.ndarray] = None) -> float:
    points = np.atleast_2d(points)

    if u is not None:
        u = np.atleast_2d(u)
    else:
        u = -1 * np.ones(len(points))

    # If necessary brute force the starting point
    if np.any(u < 0) or np.any(u > 1):
        curve.computeData()
        u = libspline.pointcurvestart(points.T, curve.uData, curve.data.T)

    D = np.zeros_like(points)
    for i, point in enumerate(points):
        u[i], D[i] = libspline.pointcurve(point, curve.knotVec, curve.degree + 1, curve.ctrlPnts.T, nIter, tol, u[i])

    return u.squeeze(), D.squeeze()


def pointSurface(point: np.ndarray, surface: SURFTYPE, nIter, tol: float) -> Tuple[float, float]:
    pass


def pointVolume():
    pass


def curveCurve(curve1: CURVETYPE, curve2: CURVETYPE, nIter, tol: float) -> float:
    pass


def curveSurface(curve: CURVETYPE, surface: SURFTYPE, nIter, tol: float):
    pass
