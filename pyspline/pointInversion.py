# Standard Python modules
from typing import Iterable, Optional, Tuple, Union

# External modules
import numpy as np

# Local modules
from . import libspline
from .customTypes import CURVETYPE, SURFTYPE


def pointInversionCurve(
    xCoords: Union[np.ndarray, float],
    coord: str,
    curve: CURVETYPE,
    lb: Union[np.ndarray, float],
    ub: Union[np.ndarray, float],
    nIter: int = 10,
    tol: float = 1e-6,
    printLevel: int = 1,
    u0: Optional[Union[np.ndarray, float]] = None,
) -> np.ndarray:
    xCoords = np.atleast_1d(xCoords)

    if coord == "x":
        coordIdx = 1
    elif coord == "y":
        coordIdx = 2
    elif coord == "z":
        coordIdx = 3
    else:
        raise ValueError("Argument 'coord' must be either 'x', 'y', or 'z'.")

    nPts = len(xCoords)

    if isinstance(lb, (float, int)):
        lb = np.array([lb] * nPts)

    if len(lb) != nPts:
        raise ValueError("Lower bound must be a single float or an array of length equal to the number of points.")

    if isinstance(ub, (float, int)):
        ub = np.array([ub] * nPts)

    if len(ub) != nPts:
        raise ValueError("Upper bound must be a single float or an array of length equal to the number of points.")

    if u0 is not None:
        if isinstance(u0, (int, float)):
            u0 = np.atleast_1d(u0)
    else:
        u0 = np.array([0.5] * nPts)

    ctrlPnts = curve.ctrlPntsW if curve.rational else curve.ctrlPnts
    uStar = libspline.pointinvcurve(
        xCoords,
        coordIdx,
        u0,
        lb,
        ub,
        nIter,
        tol,
        printLevel,
        curve.knotVec,
        curve.degree,
        ctrlPnts.T,
    )

    return uStar


def pointInversionSurface(
    x1Coords: Union[float, np.ndarray],
    x2Coords: Union[float, np.ndarray],
    coords: Iterable[str],
    surf: SURFTYPE,
    uLb: Union[np.ndarray, float],
    uUb: Union[np.ndarray, float],
    vLb: Union[np.ndarray, float],
    vUb: Union[np.ndarray, float],
    nIter: int = 10,
    tol: float = 1e-6,
    printLevel: int = 1,
    u0: Optional[Union[np.ndarray, float]] = None,
    v0: Optional[Union[np.ndarray, float]] = None,
) -> Tuple[np.ndarray]:
    # Make sure the coordinate arrays are at least one-dimensional
    x1Coords = np.atleast_1d(x1Coords)
    x2Coords = np.atleast_1d(x2Coords)

    # Error checking on the inputs
    if len(x1Coords) != len(x2Coords):
        raise ValueError("Arguments 'x1Coords' and 'x2Coords' must have the same length")

    nPts = len(x1Coords)
    xStar = np.zeros(nPts * 2)
    xStar[::2] = x1Coords
    xStar[1::2] = x2Coords

    if len(coords) != 2:
        raise ValueError(
            "The argument 'coords' must be a list or tuple of two strings "
            "corresponding to coordinate directions.  i.e ('x', 'y')"
        )

    # Combine the upper and lower bounds for the u and v parameteric coordinates
    if isinstance(uLb, (float, int)):
        uLb = np.array([uLb] * (nPts))

    if len(uLb) != nPts:
        raise ValueError(
            "Lower bound for the 'u' parametric coordinate must be a single "
            "float or an array of length equal to the number of points."
        )

    if isinstance(uUb, (float, int)):
        uUb = np.array([uUb] * (nPts))

    if len(uUb) != nPts:
        raise ValueError(
            "Upper bound for the 'u' parametric coordinate must be a single "
            "float or an array of length equal to the number of points."
        )

    if isinstance(vLb, (float, int)):
        vLb = np.array([vLb] * (nPts))

    if len(vLb) != nPts:
        raise ValueError(
            "Lower bound for the 'v' parametric coordinate must be a single "
            "float or an array of length equal to the number of points."
        )

    if isinstance(vUb, (float, int)):
        vUb = np.array([vUb] * (nPts))

    if len(vUb) != nPts:
        raise ValueError(
            "Upper bound for the 'v' parametric coordinate must be a single "
            "float or an array of length equal to the number of points."
        )

    lb = np.zeros(nPts * 2)
    ub = np.zeros(nPts * 2)

    lb[::2] = uLb
    lb[1::2] = vLb

    ub[::2] = uUb
    ub[1::2] = vUb

    # Determine the coordinate indices
    if coords[0] == "x":
        x1CoordIdx = 1
    elif coords[0] == "y":
        x1CoordIdx = 2
    elif coords[0] == "z":
        x1CoordIdx = 3
    else:
        raise ValueError("Argument 'coord' must be either 'x', 'y', or 'z'.")

    if coords[1] == "x":
        x2CoordIdx = 1
    elif coords[1] == "y":
        x2CoordIdx = 2
    elif coords[1] == "z":
        x2CoordIdx = 3
    else:
        raise ValueError("Argument 'coord' must be either 'x', 'y', or 'z'.")

    coordIdx = np.array([x1CoordIdx, x2CoordIdx])

    # Set the initial guesses for u and v
    if u0 is not None:
        if isinstance(u0, (int, float)):
            u0 = np.atleast_1d(u0)
    else:
        u0 = np.array([0.5] * nPts)

    if v0 is not None:
        if isinstance(v0, (int, float)):
            v0 = np.atleast_1d(v0)
    else:
        v0 = np.array([0.5] * nPts)

    uv0 = np.zeros(nPts * 2)
    uv0[::2] = u0
    uv0[1::2] = v0

    ctrlPnts = surf.ctrlPntsW if surf.rational else surf.ctrlPnts
    uStar = libspline.pointinvsurface(
        xStar,
        coordIdx,
        uv0,
        lb,
        ub,
        nIter,
        tol,
        printLevel,
        surf.uKnotVec,
        surf.vKnotVec,
        surf.uDegree,
        surf.vDegree,
        ctrlPnts.T,
    )

    # Split the uStar array back into u,v coordinates
    u = uStar[::2]
    v = uStar[1::2]

    return u, v
