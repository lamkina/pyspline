# Standard Python modules
from typing import Optional, Tuple, Union

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
    lb: Optional[Union[float, np.ndarray]] = None,
    ub: Optional[Union[float, np.ndarray]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Project a point onto a curve by finding the minimum distance
    between the point and the curve.

    Parameters
    ----------
    points : np.ndarray
        The points to project onto the curve
    curve : CURVETYPE
        The curve to project onto.
    nIter : int, optional
        The number of optimization iterations for the distance minimzation, by default 10
    nIterLS : int, optional
        The number of line search iterations for the distance minimization, by default 3
    tol : float, optional
        The convergence tolerance of the optimization, by default 1e-6
    rho : float, optional
        The line search backtracking factor, by default 0.5
    alpha0 : float, optional
        The line serach initial step length, by default 1.0
    wolfe : float, optional
        The sufficient decrease factor for the line search, by default 1e-3
    printLevel : int, optional
        The print verbosity level, 1 is all printing, 0 is no printing, by default 0
    u : Optional[np.ndarray], optional
        The initial guesses for the parameteric coordinates.  If the
        initial guesses violate the bounds they will be clipped to the
        bounds, by default None
    lb : Optional[float, np.ndarray], optional
        The lower bounds for the parametric coordinates.  If None, these will be
        set to the lower bound of the knot vector. Enter a single float
        value to set all lower bounds to a constant value.  Enter a numpy
        array to set individual bounds for each coordinate, by default None
    ub : Optional[float, np.ndarray], optional
        The upper bounds for the parametric coordinates. If None, these will be
        set to the upper bound of the knot vector. Enter a single float
        value to set all upper bounds to a constant value.  Enter a numpy
        array to set individual bounds for each coordinate, by default None

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        A tuple of the parmetric coordinates of the closest point on the curve to the
        input points and the distance between the point and the curve.
    """
    points = np.atleast_2d(points)

    # Handle the lower bounds
    if lb is None:
        lb = np.zeros(points.shape[0])
    else:
        if isinstance(lb, float):
            lb = np.full(points.shape[0], lb)
        else:
            lb = np.atleast_1d(lb)

    # Handle the upper bounds
    if ub is None:
        ub = np.ones(points.shape[0])
    else:
        if isinstance(ub, float):
            ub = np.full(points.shape[0], ub)
        else:
            ub = np.atleast_1d(ub)

    # Handle the initial guesses for u
    if u is not None:
        u = np.atleast_2d(u)
    else:
        u = -np.ones(points.shape[0])

    # If necessary brute force the starting point
    if np.any(u < 0) or np.any(u > 1):
        curve.computeData()
        nDim = curve.nDim - 1 if curve.rational else curve.nDim
        data = (
            curve.data[:, :nDim].reshape((len(curve.data), nDim))
            if curve.rational
            else curve.data.reshape((len(curve.data), nDim))
        )
        u = libspline.pointcurvestart(points.T, curve.uData, data.T)

    # Make sure u is inside the bounds
    try:
        u = np.clip(u, lb, ub)
    except ValueError as e:
        print(points.shape)
        print(u.shape)
        print(lb.shape)
        print(ub.shape)
        raise e

    D = np.zeros_like(points)
    ctrlPnts = curve.ctrlPntsW if curve.rational else curve.ctrlPnts

    u, D = libspline.pointcurve(
        points.T,
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
    """Find the point on two curves that minimizes the distance between them.

    Parameters
    ----------
    curve1 : CURVETYPE
        The first curve.
    curve2 : CURVETYPE
        The second curve.
    nIter : int, optional
        The number of optimization iterations for the distance minimzation, by default 10
    nIterLS : int, optional
        The number of line search iterations for the distance minimization, by default 3
    tol : float, optional
        The convergence tolerance of the optimization, by default 1e-6
    rho : float, optional
        The line search backtracking factor, by default 0.5
    alpha0 : float, optional
        The line serach initial step length, by default 1.0
    wolfe : float, optional
        The sufficient decrease factor for the line search, by default 1e-3
    printLevel : int, optional
        The print verbosity level, 1 is all printing, 0 is no printing, by default 0
    u : Optional[np.ndarray], optional
        The initial guesses for the parameteric coordinates.  If the
        initial guesses violate the bounds, they will be clipped to the
        bounds, by default None
    lb : Optional[np.ndarray], optional
        The lower bounds for the parametric coordinates.  If None, these will be
        set to the lower bound of the knot vector, by default None
    ub : Optional[np.ndarray], optional
        The upper bounds for the parametric coordinates. If None, these will be
        set to the upper bound of the knot vector, by default None

    Returns
    -------
    Tuple[float, float, float]
        A tuple of the parametric coordinates on each curve that minimize
        the distance between the two curves and the distance between the
        two curves at the minimum distance point.
    """
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
        ndim1 = curve1.nDim - 1 if curve1.rational else curve1.nDim
        ndim2 = curve2.nDim - 1 if curve2.rational else curve2.nDim
        curve1.computeData()
        curve2.computeData()
        u1, u2 = libspline.curvecurvestart(
            curve1.data[:, :ndim1].T, curve1.uData, curve2.data[:, :ndim2].T, curve2.uData
        )

    u0 = np.array([u1, u2])
    print(u1, u2)

    # Make sure u is inside the bounds
    u0 = np.clip(u0, lb, ub)

    ctrlPnts1 = curve1.ctrlPntsW if curve1.rational else curve1.ctrlPnts
    ctrlPnts2 = curve2.ctrlPntsW if curve2.rational else curve2.ctrlPnts

    u = libspline.curvecurve(
        u0,
        curve1.knotVec,
        curve1.degree,
        ctrlPnts1.T,
        curve2.knotVec,
        curve2.degree,
        ctrlPnts2.T,
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

    return u[0], u[1]


def curveSurface(curve: CURVETYPE, surface: SURFTYPE, nIter, tol: float):
    pass
