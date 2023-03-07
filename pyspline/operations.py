# Standard Python modules
from copy import deepcopy
from typing import List, Tuple, Union

# External modules
import numpy as np

# Local modules
from . import libspline
from .bspline import BSplineCurve, BSplineSurface, BSplineVolume
from .custom_types import GEOTYPE
from .utils import checkInput


def degreeElevation(degree: int, ctrlPnts: np.ndarray, num: int):
    pass


def degreeReduction(degree: int, ctrlPnts: np.ndarray):
    pass


def knotRefinement(degree: int, knotVec: np.ndarray, ctrlPnts: np.ndarray, **kwargs):
    pass


def getMultiplicity(knot: float, knotVec: np.ndarray, tol: float = 1e-10) -> int:
    """Finds the multiplicity of a knot in a given knot vector within
    a tolerance.

    Parameters
    ----------
    knot : float
        The knot.
    knotVec : np.ndarray
        The knot vector.
    tol : float, optional
        The tolerance below which two knots are considered equal, by default 1e-10

    Returns
    -------
    int
        The knot multiplicity.
    """
    nk = 0
    for k in knotVec:
        if abs(k - knot) <= tol:
            nk += 1

    return nk


def curveKnotInsertion(
    degree: int, knotVec: np.ndarray, ctrlPts: np.ndarray, knot: float, num: int
) -> Tuple[int, np.ndarray, np.ndarray]:
    # Initialize useful vars
    nDim = ctrlPts.shape[-1]
    nctl = len(ctrlPts)
    nq = nctl + num

    # Allocate new control points
    ctrlPtsNew = np.zeros((nq, nDim))

    # Allocate a temporary array
    temp = np.zeros((degree + 1, nDim))

    # First we need to find the multiplicity of the knot, denoted by "s"
    s = libspline.multiplicity(knot, knotVec, nctl, degree)

    # Next we need to find the knot span
    span = libspline.findspan(knot, degree, knotVec, nctl)

    # Load the new knot vector
    knotVecNew = np.zeros(len(knotVec) + num)
    knotVecNew[: span + 1] = knotVec[: span + 1]
    knotVecNew[span + 1 : span + num + 1] = knot
    knotVecNew[span + num + 1 : nctl + num + degree + 1] = knotVec[span + 1 : nctl + degree + 1]

    # Copy over the unaltered control points
    ctrlPtsNew[: span - degree + 1, :] = ctrlPts[: span - degree + 1, :]
    ctrlPtsNew[span - s + num : nctl + num, :] = ctrlPts[span - s : nctl, :]

    for i in range(0, degree - s + 1):
        temp[i, :] = deepcopy(ctrlPts[span - degree + 1, :])

    # Insert the knot "num" times
    for j in range(1, num + 1):
        L = span - degree + j
        for i in range(0, degree - j - s + 1):
            alpha = (knot - knotVec[L + i]) / (knotVec[i + span + 1] - knotVec[L + i])
            temp[i, :] = alpha * temp[i + 1, :] + (1.0 - alpha) * temp[i, :]

        ctrlPtsNew[L] = temp[0, :]
        ctrlPtsNew[span + num - j - s, :] = temp[degree - j - s, :]

    # Load the remaining control points
    L = span - degree + num
    for i in range(L + 1, span - s):
        ctrlPtsNew[i, :] = temp[i - L, :]

    newSpan = libspline.findspan(knot, degree, knotVecNew, nctl + num)
    return newSpan, knotVecNew, ctrlPtsNew


def insertKnot(geo: GEOTYPE, param: List[float], num: List[int]) -> Tuple[int]:
    """Insert a knot n times into a curve, surface, or volume along either
    'u', 'v', or 'w' parameteric directions.

    .. code-block: python

        # For curves
        insertKnot(curve, [0.25], [1])

        # For surfaces
        insertKnot(curve, [0.25, None], [1, 0])

        # For surfaces
        insertKnot(curve, [0.25, 0.75], [1, 2])

        # For volumes
        insertKnot(curve, [0.25, None, None], [2, 0, 0])

    Parameters
    ----------
    geo : GEOTYPE
        A geometric object (BSplineCurve, BSplineSurface, BSplineVolume)
    param : List
        The parameter coordinates in [u, v, w] format.
    num : List[int]
        The number of knot insertions in [num_u, num_v, num_w] format

    Returns
    -------
    Tuple[int]
        The actual number of times the knot was inserted and the breakpoint
        location of the inserted knot.

    Raises
    ------
    ValueError
        If the 'num' argument is not the same length as the 'param' argument.
    """
    nDim = len(param)

    if len(num) != nDim:
        raise ValueError(
            "Argument 'num' must be the same length as the dimensions of the parametric coordinates 'param'."
        )

    if isinstance(geo, BSplineCurve):
        if param[0] is not None and num[0] > 0:
            knot = checkInput(param[0], "param[0]", float, 0)
            num = checkInput(num[0], "num[0]", int, 0)

            # First we need to find the multiplicity of the knot, denoted by "s"
            s = libspline.multiplicity(knot, geo.knotVec, geo.nCtlnctl, geo.degree)

            # Check if we can add the requested number of knots
            if num > geo.degree - s:
                raise ValueError(f"Knot: {knot} cannot be inserted {num} times")

            if geo._rational:
                newSpan, knotVecNew, ctrlPntsWNew = curveKnotInsertion(
                    geo.degree, geo.knotVec, geo.ctrlPntsW, knot, num
                )
                geo.ctrlPntsW = ctrlPntsWNew
            else:
                newSpan, knotVecNew, ctrlPntsNew = curveKnotInsertion(geo.degree, geo.knotVec, geo.ctrlPnts, knot, num)
                geo.ctrlPnts = ctrlPntsNew

            geo.knotVec = knotVecNew

            return newSpan

    elif isinstance(geo, BSplineSurface):
        # u-direction
        if param[0] is not None and num[0] > 0:
            knot = checkInput(param[0], "param[0]", float, 0)
            num = checkInput(num[0], "num[0]", int, 0)

            if s <= 0.0:
                return
            elif s >= 1.0:
                return

            actualR, knotVecNew, ctrlPntsNew, breakPnt = libspline.insertknot(
                s, r, geo.uKnotVec, geo.uDegree, geo.ctrlPnts[:, 0].T
            )
            newctrlPnts = np.zeros((geo.nCtlu + actualR, geo.nCtlv, geo.nDim))

            for j in range(geo.nCtlv):
                actualR, knotVecNew, ctlPntSlice, breakPnt = libspline.insertknot(
                    s, r, geo.uKnotVec, geo.uDegree, geo.ctrlPnts[:, j].T
                )
                newctrlPnts[:, j] = ctlPntSlice[:, 0 : geo.nCtlu + actualR].T

            geo.uKnotVec = knotVecNew[0 : geo.nCtlu + geo.uDegree + actualR]
            geo.nCtlu = geo.nCtlu + actualR

            geo.ctrlPnts = newctrlPnts

        # v-direction
        if param[1] is not None and num[1] > 0:
            s = checkInput(param[1], "s", float, 0)
            r = checkInput(num[1], "r", int, 0)

            if s <= 0.0:
                return
            elif s >= 1.0:
                return

            actualR, knotVecNew, ctrlPntsNew, breakPnt = libspline.insertknot(
                s, r, geo.vKnotVec, geo.vDegree, geo.ctrlPnts[0, :].T
            )

            newCoef = np.zeros((geo.nCtlu, geo.nCtlv + actualR, geo.nDim))

            for i in range(geo.nCtlu):
                actualR, knotVecNew, coefSlice, breakPnt = libspline.insertknot(
                    s, r, geo.vKnotVec, geo.vDegree, geo.ctrlPnts[i, :].T
                )
                newCoef[i, :] = coefSlice[:, 0 : geo.nCtlv + actualR].T

            geo.vKnotVec = knotVecNew[0 : geo.nCtlv + geo.vDegree + actualR]
            geo.nCtlv = geo.nCtlv + actualR

            geo.ctrlPnts = newctrlPnts

        # Convert breakPnt back to zero based ordering
        return actualR, breakPnt - 1

    elif isinstance(geo, BSplineVolume):
        pass


def splitSurface(surf: BSplineSurface, param: float, direction: str) -> Tuple[BSplineSurface]:
    """
    Split surface into two surfaces at parametric location s

    Parameters
    ----------
    surf : BSplineSurface
        The surface to be split.
    param : float
        Parametric position along 'direction' to split
    direction : str
        Parameteric direction along which to split. Either 'u' or 'v'.

    Returns
    -------
    surf1 : pySpline.surface
        Lower part of the surface

    surf2 : pySpline.surface
        Upper part of the surface
    """
    if direction not in ["u", "v"]:
        raise ValueError("'direction' must be on of 'u' or 'v'")

    # Check the bounds
    if param <= 0:
        return None, BSplineSurface(
            surf.uDegree, surf.vDegree, surf.ctrlPnts.copy(), surf.uKnotVec.copy(), surf.vKnotVec.copy()
        )

    if param >= 1.0:
        return (
            BSplineSurface(
                surf.uDegree, surf.vDegree, surf.ctrlPnts.copy(), surf.uKnotVec.copy(), surf.vKnotVec.copy()
            ),
            None,
        )

    # Splitting in the u-direction
    if direction == "u":

        _, breakPnt = insertKnot(surf, [param, None], [surf.uDegree - 1, 0])

        # Break point is to the right so we need to adjust the counter to the left
        breakPnt = breakPnt - surf.uDegree + 2
        knot = surf.uKnotVec[breakPnt]

        # Process the knot vector
        knotVec1 = np.hstack((surf.uKnotVec[0 : breakPnt + surf.uDegree - 1].copy(), knot)) / knot
        knotVec2 = (np.hstack((knot, surf.uKnotVec[breakPnt:].copy())) - knot) / (1.0 - knot)

        ctrlPnts1 = surf.ctrlPnts[0:breakPnt, :, :].copy()
        ctrlPnts2 = surf.ctrlPnts[breakPnt - 1 :, :, :].copy()

        return (
            BSplineSurface(surf.uDegree, surf.vDegree, ctrlPnts1, knotVec1, surf.vKnotVec),
            BSplineSurface(surf.uDegree, surf.vDegree, ctrlPnts2, knotVec2, surf.vKnotVec),
        )

    # Splitting in the v-direction
    elif direction == "v":
        _, breakPnt = insertKnot(surf, [None, param], [0, surf.vDegree - 1])

        # Break point is to the right so we need to adjust the counter to the left
        breakPnt = breakPnt - surf.vDegree + 2

        knot = surf.vKnotVec[breakPnt]

        # Process knot vectors:
        knotVec1 = np.hstack((surf.vKnotVec[0 : breakPnt + surf.vDegree - 1].copy(), knot)) / knot
        knotVec2 = (np.hstack((knot, surf.vKnotVec[breakPnt:].copy())) - knot) / (1.0 - knot)

        ctrlPnts1 = surf.ctrlPnts[:, 0:breakPnt, :].copy()
        ctrlPnts2 = surf.ctrlPnts[:, breakPnt - 1 :, :].copy()

        return (
            BSplineSurface(surf.uDegree, surf.vDegree, ctrlPnts1, surf.uKnotVec, knotVec1),
            BSplineSurface(surf.uDegree, surf.vDegree, ctrlPnts2, surf.uKnotVec, knotVec2),
        )


def windowSurface(
    surf: BSplineSurface,
    uvLow: Union[List, np.ndarray],
    uvHigh: Union[List, np.ndarray],
) -> BSplineSurface:
    """Create a surface that is windowed by the rectangular
    parametric range defined by uvLow and uvHigh.  This uses the
    :func: `splitSurface` function.

    Parameters
    ----------
    surf: BSplineSurface
        The surface to be windowed
    uvLow : list or array of length 2
        (u,v) coordinates at the bottom left corner of the parameteric
        box
    uvHigh : list or array of length 2
        (u,v) coordinates at the top left corner of the parameteric
        box

    Returns
    -------
    surf : BSplineSurface
        A new surface defined only on the interior of uvLow -> uvHigh
    """
    # Do u-low split
    _, surf = splitSurface(surf, uvLow[0], "u")

    # Do u-high split (and re-normalize the split coordinate)
    surf, _ = splitSurface(surf, (uvHigh[0] - uvLow[0]) / (1.0 - uvLow[0]), "u")

    # Do v-low split
    _, surf = splitSurface(surf, uvLow[1], "v")

    # Do v-high split (and re-normalize the split coordinate)
    surf, _ = splitSurface(surf, (uvHigh[1] - uvLow[1]) / (1.0 - uvLow[1]), "v")

    return surf


def reverseCurve(curve: BSplineCurve) -> None:
    """Reverse the direction of the curve.

    Parameters
    ----------
    curve : BSplineCurve
        The curve to be reversed.
    """
    curve.ctrlPnts = curve.ctrlPnts[::-1, :]
    curve.knotVec = 1 - curve.ctrlPnts[::-1]


def splitCurve(curve: BSplineCurve, u: float) -> Tuple[BSplineCurve]:
    """
    Split the curve at parametric position u. This uses the
    :func:`insertKnot` function.

    Parameters
    ----------
    curve: BSplineCurve
        The curve being split.
    u : float
        Parametric position to insert knot.

    Returns
    -------
    Tuple[BSplineCurve, BSplineCurve]
        curve1 : BSplineCurve or None
            Curve from s=[0, u]
        curve2 : BSplineCurve or None
            Curve from s=[u, 1]

    Notes
    -----
    curve1 and curve2 may be None if the parameter u is outside
    the range of (0, 1)
    """
    u = checkInput(u, "u", float, 0)

    if u <= 0.0:
        return None, BSplineCurve(curve.degree, curve.knotVec.copy(), curve.ctrlPnts.copy())

    if u >= 1.0:
        return BSplineCurve(curve.degree, curve.knotVec.copy(), curve.ctrlPnts.copy()), None

    _, breakPnt = insertKnot(curve, [u, None], [curve.degree - 1, 0])

    breakPnt = breakPnt - curve.degree + 2

    # Process knot vectors
    knot = curve.knotVec[breakPnt]
    newKnotVec1 = np.hstack((curve.knotVec[0 : breakPnt + curve.degree - 1].copy(), knot)) / knot
    newKnotVec2 = (np.hstack((knot, curve.knotVec[breakPnt:].copy())) - knot) / (1.0 - knot)

    ctrlPnts1 = curve.ctrlPnts[0:breakPnt, :].copy()
    ctrlPnts2 = curve.ctrlPnts[breakPnt - 1 :, :].copy()

    return BSplineCurve(curve.degree, newKnotVec1, ctrlPnts1), BSplineCurve(curve.degree, newKnotVec2, ctrlPnts2)


def getSurfaceBasisPt(
    surf: BSplineSurface,
    u: float,
    v: float,
    vals: np.ndarray,
    iStart: int,
    colInd: np.ndarray,
    lIndex: np.ndarray,
):
    """This function should only be called from pyGeo. The purpose is to
    compute the basis function for a u, v point and add it to pyGeo's
    global dPt/dCoef matrix. vals, row_ptr, col_ind is the CSR data and
    lIndex in the local -> global mapping for this surface.

    Parameters
    ----------
    surf : BSplineSurface
        The B-Spline surface.
    u : float
        Parametric coordinate in the 'u' direction.
    v : float
        Parametric coordinate in the 'v' direction.
    vals : np.ndarray
        CSR matrix values.
    iStart : int
        Starting index of the surface data.
    colInd : np.ndarray
        Column indices of the CSR matrix.
    lIndex : np.ndarray
        Local to global mapping array for this surface.

    Returns
    -------
    Tuple[np.ndarray]
        A tuple of the vals and colInd for the CSR matrix
    """
    return libspline.getbasisptsurface(
        u, v, surf.uKnotVec, surf.vKnotVec, surf.uDegree, surf.vDegree, vals, colInd, iStart, lIndex
    )


def computeCurveData(curve: BSplineCurve):
    curve.calcInterpolatedGrevillePoints()
    return curve(curve.sdata)


def computeSurfaceData(surf: BSplineSurface):
    surf.edgeCurves[0].calcInterpolatedGrevillePoints()
    udata = surf.edgeCurves[0].sdata
    surf.edgeCurves[2].calcInterpolatedGrevillePoints()
    vdata = surf.edgeCurves[2].sdata
    V, U = np.meshgrid(vdata, udata)
    return surf(U, V)
