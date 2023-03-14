# Standard Python modules
from copy import deepcopy
from typing import List, Optional, Tuple, Union

# External modules
import numpy as np

# Local modules
from . import compatibility, libspline, utils
from .bspline import BSplineCurve, BSplineSurface, BSplineVolume
from .customTypes import CURVETYPE, GEOTYPE
from .nurbs import NURBSCurve, NURBSSurface
from .utils import checkInput


def decomposeCurve(curve: CURVETYPE) -> List[CURVETYPE]:
    """Decompose the curve into Bezier segments of the same degree.

    This does not modify the input curve because it operates on a copy of the curve.

    Adapted from: "NURBS-Python: An open-source object-oriented NURBS modeling framework in Python",
    by Bingol and Krishnamurthy

    Parameters
    ----------
    curve : Union[BSplineCurve, NURBSCurve]
        Input spline curve to be decomposed.

    Returns
    -------
    List[Union[BSplineCurve, NURBSCurve]]
        List of Bezier curve segments of the same degree as the input curve.
    """
    curve = deepcopy(curve)
    internalKnots = curve.knotVec[curve.degree + 1 : -(curve.degree + 1)]
    newCurves = []
    while len(internalKnots) > 0:
        knot = internalKnots[0]
        curve1, curve2 = splitCurve(curve, knot)
        newCurves.append(curve1)
        curve = curve2
        internalKnots = curve.knotVec[curve.degree + 1 : -(curve.degree + 1)]
    newCurves.append(curve)
    return newCurves


def combineCurves(
    curveList: List[CURVETYPE], tol: float = 1e-8, check: bool = False
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Combines a list of input curves into a single curve.

    Adapted from: "NURBS-Python: An open-source object-oriented NURBS modeling framework in Python",
    by Bingol and Krishnamurthy

    If `check` is set to `True`, this function will check the validity of the input curves.  The validity
    check ensures the end point of a curve is compatible with the initial point of the following curve.

    Parameters
    ----------
    curveList : List[Union[BSplineCurve, NURBSCurve]]
        The list of input curves.
    tol : float, optional
        The tolerance for the validity check, by default 1e-8
    check : bool, optional
        Flag to run validity check, by default False

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        The knot vector, control points, weights, and superfluous knots that need to
        be removed.

    Raises
    ------
    ValueError
        If any pair of successive curves are not compatible. i.e the distance between
        the last and first control points are larger than the `tol` argument.
    """
    if check:
        # Check the distance between the beginning and ending points of successive curves
        for iCurve in range(len(curveList) - 1):
            dist = np.linalg.norm(curveList[iCurve + 1].ctrlPnts[0] - curveList[iCurve].ctrlPnts[-1])
            if dist > tol:
                raise ValueError(
                    f"Curve {iCurve} and Curve {iCurve + 1} are not close enough to combine. Must be within {tol}."
                )

    # Allocate new knot vector, control points, and weights
    knotVec = []
    ctrlPnts = []
    weights = []
    knotVecConnected = []
    pdomainEnd = 0

    for curve in curveList:
        if not knotVec:
            # Remove the last superfluous knot knot
            knotVec += list(curve.knotVec[: -(curve.degree + 1)])
            ctrlPnts += list(curve.ctrlPnts)

            if curve.rational:
                weights += list(curve.weights)
            else:
                tempWeights = [1.0 for _ in range(curve.nCtl)]
                weights += tempWeights
        else:
            tempKnotVec = [pdomainEnd + k for k in curve.knotVec[1 : -(curve.degree + 1)]]
            knotVec += tempKnotVec
            ctrlPnts += list(curve.ctrlPnts[1:])
            if curve.rational:
                weights += list(curve.weights[1:])
            else:
                tempWeights = [1.0 for _ in range(curve.nCtl - 1)]
                weights += tempWeights

        pdomainEnd += curve.knotVec[-1]
        knotVecConnected.append(pdomainEnd)

    # Fix curve by appending the lat knot to the end
    knotVec += [pdomainEnd for _ in range(curve.degree + 1)]
    knotVecConnected.pop()

    return np.array(knotVec), np.array(ctrlPnts), np.array(weights), np.array(knotVecConnected)


def elevateDegree(geo: GEOTYPE, param: Union[List, np.ndarray, Tuple]) -> None:
    """Elevate the degree of a BSpline or NURBS geometric object.

    Adapted python implementation from: "NURBS-Python: An open-source object-oriented NURBS modeling framework in Python",
    by Bingol and Krishnamurthy

    Parameters
    ----------
    geo : GEOTYPE
        BSpline or NURBS geometry object
    param : Union[List, np.ndarray, Tuple]
        The number of times to elevate the degree.
    """
    if isinstance(geo, BSplineCurve):
        internalKnots = np.unique(geo.knotVec[geo.degree + 1 : -(geo.degree + 1)])
        multList = []
        for intKnot in internalKnots:
            s = utils.multiplicity(intKnot, geo.knotVec, geo.nCtl, geo.degree)
            multList.append(s)

        # Decompose the curve into bezier segments
        curveList = decomposeCurve(geo)

        if param[0] > 0:
            for curve in curveList:
                ctrlPnts = curve.ctrlPntsW if curve.rational else curve.ctrlPnts
                newCtrlPnts = utils.elevateDegreeCurve(curve.degree, ctrlPnts, num=param[0])
                curve.degree += param[0]
                if curve.rational:
                    curve.ctrlPntsW = newCtrlPnts
                else:
                    curve.ctrlPnts = newCtrlPnts

                knotStart = np.repeat(curve.knotVec[0], param[0])
                knotEnd = np.repeat(curve.knotVec[-1], param[0])
                curve.knotVec = np.concatenate([knotStart, curve.knotVec, knotEnd])

            nd = geo.degree + param[0]

            num = geo.degree + 1

        # Combine the Bezier curve segments back into a full
        knotVec, ctrlPnts, weights, knotList = combineCurves(curveList, check=False)

        # Set the control points depending on rational/non-rational curve
        ctrlPnts = compatibility.combineCtrlPnts(ctrlPnts, weights) if geo.rational else ctrlPnts

        # Apply knot removal
        for knot, s in zip(knotList, multList):
            span = utils.findSpan(knot, nd, knotVec, len(ctrlPnts))
            ctrlPnts = utils.removeKnotCtrlPnts(nd, knotVec, ctrlPnts, knot, num=num - s)
            knotVec = utils.removeKnotKV(knotVec, span, num - s)

        # Update the input curve
        geo.degree = nd
        if geo.rational:
            geo.ctrlPntsW = ctrlPnts
        else:
            geo.ctrlPnts = ctrlPnts

        geo.knotVec = knotVec


def reduceDegree(geo: GEOTYPE, param: List[int]) -> None:
    """Reduces the degree of a spline geometry by an amount specified
    by `param`.

    Adapted python implementation from: "NURBS-Python: An open-source object-oriented NURBS modeling framework in Python",
    by Bingol and Krishnamurthy

    Parameters
    ----------
    geo : GEOTYPE
        A geomtry object.
    param : Union[List, np.ndarray, Tuple]
        The amount to decrease the degree.
    """
    if isinstance(geo, BSplineCurve):
        internalKnots = np.unique(geo.knotVec[geo.degree + 1 : -(geo.degree + 1)])
        multList = []
        for intKnot in internalKnots:
            s = utils.multiplicity(intKnot, geo.knotVec, geo.nCtl, geo.degree)
            multList.append(s)

        # Decompose the curve into bezier segments
        curveList = decomposeCurve(geo)

        for curve in curveList:
            ctrlPnts = curve.ctrlPntsW if curve.rational else curve.ctrlPnts
            newCtrlPnts = utils.reduceDegreeCurve(curve.degree, ctrlPnts)
            curve.degree -= 1

            if curve.rational:
                curve.ctrlPntsW = newCtrlPnts
            else:
                curve.ctrlPnts = newCtrlPnts

            curve.knotVec = curve.knotVec[1:-1]

        # Compute new degree
        nd = geo.degree - 1

        # Number of knot removals
        num = geo.degree - 1

        # Combine the Bezier curve segments back into a full
        knotVec, ctrlPnts, weights, knotList = combineCurves(curveList, check=False)

        # Set the control points depending on rational/non-rational curve
        ctrlPnts = compatibility.combineCtrlPnts(ctrlPnts, weights) if geo.rational else ctrlPnts

        # Apply knot removal
        for knot, s in zip(knotList, multList):
            span = utils.findSpan(knot, nd, knotVec, len(ctrlPnts))
            ctrlPnts = utils.removeKnotCtrlPnts(nd, knotVec, ctrlPnts, knot, num=num - s)
            knotVec = utils.removeKnotKV(knotVec, span, num - s)

        # Update the input curve
        geo.degree = nd
        if geo.rational:
            geo.ctrlPntsW = ctrlPnts
        else:
            geo.ctrlPnts = ctrlPnts

        geo.knotVec = knotVec


def refineKnotVector(geo: GEOTYPE, param: Union[List, np.ndarray, Tuple], density: int = 1, check: bool = True):
    if check:
        if len(param) != geo.pDim:
            raise ValueError(
                f"The length of the param array ({len(param)}) must be equal to the "
                f"number of parameteric dimensions ({geo.pDim})"
            )

    if isinstance(geo, BSplineCurve):
        if param[0] > 0:
            ctrlPnts = geo.ctrlPntsW if geo.rational else geo.ctrlPnts
            newCtrlPnts, newKnotVec = utils.refineKnotCurve(geo.degree, geo.knotVec, ctrlPnts, density)

            if geo.rational:
                geo.ctrlPntsW = newCtrlPnts
            else:
                geo.ctrlPnts = newCtrlPnts

            geo.knotVec = newKnotVec

    if isinstance(geo, BSplineSurface):
        # TODO: AL, add surface knot refinement
        pass


def insertKnot(geo: GEOTYPE, param: List[float], num: List[int], check: bool = True) -> Tuple[int]:
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
            s = utils.multiplicity(knot, geo.knotVec, geo.nCtl, geo.degree)

            # Get the knot span
            span = utils.findSpan(knot, geo.degree, geo.knotVec, geo.nCtl)

            newKnotVec = utils.insertKnotKV(geo.knotVec, knot, num, span)

            # Check if we can add the requested number of knots
            if check:
                if num > geo.degree - s:
                    raise ValueError(f"Knot: {knot} cannot be inserted {num} times")

            if geo.rational:
                ctrlPntsWNew = utils.insertKnotCP(geo.degree, geo.knotVec, geo.ctrlPntsW, knot, num, s, span)
                geo.ctrlPntsW = ctrlPntsWNew
            else:
                ctrlPntsNew = utils.insertKnotCP(geo.degree, geo.knotVec, geo.ctrlPnts, knot, num, s, span)
                geo.ctrlPnts = ctrlPntsNew

            geo.knotVec = newKnotVec

            newSpan = utils.findSpan(knot, geo.degree, geo.knotVec, geo.nCtl)

            return newSpan

    elif isinstance(geo, BSplineSurface):
        # TODO: Fix this to work with new knot insertion function

        # u-direction
        if param[0] is not None and num[0] > 0:
            knot = checkInput(param[0], "param[0]", float, 0)
            ru = checkInput(num[0], "num[0]", int, 0)

            su = utils.multiplicity(knot, geo.uKnotVec)

            if num[0] > geo.uDegree - su:
                raise ValueError(f"Knot {knot} cannot be inserted {ru} times in the u-direction.")

            # Find the knot span
            spanu = utils.findSpan(knot, geo.uDegree, geo.uKnotVec, geo.nCtlu)

            # Compute the new knot vector
            uKnotVecNew = utils.insertKnotKV(geo.uKnotVec, knot, ru, spanu)

            # Get the control points for rational/non-rational surface
            ctrlPnts = geo.ctrlPntsW if geo.rational else geo.ctrlPnts

            # Allocate the new control point array
            newCtrlPnts = np.zeros((geo.nCtlu + ru, geo.nCtlv, ctrlPnts.shape[-1]))

            # Looop over the v-direction
            for v in range(geo.nCtlv):
                ccu = ctrlPnts[:, v]
                ctrlPntsTemp = utils.insertKnotCP(geo.uDegree, geo.uKnotVec, ccu, knot, ru, su, spanu)

                # Set the control point slice
                newCtrlPnts[:, v] = ctrlPntsTemp[:, : geo.nCtlu + ru]

            # Set the knot vector and control points
            geo.uKnotVec = uKnotVecNew
            if geo.rational:
                geo.ctrlPntsW = newCtrlPnts
            else:
                geo.ctrlPnts = newCtrlPnts

        # v-direction
        if param[1] is not None and num[1] > 0:
            knot = checkInput(param[1], "param[1]", float, 0)
            rv = checkInput(num[1], "num[1]", int, 0)

            sv = utils.multiplicity(knot, geo.vKnotVec)

            if rv > geo.vDegree - sv:
                raise ValueError(f"Knot {knot} cannot be inserted {rv} times in the u-direction.")

            # Find the knot span
            spanv = utils.findSpan(knot, geo.vDegree, geo.vKnotVec, geo.nCtlv)

            # Compute the new knot vector
            vKnotVecNew = utils.insertKnotKV(geo.vKnotVec, knot, rv, spanv)

            # Get the control points for rational/non-rational surface
            ctrlPnts = geo.ctrlPntsW if geo.rational else geo.ctrlPnts

            # Allocate the new control point array
            newCtrlPnts = np.zeros((geo.nCtlu, geo.nCtlv + rv, ctrlPnts.shape[-1]))

            # Looop over the v-direction
            for u in range(geo.nCtlu):
                ccu = ctrlPnts[u, :]
                ctrlPntsTemp = utils.insertKnotCP(geo.vDegree, geo.vKnotVec, ccu, knot, rv, sv, spanv)

                # Set the control point slice
                newCtrlPnts[:, v] = ctrlPntsTemp[:, : geo.nCtlv + rv]

            # Set the knot vector and control points
            geo.vKnotVec = vKnotVecNew
            if geo.rational:
                geo.ctrlPntsW = newCtrlPnts
            else:
                geo.ctrlPnts = newCtrlPnts

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
    # TODO: AL, Fix this to work with new knot insertion algorithm
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


def reverseCurve(curve: CURVETYPE) -> None:
    """Reverse the direction of the curve.

    Parameters
    ----------
    curve : Union[BSplineCurve, NURBSCurve]
        The curve to be reversed.
    """
    if curve.rational:
        curve.ctrlPntsW = curve.ctrlPntsW[::-1, :]
        curve.knotVec = 1 - curve.ctrlPnts[::-1]
    else:
        curve.ctrlPnts = curve.ctrlPnts[::-1, :]
        curve.knotVec = 1 - curve.ctrlPnts[::-1]


def splitCurve(curve: CURVETYPE, u: float) -> Tuple[CURVETYPE, CURVETYPE]:
    """
    Split the curve at parametric position u. This uses the
    :func:`insertKnot` function.

    Parameters
    ----------
    curve: Union[BSplineCurve, NURBSCurve]
        The curve being split.
    u : float
        Parametric position to insert knot.

    Returns
    -------
    Tuple[Union[BSplineCurve, NURBSCurve], Union[BSplineCurve, NURBSCurve]]
        curve1 : Union[BSplineCurve, NURBSCurve] or None
            Curve from s=[0, u]
        curve2 : Union[BSplineCurve, NURBSCurve] or None
            Curve from s=[u, 1]

    Notes
    -----
    curve1 and curve2 may be None if the parameter u is outside
    the range of (0, 1)
    """
    u = checkInput(u, "u", float, 0)
    curve = deepcopy(curve)  # Make a copy so we don't alter the original curve

    if u <= 0.0:
        if curve._rational:
            return None, NURBSCurve(curve.degree, curve.knotVec.copy(), curve.ctrlPntsW.copy())
        else:
            return None, BSplineCurve(curve.degree, curve.knotVec.copy(), curve.ctrlPnts.copy())

    if u >= 1.0:
        if curve._rational:
            return NURBSCurve(curve.degree, curve.knotVec.copy(), curve.ctrlPntsW.copy()), None
        else:
            return BSplineCurve(curve.degree, curve.knotVec.copy(), curve.ctrlPnts.copy()), None

    # Copy over some useful variables to more manageable names
    nCtl = curve.nCtl

    # Get the multiplicity of the knot
    kOrig = utils.findSpan(u, curve.degree, curve.knotVec, nCtl) - curve.degree + 1
    s = utils.multiplicity(u, curve.knotVec, nCtl, curve.degree)

    # r = degree - multiplicity
    r = curve.degree - s

    insertKnot(curve, [u], [r], check=False)
    kNew = utils.findSpan(u, curve.degree, curve.knotVec, curve.nCtl) + 1

    # Allocate the new knot vectors
    knotVec1 = np.zeros(len(curve.knotVec[:kNew]) + 1)
    knotVec2 = np.zeros(len(curve.knotVec[kNew:]) + curve.degree + 1)

    # Copy over the knots and make sure they are clamped
    knotVec1[:kNew] = curve.knotVec[:kNew]
    knotVec2[curve.degree + 1 :] = curve.knotVec[kNew:]
    knotVec1[-1] = u
    knotVec2[: curve.degree + 1] = u

    # Allocate the new control points
    ctlPnts = curve.ctrlPntsW if curve._rational else curve.ctrlPnts
    ctlPnts1 = np.zeros((len(ctlPnts[: kOrig + r]), ctlPnts.shape[-1]))
    ctlPnts2 = np.zeros((len(ctlPnts[kOrig + r - 1 :]), ctlPnts.shape[-1]))

    ctlPnts1 = ctlPnts[: kOrig + r, :]
    ctlPnts2 = ctlPnts[kOrig + r - 1 :, :]

    # Create the new curves
    if curve._rational:
        newCurve1 = NURBSCurve(curve.degree, knotVec1, ctlPnts1)
        newCurve2 = NURBSCurve(curve.degree, knotVec2, ctlPnts2)
    else:
        newCurve1 = BSplineCurve(curve.degree, knotVec1, ctlPnts1)
        newCurve2 = BSplineCurve(curve.degree, knotVec2, ctlPnts2)

    return newCurve1, newCurve2


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
    # TODO: AL, Fix the fortran layer to have the correct indexing and dimensions (Almost certainly wrong)
    return libspline.getbasisptsurface(
        u, v, surf.uKnotVec, surf.vKnotVec, surf.uDegree, surf.vDegree, vals, colInd, iStart, lIndex
    )
