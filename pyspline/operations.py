# Standard Python modules
from copy import deepcopy
from typing import List, Optional, Tuple, Union

# External modules
import numpy as np
from scipy.special import binom

# Local modules
from . import compatibility, libspline
from .bspline import BSplineCurve, BSplineSurface, BSplineVolume
from .custom_types import GEOTYPE
from .nurbs import NURBSCurve, NURBSSurface
from .utils import checkInput


def decomposeCurve(curve: Union[BSplineCurve, NURBSCurve]) -> List[Union[BSplineCurve, NURBSCurve]]:
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
    curveList: List[Union[BSplineCurve, NURBSCurve]], tol: float = 1e-8, check: bool = False
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


def removeKnotCtrlPnts(
    degree: int,
    knotVec: np.ndarray,
    ctrlPnts: np.ndarray,
    knot: float,
    num: int = 1,
    tol: float = 1e-4,
    s: Optional[int] = None,
    span: Optional[int] = None,
) -> np.ndarray:
    nCtl = len(ctrlPnts)
    s = libspline.multiplicity(knot, knotVec, nCtl, degree) if s is None else s
    span = libspline.findspan(knot, degree, knotVec, nCtl) - 1 if span is None else span

    # Check for edge case where we aren't removing any knots
    if num < 1:
        return ctrlPnts

    # Initialize variables
    first = span - degree
    last = span - s

    # Dont change the input control point array, just copy it over
    newCtrlPnts = deepcopy(ctrlPnts)

    # We need to check the control point data structure for the geometry type
    if len(ctrlPnts.shape) > 2:
        temp = np.zeros(((2 * degree) + 1, nCtl, ctrlPnts.shape[-1]))
    else:
        temp = np.zeros(((2 * degree) + 1, ctrlPnts.shape[-1]))

    # Loop to compute Eqs. 5.28 and 5.29 from The NURBS Book
    for t in range(0, num):
        temp[0] = ctrlPnts[first - 1]
        temp[last - first + 2] = ctrlPnts[last + 1]
        i = first
        j = last
        ii = 1
        jj = last - first + 1
        remFlag = False

        # Compute control points for one removal step
        while j - i >= 1:
            alphai = (knot - knotVec[i]) / (knotVec[i + degree + 1 + t] - knotVec[i])
            alphaj = (knot - knotVec[j]) / (knotVec[j + degree + 1 + t] - knotVec[j])

            if len(ctrlPnts.shape) > 2:
                temp[ii, :nCtl] = (ctrlPnts[i, :nCtl] - (1.0 - alphai) * temp[ii - 1, :nCtl]) / alphai
                temp[jj, :nCtl] = (ctrlPnts[j, :nCtl] - alphaj * temp[jj + 1, :nCtl]) / (1.0 - alphaj)
            else:
                temp[ii] = (ctrlPnts[i] - (1.0 - alphai) * temp[ii - 1]) / alphai
                temp[jj] = (ctrlPnts[j] - alphaj * temp[jj + 1]) / (1.0 - alphaj)

            i += 1
            j -= 1
            ii += 1
            jj -= 1

        # Now we need to check if the knot can be removed
        if j - i < t:
            if len(ctrlPnts.shape) > 2:
                if np.linalg.norm(temp[jj + 1, 0] - temp[ii - 1, 0]) <= tol:
                    remFlag = True
            else:
                if np.linalg.norm(temp[jj + 1] - temp[ii - 1]) <= tol:
                    remFlag = True

        else:
            alphai = (knot - knotVec[i]) / (knotVec[i + degree + 1 + t] - knotVec[i])
            if len(ctrlPnts.shape) > 2:
                ptn = (alphai * temp[ii + t + 1, 0]) + ((1.0 - alphai) * temp[ii - 1, 0])
            else:
                ptn = (alphai * temp[ii + t + 1]) + ((1.0 - alphai) * temp[ii - 1])

            if np.linalg.norm(ptn - ctrlPnts[i]) < tol:
                remFlag = True

        # Check if we can remove the knot and update the control point array
        if remFlag:
            i = first
            j = last
            while j - 1 > t:
                newCtrlPnts[i] = temp[i - first + 1]
                newCtrlPnts[j] = temp[j - first + 1]
                i += 1
                j -= 1

        # Update indices
        first -= 1
        last += 1

        # Fix indexing
        t += 1

        # Shift control points (p.183 of The NURBS Book)
        j = int((2 * span - s - degree) / 2)
        i = j
        for k in range(1, t):
            if k % 2 == 1:
                i += 1
            else:
                j -= 1

        for k in range(i + 1, nCtl):
            newCtrlPnts[j] = ctrlPnts[k]
            j += 1

        # Slice to get the new control points
        newCtrlPnts = newCtrlPnts[0:-t]

        return newCtrlPnts


def removeKnotKnotVec(knotVec: np.ndarray, span: int, num: int) -> np.ndarray:
    """Computes the knot vector after knot removal.

    Adapted from: "NURBS-Python: An open-source object-oriented NURBS modeling framework in Python",
    by Bingol and Krishnamurthy

    Orginally from part of Algorithm A5.8 of The NURBS book by Piegl & Tiller

    Parameters
    ----------
    knotVec : np.ndarray
        The knot vector.
    span : int
        The knot span.
    num : int
        The number of knot removals.

    Returns
    -------
    np.ndarray
        The updated knot vector
    """
    if num < 1:
        return knotVec

    newKnotVec = deepcopy(knotVec)

    for iKnot in range(span + 1, len(knotVec)):
        newKnotVec[iKnot - num] = knotVec[iKnot]

    newKnotVec = newKnotVec[0:-num]

    return newKnotVec


def degreeElevation(degree: int, ctrlPnts: np.ndarray, num: int = 1, check: bool = True):
    """Computes the control points of the rational/non-rational spline after degree elevation.

    Adapted python implementation from: "NURBS-Python: An open-source object-oriented NURBS modeling framework in Python",
    by Bingol and Krishnamurthy

    The original source is Eq. 5.36 of the NURBS Book by Piegl &B Tiller, p.205

    Parameters
    ----------
    degree : int
        The degree of the spline geometry
    ctrlPnts : np.ndarray
        The control points of the spline geometry
    num : int, optional
        The number of times to elevate the degree, by default 1
    check : bool, optional
        If True, checks the validity of the degree elevation.  Skips the
        check if False, by default True

    Returns
    -------
    np.ndarray
        The control points of the elevated Bezier geometry

    Raises
    ------
    ValueError
        If the underlying geometry is not a Bezier type
    ValueError
        If the number of degree elevations is infeasible
    """
    if check:
        if degree + 1 != len(ctrlPnts):
            raise ValueError("Can only use degree elevation with Bezier geometries.")
        if num <= 0:
            raise ValueError(f"Cannot elevate the degree {num} times.")

    numElevPnts = degree + 1 + num
    elevPnts = np.zeros((numElevPnts, len(ctrlPnts[0])))

    for i in range(0, numElevPnts):
        start = max(0, (i - num))
        end = min(degree, i)
        for j in range(start, end + 1):
            coeff = binom(degree, j) * binom(num, (i - j))
            coeff /= binom((degree + num), i)
            elevPnts[i] = elevPnts[i] + (coeff * ctrlPnts[j])

    return elevPnts


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
            s = libspline.multiplicity(intKnot, geo.knotVec, geo.nCtl, geo.degree)
            multList.append(s)

        # Decompose the curve into bezier segments
        curveList = decomposeCurve(geo)

        if param[0] > 0:
            for curve in curveList:
                ctrlPnts = curve.ctrlPntsW if curve.rational else curve.ctrlPnts
                newCtrlPnts = degreeElevation(curve.degree, ctrlPnts, num=param[0])
                curve.degree += param[0]
                if curve.rational:
                    curve.ctrlPntsW = newCtrlPnts
                else:
                    curve.ctrlPnts = newCtrlPnts

                knotStart = np.repeat(curve.knotVec[0], param[0])
                knotEnd = np.repeat(curve.knotVec[-1], param[0])
                curve.knotVec = np.concatenate([knotStart, curve.knotVec, knotEnd])

            nd = geo.degree + param[0]

            num = geo.degree - 1

        # Combine the Bezier curve segments back into a full
        knotVec, ctrlPnts, weights, knotList = combineCurves(curveList, check=False)

        # Set the control points depending on rational/non-rational curve
        ctrlPnts = compatibility.combineCtrlPnts(ctrlPnts, weights) if geo.rational else ctrlPnts

        # Apply knot removal
        for knot, s in zip(knotList, multList):
            span = libspline.findspan(knot, nd, knotVec, len(ctrlPnts)) - 1  # Make sure this is zero-based index
            ctrlPnts = removeKnotCtrlPnts(nd, knotVec, ctrlPnts, knot, num=num - s)
            knotVec = removeKnotKnotVec(knotVec, span, num - s)

        # Update the input curve
        geo.degree = nd
        if geo.rational:
            geo.ctrlPntsW = ctrlPnts
        else:
            geo.ctrlPnts = ctrlPnts

        geo.knotVec = knotVec


def degreeReduction(degree: int, ctrlPnts: np.ndarray, check: bool = False) -> np.ndarray:
    if check:
        if degree + 1 != len(ctrlPnts):
            raise ValueError("Degree reduction can only work with Bezier-type geometries")
        if degree < 2:
            raise ValueError("Input spline geometry must have degree > 1")

    # Allocate the reduced control points
    newCtrlPnts = np.zeros((degree, ctrlPnts.shape[-1]))

    # Fix the start and end control points
    newCtrlPnts[0] = ctrlPnts[0]
    newCtrlPnts[-1] = ctrlPnts[-1]

    # Determine the if the degree is odd or ever
    degOdd = True if degree % 2 != 0 else False

    # Compute the control points of the reduced degree shape
    r = int((degree - 1) / 2)

    # Special case when degree equals 2
    if degree == 2:
        r1 = r - 2
    else:
        r1 = r - 1 if degOdd else r

    alpha = np.arange(1, r1 + 1) / degree
    newCtrlPnts[1 : r1 + 1] = (ctrlPnts[1 : r1 + 1] - (alpha * newCtrlPnts[:r1])) / (1 - alpha)

    alpha = np.arange(degree - 2, r1 + 2) / degree
    newCtrlPnts[degree - 2, r1 + 2] = (
        ctrlPnts[degree - 1 : r1 + 3] - ((1 - alpha) * newCtrlPnts[degree - 1 : r1 + 3])
    ) / alpha

    if degOdd:
        # Compute control points to the left
        alpha = r / degree
        left = (ctrlPnts[r] - (alpha * newCtrlPnts[r - 1])) / (1 - alpha)

        # Comptue control points to the right
        alpha = (r + 1) / degree
        right = (ctrlPnts[r + 1] - ((1 - alpha) * newCtrlPnts[r + 1])) / alpha

        # Compute the average of the left and right
        newCtrlPnts[r] = 0.5 * (left + right)

    # Return computed control points after degree reduction
    return newCtrlPnts


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
            s = libspline.multiplicity(intKnot, geo.knotVec, geo.nCtl, geo.degree)
            multList.append(s)

        # Decompose the curve into bezier segments
        curveList = decomposeCurve(geo)

        for curve in curveList:
            ctrlPnts = curve.ctrlPntsW if curve.rational else curve.ctrlPnts
            newCtrlPnts = degreeReduction(curve.degree, ctrlPnts)
            curve.degree -= 1

            if curve.rational:
                curve.ctrlPntsW = newCtrlPnts
            else:
                curve.ctrlPnts = newCtrlPnts

            curve.knotVec = curve.knotVec[1:-1]

        # Compute new degree
        nd = curve.degree - 1

        # Number of knot removals
        num = curve.degree - 1

        # Combine the Bezier curve segments back into a full
        knotVec, ctrlPnts, weights, knotList = combineCurves(curveList, check=False)

        # Set the control points depending on rational/non-rational curve
        ctrlPnts = compatibility.combineCtrlPnts(ctrlPnts, weights) if geo.rational else ctrlPnts

        # Apply knot removal
        for knot, s in zip(knotList, multList):
            span = libspline.findspan(knot, nd, knotVec, len(ctrlPnts)) - 1  # Make sure this is zero-based index
            ctrlPnts = removeKnotCtrlPnts(nd, knotVec, ctrlPnts, knot, num=num - s)
            knotVec = removeKnotKnotVec(knotVec, span, num - s)

        # Update the input curve
        geo.degree = nd
        if geo.rational:
            geo.ctrlPntsW = ctrlPnts
        else:
            geo.ctrlPnts = ctrlPnts

        geo.knotVec = knotVec


def knotRefinement(
    degree: int,
    knotVec: np.ndarray,
    ctrlPnts: np.ndarray,
    density: int = 1,
    tol: float = 1e-8,
    knotRange: Optional[Tuple[int, int]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    nCtl = len(ctrlPnts)
    if density < 1:
        raise ValueError(f"Argument 'density' is ({density}) but must be an integer >= 1")

    if knotRange is not None:
        knotArry = knotVec[knotRange[0], knotRange[1]]
    else:
        knotArry = knotVec[degree:-degree]

    # Remove any duplicated knots and sort the knots in ascending order
    knotArry = np.sort(np.unique(knotArry))

    # Increase the knot density
    for _ in range(0, density):
        midpoints = knotArry[:-1] + np.diff(knotArry) / 2.0
        knotArry = np.insert(knotArry, np.arange(1, len(knotArry)), midpoints, axis=0)

    # Determine the number of knot insertions
    X = []
    for knot in knotArry:
        s = libspline.multiplicity(knot, knotVec, nCtl, degree)
        r = degree - s
        X += [knot for _ in range(r)]

    # Check if we can do knot refinement
    if not X:
        raise ValueError("Cannot refine knot vector in this parametric dimension")

    # Initialize variables
    r = len(X) - 1
    n = nCtl - 1
    m = n + degree + 1
    a = libspline.findspan(X[0], degree, knotVec, nCtl) - 1
    b = libspline.findspan(X[r], degree, knotVec, nCtl)

    # Allocate new control point array
    # Check the dimensions fo the control points to figure out the shape of the new control points
    if len(ctrlPnts.shape) == 2:  # Curve
        newCtrlPnts = np.zeros((n + r + 2, ctrlPnts.shape[-1]))
    elif len(ctrlPnts.shape) == 3:  # Surface
        newCtrlPnts = np.zeros((ctrlPnts.shape[0], n + r + 2, ctrlPnts.shape[-1]))

    # Fill unchanged control points
    newCtrlPnts[: a - degree + 1] = ctrlPnts[: a - degree + 1]
    newCtrlPnts[b + r : n + r + 2] = ctrlPnts[b - 1 : n + 1]

    # Allocate new knot vector array
    newKnotVec = np.zeros(m + r + 2)

    # Fill unchanged knots
    newKnotVec[: a + 1] = knotVec[: a + 1]
    newKnotVec[b + degree + r + 1 : m + r + 2] = knotVec[b + degree : m + 1]

    # Initialize vars for knot refinement
    i = b + degree - 1
    k = b + degree + r
    j = r

    while j >= 0:
        while X[j] <= knotVec[i] and i > a:
            newCtrlPnts[k - degree - 1] = ctrlPnts[i - degree - 1]
            newKnotVec[k] = knotVec[i]
            k -= 1
            i -= 1

        newCtrlPnts[k - degree - 1] = deepcopy(newCtrlPnts[k - degree])

        for l in range(1, degree + 1):
            idx = k - degree + l
            alpha = newKnotVec[k + l] - X[j]

            if abs(alpha) < tol:
                newCtrlPnts[idx - 1] = deepcopy(newCtrlPnts[idx])
            else:
                alpha = alpha / (newKnotVec[k + l] - knotVec[i - degree + l])
                if ctrlPnts.shape == 2:
                    newCtrlPnts[idx - 1] = alpha * newCtrlPnts[idx - 1] + (1.0 - alpha) * newCtrlPnts[idx]
                else:
                    ii = ctrlPnts.shape[0]
                    newCtrlPnts[idx - 1, :ii] = (
                        alpha * newCtrlPnts[idx - 1, :ii] + (1.0 - alpha) * newCtrlPnts[idx, :ii]
                    )

        newKnotVec[k] = X[j]
        k = k - 1
        j -= 1

    return newCtrlPnts, newKnotVec


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
            newCtrlPnts, newKnotVec = knotRefinement(geo.degree, geo.knotVec, ctrlPnts, density)

            if geo.rational:
                geo.ctrlPntsW = newCtrlPnts
            else:
                geo.ctrlPnts = newCtrlPnts

            geo.knotVec = newKnotVec

    if isinstance(geo, BSplineSurface):
        # TODO: AL, add surface knot refinement
        pass


def curveKnotInsertion(
    degree: int, knotVec: np.ndarray, ctrlPts: np.ndarray, knot: float, num: int, s: int
) -> Tuple[int, np.ndarray, np.ndarray]:
    # Initialize useful vars
    nDim = ctrlPts.shape[-1]
    nctl = len(ctrlPts)
    nq = nctl + num

    # Allocate new control points
    ctrlPtsNew = np.zeros((nq, nDim))

    # Allocate a temporary array
    temp = np.zeros((degree + 1, nDim))

    # Next we need to find the knot span
    span = libspline.findspan(knot, degree, knotVec, nctl) - 1

    # Load the new knot vector
    knotVecNew = np.zeros(len(knotVec) + num)
    knotVecNew[: span + 1] = knotVec[: span + 1]
    knotVecNew[span + 1 : span + num + 1] = knot
    knotVecNew[span + num + 1 : nctl + num + degree + 1] = knotVec[span + 1 : nctl + degree + 1]

    # Copy over the unaltered control points
    ctrlPtsNew[: span - degree + 1, :] = ctrlPts[: span - degree + 1, :]
    ctrlPtsNew[span - s + num : nctl + num, :] = ctrlPts[span - s : nctl, :]

    temp[0 : degree - s + 1, :] = ctrlPts[0 + span - degree : span - s + 1, :]

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
    newSpan -= 1  # Convert to zero based index
    return newSpan, knotVecNew, ctrlPtsNew


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
            s = libspline.multiplicity(knot, geo.knotVec, geo.nCtl, geo.degree)

            # Check if we can add the requested number of knots
            if check:
                if num > geo.degree - s:
                    raise ValueError(f"Knot: {knot} cannot be inserted {num} times")

            if geo.rational:
                newSpan, knotVecNew, ctrlPntsWNew = curveKnotInsertion(
                    geo.degree, geo.knotVec, geo.ctrlPntsW, knot, num, s
                )
                geo.ctrlPntsW = ctrlPntsWNew
            else:
                newSpan, knotVecNew, ctrlPntsNew = curveKnotInsertion(
                    geo.degree, geo.knotVec, geo.ctrlPnts, knot, num, s
                )
                geo.ctrlPnts = ctrlPntsNew

            geo.knotVec = knotVecNew

            return newSpan

    elif isinstance(geo, BSplineSurface):
        # TODO: Fix this to work with new knot insertion function
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


def reverseCurve(curve: Union[BSplineCurve, NURBSCurve]) -> None:
    """Reverse the direction of the curve.

    Parameters
    ----------
    curve : BSplineCurve
        The curve to be reversed.
    """
    if curve.rational:
        curve.ctrlPntsW = curve.ctrlPntsW[::-1, :]
        curve.knotVec = 1 - curve.ctrlPnts[::-1]
    else:
        curve.ctrlPnts = curve.ctrlPnts[::-1, :]
        curve.knotVec = 1 - curve.ctrlPnts[::-1]


def splitCurve(
    curve: Union[BSplineCurve, NURBSCurve], u: float
) -> Tuple[Union[BSplineCurve, NURBSCurve], Union[BSplineCurve, NURBSCurve]]:
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
    kOrig = libspline.findspan(u, curve.degree, curve.knotVec, nCtl) - curve.degree
    s = libspline.multiplicity(u, curve.knotVec, nCtl, curve.degree)

    # r = degree - multiplicity
    r = curve.degree - s

    insertKnot(curve, [u], [r], check=False)
    kNew = libspline.findspan(u, curve.degree, curve.knotVec, curve.nCtl)

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
