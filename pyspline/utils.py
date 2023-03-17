# Standard Python modules
from copy import deepcopy
from typing import Optional, Tuple

# External modules
from baseclasses.utils import Error
import numpy as np
from scipy import sparse
from scipy.special import binom

# Local modules
from . import libspline


def calculateGrevillePoints(degree: int, nCtl: int, knotVec: np.ndarray) -> np.ndarray:
    gpts = np.zeros(nCtl)

    for i in range(nCtl):
        for n in range(degree):
            gpts[i] += knotVec[i + n + 1]
        gpts[i] = gpts[i] / (degree)

    return gpts


def calculateInterpolatedGrevillePoints(numPts: int, gPts: np.ndarray) -> np.ndarray:
    s = [gPts[0]]
    for i in range(len(gPts) - 1):
        for j in range(numPts):
            s.append((gPts[i + 1] - gPts[i]) * (j + 1) / (numPts + 1) + gPts[i])
        s.append(gPts[i + 1])

    return np.array(s)


def multiplicity(knot: float, knotVec: np.ndarray, nCtl: int, degree: int) -> int:
    """Wrapper around the fortran implementation of `multiplicity`.

    Finds the multiplicty of the knot in the given knot vector.

    Parameters
    ----------
    knot : float
        The knot.
    knotVec : np.ndarray
        The knot vector.
    nCtl : int
        The number of control points.
    degree : int
        The degree of the curve.

    Returns
    -------
    int
        The multiplicity of the knot.
    """
    return libspline.multiplicity(knot, knotVec, nCtl, degree)


def findSpan(knot: float, degree: int, knotVec: np.ndarray, nCtl: int) -> int:
    """Wrapper around the fortran implementation of `findspan`.

    Finds the span of the knot in the knot vector.

    Parameters
    ----------
    knot : float
        The knot.
    degree : int
        The degree of the curve.
    knotVec : np.ndarray
        The knot vector.
    nCtl : int
        The number of control points.

    Returns
    -------
    int
        The knot span in 0-based indexing.
    """
    return libspline.findspan(knot, degree, knotVec, nCtl)


def insertKnotKV(knotVec: np.ndarray, knot: float, num: int, span: int) -> np.ndarray:
    """Inserts a knot 'num' times into the knot vector.

    From Algorithm A5.1 of The NURBS Book by Piegl & Tiller.

    Parameters
    ----------
    knotVec : np.ndarray
        The knot vector.
    knot : float
        The knot to be inserted.
    num : int
        The number of times to insert the knot.
    span : int
        The knot span.

    Returns
    -------
    np.ndarray
        The updated knot vector.
    """
    size = len(knotVec)
    newKnotVec = np.zeros(size + num)

    newKnotVec[: span + 1] = knotVec[: span + 1]
    newKnotVec[span + 1 : span + num + 1] = knot
    newKnotVec[span + 1 + num : size + num] = knotVec[span + 1 : size]

    return newKnotVec


def insertKnotCP(
    degree: int, knotVec: np.ndarray, ctrlPts: np.ndarray, knot: float, num: int, s: int, span: int
) -> np.ndarray:
    """Compute the control points of a b-spline curve after knot insertion.

    Adapted python implementation from: "NURBS-Python: An open-source object-oriented NURBS modeling framework in Python",
    by Bingol and Krishnamurthy

    Parameters
    ----------
    degree : int
        The degree of the curve.
    knotVec : np.ndarray
        The knot vector.
    ctrlPts : np.ndarray
        The control points.
    knot : float
        The knot to be inserted.
    num : int
        The number of times to insert the knot.
    s : int
        The multiplicity of the knot
    span : int
        The knot span.

    Returns
    -------
    np.ndarray
        The updated control points after knot insertion.
    """
    # Initialize useful vars
    nDim = ctrlPts.shape[-1]
    nctl = len(ctrlPts)
    nq = nctl + num

    # Allocate new control points
    ctrlPtsNew = np.zeros((nq, nDim))

    # Allocate a temporary array
    temp = np.zeros((degree + 1, nDim))

    # Copy over the unaltered control points
    ctrlPtsNew[: span - degree + 1] = ctrlPts[: span - degree + 1]
    ctrlPtsNew[span - s + num : nctl + num] = ctrlPts[span - s : nctl]

    temp[: degree - s + 1] = deepcopy(ctrlPts[span - degree : span - s + 1])

    # Insert the knot "num" times
    for j in range(1, num + 1):
        L = span - degree + j
        for i in range(0, degree - j - s + 1):
            alpha = (knot - knotVec[L + i]) / (knotVec[i + span + 1] - knotVec[L + i])
            temp[i] = alpha * temp[i + 1] + (1.0 - alpha) * temp[i]

        ctrlPtsNew[L] = deepcopy(temp[0])
        ctrlPtsNew[span + num - j - s] = deepcopy(temp[degree - j - s])

    # Load the remaining control points
    L = span - degree + num
    ctrlPtsNew[L + 1 : span - s] = deepcopy(temp[1 : span - s - L])

    return ctrlPtsNew


def refineKnotCurve(
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

        for l in range(1, degree + 1):  # noqa
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


def reduceDegreeCurve(degree: int, ctrlPnts: np.ndarray, check: bool = False) -> np.ndarray:
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

    istart, iend = 1, r1 + 1
    if istart != iend:
        alpha = np.arange(istart, iend) / degree
        newCtrlPnts[istart:iend] = (ctrlPnts[istart:iend] - (alpha * newCtrlPnts[istart - 1 : iend - 1])) / (1 - alpha)

    istart, iend = degree - 2, r1 + 2
    if istart + 1 != iend + 1:
        alpha = np.arange(istart + 1, iend + 1) / degree
        newCtrlPnts[istart:iend] = (
            ctrlPnts[istart + 1, iend + 1] - ((1 - alpha) * newCtrlPnts[istart + 1, iend + 1])
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


def removeKnotCtrlPnts(
    degree: int,
    knotVec: np.ndarray,
    ctrlPnts: np.ndarray,
    knot: float,
    num: int = 1,
    tol: float = 1e-6,
    s: Optional[int] = None,
    span: Optional[int] = None,
) -> np.ndarray:
    """Compute the control points after knot removal

    Adapted from: "NURBS-Python: An open-source object-oriented NURBS modeling framework in Python",
    by Bingol and Krishnamurthy

    Parameters
    ----------
    degree : int
        The degree of the curve.
    knotVec : np.ndarray
        The knot vector.
    ctrlPnts : np.ndarray
        The weighted or unweighted control points.
    knot : float
        The knot to be removed.
    num : int, optional
        The number of times to remove the knot, by default 1
    tol : float, optional
        The tolerance that determines if the knto can be removed, by default 1e-4
    s : Optional[int], optional
        The multiplicity of the knot.  If not provided this will be
        re-computed, by default None
    span : Optional[int], optional
        The knot span.  If not provided this will be re-computed, by default None

    Returns
    -------
    np.ndarray
        _description_
    """
    nCtl = len(ctrlPnts)
    s = multiplicity(knot, knotVec, nCtl, degree) if s is None else s
    span = findSpan(knot, degree, knotVec, nCtl) if span is None else span

    # Check for edge case where we aren't removing any knots
    if num < 1:
        return ctrlPnts

    # Initialize variables
    first = span - degree
    last = span - s

    # Dont change the input control point array, just copy it over
    newCtrlPnts = deepcopy(ctrlPnts)

    # Check if the geometry is a curve, surface, or volume by the shape of the cpts
    isVolume = True if len(ctrlPnts.shape) > 2 else False

    # We need to check the control point data structure for the geometry type
    if isVolume:
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
        while j - i >= t:
            alphai = (knot - knotVec[i]) / (knotVec[i + degree + 1 + t] - knotVec[i])
            alphaj = (knot - knotVec[j - t]) / (knotVec[j + degree + 1] - knotVec[j - t])

            if isVolume:
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
            if isVolume:
                if np.linalg.norm(temp[jj + 1, 0] - temp[ii - 1, 0]) <= tol:
                    remFlag = True
            else:
                if np.linalg.norm(temp[jj + 1] - temp[ii - 1]) <= tol:
                    remFlag = True

        else:
            alphai = (knot - knotVec[i]) / (knotVec[i + degree + 1 + t] - knotVec[i])
            if isVolume:
                ptn = (alphai * temp[ii + t + 1, 0]) + ((1.0 - alphai) * temp[ii - 1, 0])
            else:
                ptn = (alphai * temp[ii + t + 1]) + ((1.0 - alphai) * temp[ii - 1])

            if np.linalg.norm(ptn - ctrlPnts[i]) <= tol:
                remFlag = True

        # Check if we can remove the knot and update the control point array
        if remFlag:
            i = first
            j = last
            while j - i > t:
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


def removeKnotKV(knotVec: np.ndarray, span: int, num: int) -> np.ndarray:
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


# def elevateDegreeCurve(degree: int, ctrlPnts: np.ndarray, num: int = 1, check: bool = True):
#     """Computes the control points of the rational/non-rational spline after degree elevation.

#     Adapted python implementation from: "NURBS-Python: An open-source object-oriented NURBS modeling framework in Python",
#     by Bingol and Krishnamurthy

#     The original source is Eq. 5.36 of the NURBS Book by Piegl &B Tiller, p.205

#     Parameters
#     ----------
#     degree : int
#         The degree of the spline geometry
#     ctrlPnts : np.ndarray
#         The control points of the spline geometry
#     num : int, optional
#         The number of times to elevate the degree, by default 1
#     check : bool, optional
#         If True, checks the validity of the degree elevation.  Skips the
#         check if False, by default True

#     Returns
#     -------
#     np.ndarray
#         The control points of the elevated Bezier geometry

#     Raises
#     ------
#     ValueError
#         If the underlying geometry is not a Bezier type
#     ValueError
#         If the number of degree elevations is infeasible
#     """
#     if check:
#         if degree + 1 != len(ctrlPnts):
#             raise ValueError("Can only use degree elevation with Bezier geometries.")
#         if num <= 0:
#             raise ValueError(f"Cannot elevate the degree {num} times.")

#     numElevPnts = degree + 1 + num
#     elevPnts = np.zeros((numElevPnts, len(ctrlPnts[0])))

#     for i in range(0, numElevPnts):
#         start = max(0, (i - num))
#         end = min(degree, i)
#         for j in range(start, end + 1):
#             coeff = binom(degree, j) * binom(num, (i - j))
#             coeff /= binom((degree + num), i)
#             elevPnts[i] = elevPnts[i] + (coeff * ctrlPnts[j])

#     return elevPnts


def assembleMatrix(data, indices, indptr, shape):
    """
    Generic assemble matrix function to create a CSR matrix

    Parameters
    ----------
    data : array
        Data values for matrix
    indices : int array
        CSR type indices
    indptr : int array
        Row pointer
    shape : tuple-like
        Actual shape of matrix

    Returns
    -------
    M : scipy csr sparse matrix
        The assembled matrix
    """
    M = sparse.csr_matrix((data, indices, indptr), shape)

    return M


def checkInput(inputVal, inputName, dataType, dataRank, dataShape=None):
    """This is a generic function to check the data type and sizes of
    inputs in functions where the user must supply proper
    values. Since Python does not do type checking on Inputs, this is
    required

    Parameters
    ----------
    input : int, float, or complex
        The input argument to check
    inputName : str
        The name of the variable, so it can be identified in an Error message
    dataType : str
        Numpy string dtype code
    dataRank : int
        Desired rank. 0 for scalar, 1 for vector 2 for matrix etc
    dataShape : tuple
        The required shape of the data.
        Scalar is denoted by ()
        Vector is denoted by (n, ) where n is the shape
        Matrix is denoted by (n, m) where n and m are rows/columns

    Returns
    -------
    output : various
        The input transformed to the correct type and guaranteed to
        the desired size and shape.
    """

    # Check the data rank:
    rank = np.ndim(inputVal)
    if rank != dataRank:
        raise Error("'%s' must have rank %d. Input was of rank %d." % (inputName, dataRank, rank))

    # Check the data type
    inputVal = np.array(inputVal)
    tmp = inputVal.astype(dataType)

    # Check if the values are the same:
    diff = (tmp - inputVal).flatten()
    if np.dot(diff, diff) > 10 * np.finfo(1.0).eps:
        raise Error("'%s' could not be safely cast to required type without losing information" % inputName)

    # Finally check that the shape is correct:
    if dataShape is not None:
        if tmp.shape != tuple(dataShape):
            raise Error(
                "'%s' was not the correct shape. Input was shape "
                "%s while requested dataShape was '%s'." % (inputName, repr(tmp.shape), repr(dataShape))
            )
    if dataRank == 0:
        return tmp.squeeze()
    else:
        return tmp


def plane_line(ia, vc, p0, v1, v2):
    """
    Check a plane against multiple lines

    Parameters
    ----------
    ia : ndarray[3, n]
        initial point
    vc : ndarray[3, n]
        search vector from initial point
    p0 : ndarray[3]
        vector to triangle origins
    v1 : ndarray[3]
        vector along first triangle direction
    v2 : ndarray[3]
        vector along second triangle direction

    Returns
    -------
    sol : ndarray[6, n]
        Solution vector - parametric positions + physical coordiantes
    nSol : int
        Number of solutions
    """

    return libspline.plane_line(ia, vc, p0, v1, v2)


def tfi2d(e0, e1, e2, e3):
    """
    Perform a simple 2D transfinite interpolation in 3D.

    Parameters
    ----------
    e0 : ndarray[3, Nu]
        coordinates along 0th edge
    e1 : ndarray[3, Nu]
        coordinates along 1st edge
    e2 : ndarray[3, Nv]
        coordinates along 2nd edge
    e3 : ndarray[3, Nv]
        coordinates along 3rd edge

    Returns
    -------
    X : ndarray[3 x Nu x Nv]
        evaluated points
    """
    return libspline.tfi2d(e0, e1, e2, e3)


def line_plane(ia, vc, p0, v1, v2):
    r"""
    Check a line against multiple planes.
    Solve for the scalars :math:`\alpha, \beta, \gamma` such that

    .. math::

        i_a + \alpha \times v_c &= p_0 + \beta \times v_1 + \gamma \times v_2 \\
        i_a - p_0 &= \begin{bmatrix}-v_c & v_1 & v_2\end{bmatrix}\begin{bmatrix}\alpha\\\beta\\\gamma\end{bmatrix}\\
        \alpha &\ge 0: \text{The point lies above the initial point}\\
        \alpha  &< 0: \text{The point lies below the initial point}

    The domain of the triangle is defined by

    .. math::

       \beta + \gamma = 1

    and

    .. math::

       0 < \beta, \gamma < 1

    Parameters
    ----------
    ia : ndarray[3]
        initial point
    vc : ndarray[3]
        search vector from initial point
    p0 : ndarray[3, n]
        vector to triangle origins
    v1 : ndarray[3, n]
        vector along first triangle direction
    v2 : ndarray[3, n]
        vector along second triangle direction

    Returns
    -------
    sol : real ndarray[6, n]
        Solution vector---parametric positions + physical coordinates
    nSol : int
        Number of solutions
    pid : int ndarray[n]
    """

    return libspline.line_plane(ia, vc, p0, v1, v2)


def searchQuads(pts, conn, searchPts):
    """
    This routine searches for the closest point on a set of quads for each searchPt.
    An ADT tree is built and used for the search and subsequently destroyed.

    Parameters
    ----------
    pts : ndarray[3, nPts]
        points defining the quad elements
    conn : ndarray[4, nConn]
        local connectivity of the quad elements
    searchPts : ndarray[3, nSearchPts]
        set of points to search for

    Returns
    -------
    faceID : ndarray[nSearchPts]
        index of the quad elements, one for each search point
    uv : ndarray[2, nSearchPts]
        parametric ``u`` and ``v`` weights of the projected point on the closest quad
    """

    return libspline.adtprojections.searchquads(pts, conn, searchPts)


def elevateDegreeCurve(
    n: int, degree: int, knotVec: np.ndarray, ctrlPnts: np.ndarray, t: int
) -> Tuple[int, np.ndarray, np.ndarray]:
    m = n + degree + 1
    ph = degree + t
    ph2 = int(np.floor(ph / 2))

    # Compute Bézier degree elevation coefficients
    bezalfs = np.zeros((degree + t + 1, degree + 1))
    bezalfs[0, 0] = 1.0
    bezalfs[ph, degree] = 1.0

    for i in range(1, ph2 + 1):
        inv = 1.0 / binom(ph, i)
        mpi = min(degree, i)

        for j in range(max(0, i - t), mpi + 1):
            bezalfs[i, j] = inv * binom(degree, j) * binom(t, i - j)

    for i in range(ph2 + 1, ph):
        mpi = min(degree, i)

        for j in range(max(0, i - t), mpi + 1):
            bezalfs[i, j] = bezalfs[ph - i, degree - j]

    mh = ph
    kind = ph + 1
    r = -1
    a = degree
    b = degree + 1
    cind = 1
    ua = knotVec[0]

    # We need to figure out the internal knot multiplicities to correctly
    # allocate the new knot vector and control points
    internalKnots = np.unique(knotVec[degree + 1 : -(degree + 1)])
    s = mh
    for knot in internalKnots:
        s += multiplicity(knot, knotVec, len(ctrlPnts), degree) + t

    Qw = np.zeros((s + 1, ctrlPnts.shape[-1]))
    Uh = np.zeros(s + ph + 2)
    bpts = np.zeros((degree + 1, ctrlPnts.shape[-1]))
    nextbpts = np.zeros((degree - 1, ctrlPnts.shape[-1]))
    ebpts = np.zeros((degree + t + 1, ctrlPnts.shape[-1]))

    Qw[0] = ctrlPnts[0]

    for i in range(0, ph + 1):
        Uh[i] = ua

    # Initialize first Bézier segment
    for i in range(0, degree + 1):
        bpts[i] = ctrlPnts[i]

    # Big loop through the knot vector
    while b < m:
        i = b
        while b < m and knotVec[b] == knotVec[b + 1]:
            b += 1
        mul = b - i + 1
        mh += mul + t
        ub = knotVec[b]
        oldr = r
        r = degree - mul

        # Insert knot u[b] r times
        lbz = int(np.floor((oldr + 2) / 2)) if oldr > 0 else 1
        rbz = int(np.floor(ph - (r + 1) / 2)) if r > 0 else ph

        if r > 0:
            # Insert knot to get Bézier segment
            numer = ub - ua
            alfs = np.zeros(degree - 1)
            for k in range(degree, mul, -1):
                alfs[k - mul - 1] = numer / (knotVec[a + k] - ua)

            for j in range(1, r + 1):
                save = r - j
                s = mul + j

                for k in range(degree, s - 1, -1):
                    bpts[k] = alfs[k - s] * bpts[k] + (1.0 - alfs[k - s]) * bpts[k - 1]

                nextbpts[save] = bpts[degree]

            # End of knot insertion

        # Degree elevate the Bêzier segments
        for i in range(lbz, ph + 1):
            # Only points in lbz, ... , ph are used below
            ebpts[i] = 0.0
            mpi = min(degree, i)
            for j in range(max(0, i - t), mpi + 1):
                ebpts[i] = ebpts[i] + bezalfs[i, j] * bpts[j]

        # End of degree elevation for Bézier segments

        if oldr > 1:
            # Must remove knot u=U[a] oldr times
            first = kind - 2
            last = kind
            den = ub - ua
            bet = (ub - Uh[kind - 1]) / den
            for tr in range(1, oldr):
                # Knot removal loop
                i = first
                j = last
                kj = j - kind + 1

                # Loop and compute the new control points for one removal step
                while j - i > tr:
                    if i < cind:
                        alf = (ub - Uh[i]) / (ua - Uh[i])
                        Qw[i] = alf * Qw[i] + (1.0 - alf) * Qw[i - i]

                    if j >= lbz:
                        if j - tr <= kind - ph + oldr:
                            gam = (ub - Uh[j - tr]) / den
                            ebpts[k] = gam * ebpts[kj] + (1.0 - gam) * ebpts[kj + 1]
                        else:
                            ebpts[kj] = bet * ebpts[kj] + (1.0 - bet) * ebpts[kj + 1]

                first -= 1
                last += 1
                # End of removing knot u=U[a]

        if a != degree:  # Load the knot ua
            for _i in range(0, ph - oldr):
                Uh[kind] = ua
                kind += 1

        # Load ctrl pts into Qw
        for j in range(lbz, rbz + 1):
            Qw[cind] = ebpts[j]
            cind += 1

        if b < m:
            # set up for next pass through loop
            for j in range(0, r):
                bpts[j] = nextbpts[j]

            for j in range(r, degree + 1):
                bpts[j] = ctrlPnts[b - degree + j]

            a = b
            b += 1
            ua = ub
        else:
            # End knot
            for i in range(0, ph + 1):
                Uh[kind + i] = ub

    # End of big while loop
    nh = mh - ph - 1

    return nh, Uh, Qw
