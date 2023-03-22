# Standard Python modules
from typing import Optional

# External modules
import numpy as np
from scipy.sparse import csr_matrix, linalg

# Local modules
from . import libspline
from . import parametrizations as param
from .bspline import BSplineCurve, BSplineSurface
from .utils import checkInput


def curveLMSApprox(
    points: np.ndarray,
    degree: int,
    nCtl: int,
    maxIter: int = 1,
    tol: float = 0.01,
    u: Optional[np.ndarray] = None,
    weights: Optional[np.ndarray] = None,
    deriv: Optional[np.ndarray] = None,
    derivPtr: Optional[np.ndarray] = None,
    derivWeights: Optional[np.ndarray] = None,
    paramType: str = "arc",
) -> BSplineCurve:
    """
    Fit a B-spline curve to a set of input points using a least-mean-squares approach.

    Parameters
    ----------
    points : np.ndarray
        The input points. The array should have shape (nPts, nDim), where nPts is the number of points
        and nDim is the dimensionality of each point.
    degree : int
        The degree of the B-spline curve to fit.
    nCtl : int
        The number of control points to use in the fitted curve.  If maxIter > 1, this will be incremented
        to each iteration until the rms error of the fitted curve is below the tolerance or until the maxIter
        limit is reached.
    maxIter : int, optional
        The number of iterations to use in the fitting process. Default is 1.
    tol : float, optional
        The tolerance of the LMS fitting process. Tighter tolerances generally mean more control points
        are necessary to reduce the rms error.
    u : np.ndarray, optional
        The parametric coordinates to use for the input points. If not provided, the coordinates are
        computed automatically.
    weights : np.ndarray, optional
        The interpolation weights to use for the input points. If not provided, all points are assumed
        to have a weight of 1.0.
    deriv : np.ndarray, optional
        The derivative vectors to use for the input points. If not provided, the curve will not be fit
        using derivative information.
    derivPtr : np.ndarray, optional
        The pointer to the derivative vectors in the input `u` array. This should have the same shape
        as `deriv`. If not provided, the curve will not be fit using derivative information.
    derivWeights : np.ndarray, optional
        The interpolation weights to use for the derivative vectors. If not provided, all derivatives are
        assumed to have a weight of 1.0.
    paramType : str, optional
        The type of parameterization to use for the curve. Default is "arc".

    Returns
    -------
    BSplineCurve
        The fitted B-spline curve.

    Raises
    ------
    ValueError
        If an invalid input is provided for any parameter.

    Notes
    -----
    This implementation uses a least-mean-squares approach to fit the B-spline curve to the input points.
    It can handle input points with or without derivative information, and can use interpolation weights
    for the input points and derivative vectors. The function also supports constraining the interpolation
    to certain points by setting their weights to -1.

    Examples
    --------
    >>> import numpy as np
    >>> from pysplines.bspline import lmsFitCurve
    >>> points = np.array([[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]])
    >>> curve = lmsFitCurve(points, 2, 5)
    """

    # Get the number of points and the dimension of the curve
    nPts, nDim = points.shape

    # Check the degree input
    degree = checkInput(degree, "degree", int, 0)
    order = degree + 1  # The order of the Bspline

    # Check the parameteric coordinate vector input
    if u is not None:
        u = checkInput(u, "u", float, 1, (nPts,))
    else:
        u = param.paramCurve(points, paramType=paramType)

    # Check the interpolation weights input
    if weights is not None:
        weights = checkInput(weights, "weights", float, 1, (nPts,))
    else:
        weights = np.ones(nPts)

    # Check the derivative input
    if deriv is not None:
        deriv = checkInput(deriv, "deriv", float, 2)
        derivPtr = checkInput(derivPtr, "derivPtr", int, 1, (len(deriv),))
    else:
        derivPtr = np.array([])

    # Check the derivative interpolation weights input
    if derivWeights is not None and deriv is not None:
        derivWeights = checkInput(derivWeights, "derivWeights", float, 1, (len(derivPtr),))
    else:
        if deriv is not None:
            derivWeights = np.ones(len(deriv))

    # Check the number of control points input
    nCtl = checkInput(nCtl, "nCtl", int, 0)

    # Create masks for constrained and unconstrained interpolations
    # based on the values of the weights
    # We use "s" for unconstrained and "t" for constrained
    sMask = np.where(weights > 0.0)
    tMask = np.where(weights <= 0.0)

    # Split the points into con vs. uncon
    S = points[sMask]
    T = points[tMask]

    weights = weights[np.where(weights > 0.0)]
    ns = len(S)
    nt = len(T)

    # Split the parameteric coordinates into con vs. uncon
    us = u[sMask]
    ut = u[tMask]

    # Figure out the derivative information
    if deriv is not None:
        # Create masks for con vs. uncon derivative weights
        sderivMask = np.where(derivWeights > 0.0)
        tderivMask = np.where(derivWeights <= 0.0)

        # Add the derivatives to the S and T matrices
        S = np.vstack((S, deriv[sderivMask]))
        T = np.vstack((T, deriv[tderivMask]))

        # Split the parametric coordinates using the derivative pointer
        uds = u[derivPtr][sderivMask]
        udt = u[derivPtr][tderivMask]

        # Update the weights vector
        weights = np.append(weights, derivWeights[sderivMask])

        # Set the number of con and uncon parametric derivative points
        nuds = len(uds)
        nudt = len(udt)

    else:
        uds = np.array([], "d")
        udt = np.array([], "d")
        nuds = 0
        nudt = 0

    # Create a sparse CSR matrix to hold the weights
    nw = len(weights)  # length of the weights
    W = csr_matrix((weights, np.arange(nw), np.arange(nw + 1)), (nw, nw))

    # Sanity check to make sure the order is ok
    if ns + nt + nuds + nudt < order:
        order = ns + nt + nuds + nudt

    currIter = 0
    while True:
        # Compute the knot vector
        knotVec = computeKnotVecLMS(u, nPts, nCtl, degree)

        # Create a matrix to hold the control point coefficients
        ctrlPnts = np.zeros((nCtl, nDim), "d")

        # Build the 'N' jacobian
        nVals = np.zeros((ns + nuds) * order)
        nRowPtr = np.zeros(ns + nuds + 1, "intc")
        nColInd = np.zeros((ns + nuds) * order, "intc")
        libspline.buildcurvecoeffmatrix(us, uds, knotVec, degree, nCtl, nVals, nRowPtr, nColInd)
        N = csr_matrix((nVals, nColInd, nRowPtr), (ns + nuds, nCtl)).tocsc()

        length = libspline.polylength(points.T)

        # Update the constrained and unconstrained parametric coordinates
        # based on the interpolation weights
        us = u[sMask]
        ut = u[tMask]

        # Update the constrained and uncontrained parameteric coordinates
        # based on the derivative weights
        if deriv is not None:
            uds = u[derivPtr][sderivMask]
            udt = u[derivPtr][tderivMask]

        libspline.buildcurvecoeffmatrix(us, uds, knotVec, degree, nCtl, nVals, nRowPtr, nColInd)
        NTWN = (N.transpose() * W * N).tocsc()

        # Check if we are doing unconstrained LMS
        if nt + nudt == 0:
            solve = linalg.factorized(NTWN)
            for idim in range(nDim):
                ctrlPnts[:, idim] = solve(N.transpose() * W * S[:, idim])

        # More complicated because we have constraints
        # *** Only works with scipy sparse matrices ***
        else:
            mVals = np.zeros((nt + nudt) * order)
            mRowPtr = np.zeros(nt + nudt + 1, "intc")
            mColInd = np.zeros((nt + nudt) * order, "intc")

            libspline.buildcurvecoeffmatrix(ut, udt, knotVec, degree, nCtl, mVals, mRowPtr, mColInd)
            M = csr_matrix((mVals, mColInd, mRowPtr), (nt + nudt, nCtl))

            # Now we must assemble the constrained jacobian
            # [ N^T*W*T      M^T][P] = [ N^T*W*S]
            # [ M            0  ][R]   [ T      ]

            MT = M.transpose().tocsr()

            jVal, jColInd, jRowPtr = libspline.buildcurveconjac(
                NTWN.data,
                NTWN.indptr,
                NTWN.indices,
                MT.data,
                MT.indptr,
                MT.indices,
                M.data,
                M.indptr,
                M.indices,
                nCtl,
            )

            # Create sparse csr matrix and factorize
            J = csr_matrix((jVal, jColInd, jRowPtr), (nCtl + nt + nudt, nCtl + nt + nudt)).tocsc()
            solve = linalg.factorized(J)
            for idim in range(nDim):
                rhs = np.hstack((N.transpose() * W * S[:, idim], T[:, idim]))
                ctrlPnts[:, idim] = solve(rhs)[0:nCtl]

        # Run the parameteric correction
        libspline.curveparamcorr(knotVec, degree, u, ctrlPnts.T, length, points.T)

        err = 0.0
        for idim in range(nDim):
            err += np.linalg.norm(N * ctrlPnts[:, idim] - S[:, idim]) ** 2
        err = np.sqrt(err / nPts)

        currIter += 1

        print(f"Iteration: {currIter:03d} | RMS error: {err:.4e} | Nctl: {nCtl}")

        if err <= tol or currIter >= maxIter:
            print(
                f"Fitting converged in {currIter} iterations with final RMS Error={err:.4%} and {nCtl} control points."
            )
            break
        else:
            nCtl += 1

    # Create the BSpline curve
    curve = BSplineCurve(degree, knotVec, ctrlPnts)
    curve.X = points

    return curve


def curveInterpGlobal(
    points: np.ndarray,
    degree: int,
    deriv: Optional[np.ndarray] = None,
    derivPtr: Optional[np.ndarray] = None,
    paramType: str = "arc",
) -> BSplineCurve:
    """
    Interpolates a curve through the given points.

    Parameters
    ----------
    points : numpy.ndarray
        An n x m array of n m-dimensional control points.
    degree : int
        The degree of the spline.
    deriv : Optional[numpy.ndarray], optional
        An array of derivative values at the control points, by default None.
    derivPtr : Optional[numpy.ndarray], optional
        An array of indices indicating which derivatives are to be used, by default None.
    paramType : str, optional
        The type of parameterization to use, by default "arc".

    Returns
    -------
    BSplineCurve
        The interpolated curve as a BSplineCurve object.

    Raises
    ------
    TypeError
        If `points`, `deriv`, or `derivPtr` are not of the correct data type.

    Notes
    -----
    This function computes a curve that passes through the given control points
    and optionally through their derivatives. The parameterization of the curve
    is computed, which maps the control points to a set of values in parameteric space.
    The parameteric values are then used to compute the knot vector for the curve.

    The interpolation of the curve is then performed by solving a linear system of
    equations. The coefficient matrix of this system is computed using the
    `curve_jacobian_wrap` subroutine in the Fortran layer. The solution of the linear
    system is then used to construct a BSplineCurve object representing the interpolated curve.
    """
    # Number of control points
    nPts, nDim = points.shape

    # Calculate the order of the curve
    order = degree + 1

    # Set the ihterpolation weights
    weights = np.ones(nPts)

    # Check the derivative input
    if deriv is not None:
        deriv = checkInput(deriv, "derivatives", float, 2)
        derivPtr = checkInput(derivPtr, "derivativesPtr", int, 1, (deriv.shape[0],))
        derivWeights = np.ones(deriv.shape[0])
    else:
        derivPtr = np.array([])

    # Get the parameteric coordinates
    u = param.paramCurve(points, paramType=paramType)

    # Rename the points input to make the next part a bit clearer
    S = points

    # Add the derivative information if it exists
    if deriv is not None:
        S = np.vstack((S, deriv))
        uderiv = u[derivPtr]
        weights = np.append(weights, derivWeights)
        nderivPts = len(uderiv)
    else:
        uderiv = np.array([], "d")
        nderivPts = 0

    # Set the number of control points for the interpolation
    nCtl = nPts + nderivPts

    # Compute the knot vector
    knotVec = computeKnotVecInterp(u, nPts, degree)

    # Sanity check to make sure the order and degree are ok
    if nPts + nderivPts < order:
        order = nPts + nderivPts
        degree = order - 1

    # Build the coefficient matrix for the interpolation linear system
    ctrlPnts = np.zeros((nCtl, nDim), "d")
    nVals = np.zeros((nPts + nderivPts) * order)
    nRowPtr = np.zeros(nPts + nderivPts + 1, "intc")
    nColInd = np.zeros((nPts + nderivPts) * order, "intc")
    libspline.buildcurvecoeffmatrix(u, uderiv, knotVec, degree, nCtl, nVals, nRowPtr, nColInd)
    N = csr_matrix((nVals, nColInd, nRowPtr), (nPts, nPts)).tocsc()

    # Factorize once for efficiency
    solve = linalg.factorized(N)
    for idim in range(nDim):
        ctrlPnts[:, idim] = solve(S[:, idim])

    # Create the BSplineCurve
    curve = BSplineCurve(degree, knotVec, ctrlPnts)
    curve.X = points

    return curve


def surfaceLMSApprox(
    points: np.ndarray,
    nCtlu: int,
    nCtlv: int,
    uDegree: int,
    vDegree: int,
    u: Optional[np.ndarray] = None,
    v: Optional[np.ndarray] = None,
    nIter: Optional[int] = 1,
) -> BSplineSurface:
    """
    Fits a B-spline surface to a set of 3D points using the Least Mean Square (LMS) method.

    Parameters
    ----------
    points : np.ndarray
        The array of 3D points to fit the surface to, with shape (nu, nv, 3).
    nCtlu : int
        The number of control points in the u direction.
    nCtlv : int
        The number of control points in the v direction.
    uDegree : int
        The degree of the B-spline in the u direction.
    vDegree : int
        The degree of the B-spline in the v direction.
    u : Optional[np.ndarray]
        The array of u parameter values, with shape (nu,). If not provided,
        the parameter values will be computed, by default None.
    v : Optional[np.ndarray]
        The array of v parameter values, with shape (nv,). If not provided,
        the parameter values will be computed, by default None.
    nIter : Optional[int]
        The number of iterations to use in the koscheck parameter correction, by default 1.

    Returns
    -------
    BSplineSurface
        The B-spline surface that best fits the input points.

    Raises
    ------
    ValueError
        - If the dimension of the points is not equal to 3
        - If any of the inputs are the wrong type or rank
    """

    # Get the number of control points in each direction and the
    # dimension of the interpolation data. (nDim should always equal 3)
    nu, nv, nDim = points.shape
    if nDim != 3 and u is None and v is None:
        raise ValueError(
            "Automatic parameterization of ONLY available for spatial data in 3 dimensions. "
            "Please supply u and v key word arguments otherwise."
        )

    # Check degree and number of control points input
    uDegree = checkInput(uDegree, "uDegree", int, 0)
    vDegree = checkInput(vDegree, "vDegree", int, 0)
    nCtlu = checkInput(nCtlu, "nCtlu", int, 0)
    nCtlv = checkInput(nCtlv, "nCtlv", int, 0)

    # Sanity check the number of control points
    # If the user supplies more points than the number of control points
    # requested, we need to decrease the number of control points and
    # tell the user.
    if nCtlu >= nu:
        print(
            f"The number of u-control points requested {nCtlu} is more than the number of u-points supplied {nu}. "
            "The number of u-control points will be reduced to match the number of supplied u-points."
        )
        nCtlu = nu

    if nCtlv >= nv:
        print(
            f"The number of v-control points requested {nCtlv} is more than the number of v-points supplied {nv}. "
            "The number of v-control points will be reduced to match the number of supplied v-points."
        )
        nCtlv = nv

    # Sanity check to make sure the degree is less than the number of points
    uDegree = nu if nu < uDegree else uDegree
    vDegree = nv if nv < vDegree else vDegree
    uDegree = nCtlu if nCtlu < uDegree else uDegree
    vDegree = nCtlv if nCtlv < vDegree else vDegree

    # Check the iteration input
    nIter = checkInput(nIter, "nIter", int, 0)

    # Check the parameteric coordinate inputs if they are given
    if u is not None and v is not None:
        u = checkInput(u, "u", float, 1, (nu,))
        v = checkInput(v, "v", float, 1, (nv,))

        # Normalize u and v
        u = u / u[-1]
        v = v / v[-1]

        # Build the grid of parameteric coordinates
        V, U = np.meshgrid(v, u)

    # If not given, we need to calculate the parameterization
    else:
        u, v, U, V = param.paramSurf(points)

    # Now we need to calculate the knot vectors for the surface
    uKnotVec = computeKnotVecLMS(u, nu, nCtlu, uDegree)
    vKnotVec = computeKnotVecLMS(v, nv, nCtlv, vDegree)

    # Create the control points
    ctrlPnts = np.zeros((nCtlu, nCtlv, nDim))

    # Compute the surface
    vals, rowPtr, colInd = libspline.buildsurfacecoeffmatrix(
        U.T, V.T, uKnotVec, vKnotVec, uDegree, vDegree, nCtlu, nCtlv
    )
    N = csr_matrix((vals, colInd, rowPtr), (nu * nv, nCtlu * nCtlv))
    NT = N.transpose()

    # Factorize and solve the sparse linear system to get the control
    # point vector
    solve = linalg.factorized(NT * N)
    for idim in range(nDim):
        rhs = NT * points[:, :, idim].flatten()
        ctrlPnts[:, :, idim] = solve(rhs).reshape((nCtlu, nCtlv))

    rms = libspline.surfaceparamcorr(uKnotVec, vKnotVec, uDegree, vDegree, U.T, V.T, ctrlPnts.T, points.T)
    print(f"{rms:.4e}")

    # Create the BSpline surface
    surface = BSplineSurface(uDegree, vDegree, ctrlPnts, uKnotVec, vKnotVec)
    surface.X = points

    # Return the surface
    return surface


def surfaceInterpGlobal(points: np.ndarray, uDegree: int, vDegree: int):
    """Interpolates a BSpline surface given an array of points and the
    degree of the curves in the u and v directions.

    Parameters
    ----------
    points : np.ndarray
        The array of 3D points to fit the surface to, with shape (nu, nv, 3).
    uDegree : int
        The degree of the B-spline in the u direction.
    vDegree : int
        The degree of the B-spline in the v direction.

    Returns
    -------
    BSplineSurface
        The B-spline surface that best fits the input points.

    Raises
    ------
    ValueError
        - If the dimension of the points is not equal to 3
        - If any of the inputs are the wrong type or rank
    """
    nu, nv, nDim = points.shape

    if nDim != 3:
        raise ValueError(
            "Automatic parameterization of ONLY available for spatial data in 3 dimensions. "
            "Please supply u and v key word arguments otherwise."
        )

    # Check degree and number of control points input
    uDegree = checkInput(uDegree, "uDegree", int, 0)
    vDegree = checkInput(vDegree, "vDegree", int, 0)

    nCtlu, nCtlv = nu, nv

    # Sanity check to make sure the degree is less than the number of points
    uDegree = nCtlu if nCtlu < uDegree else uDegree
    vDegree = nCtlv if nCtlv < vDegree else vDegree

    # Compute the parametrization
    u, v, U, V = param.paramSurf(points)

    # Compute the knot vectors
    uKnotVec = computeKnotVecInterp(u, nu, uDegree)
    vKnotVec = computeKnotVecInterp(v, nv, vDegree)

    # Initialize the control points
    ctrlPnts = np.zeros((nCtlu, nCtlv, nDim))

    # Compute the surface
    vals, rowPtr, colInd = libspline.buildsurfacecoeffmatrix(
        U.T, V.T, uKnotVec, vKnotVec, uDegree, vDegree, nCtlu, nCtlv
    )
    N = csr_matrix((vals, colInd, rowPtr), (nu * nv, nCtlu * nCtlv)).tocsc()

    # Factorize and solve the sparse linear system to get the control
    # point vector
    solve = linalg.factorized(N)
    for idim in range(nDim):
        ctrlPnts[:, :, idim] = solve(points[:, :, idim].flatten()).reshape((nCtlu, nCtlv))

    # Create the BSpline surface
    surface = BSplineSurface(uDegree, vDegree, ctrlPnts, uKnotVec, vKnotVec)
    surface.X = points

    # Return the surface
    return surface


def computeKnotVecInterp(u: np.ndarray, nPts: int, degree: int) -> np.ndarray:
    """Generate a knot vector suitible for global curve interpolation using averaging.

    This is an implementation of Eq. 9.8 from The NURBS Book by Piegl and Tiller, pg. 365
    Parameters
    ----------
    u : np.ndarray
        The parameteric coordinate vector.
    nPts : int
        The number of points.
    degree : int
        The degree of the curve.

    Returns
    -------
    np.ndarray
        The knot vector.
    """
    knotVec = np.zeros(nPts + degree + 1)

    for j in range(1, nPts - degree + 1):
        knotVec[j + degree] = np.sum(u[j : j + degree]) / degree

    knotVec[-(degree + 1) :] = 1.0

    return knotVec


def computeKnotVecLMS(u: np.ndarray, nPts: int, nCtl: int, degree: int) -> np.ndarray:
    n = nCtl - 1
    knotVec = np.zeros(nCtl + degree + 1)

    d = (nPts + 1) / (n - degree + 1)

    for j in range(1, n - degree + 1):
        i = int(j * d)
        alpha = (j * d) - i
        knotVec[j + degree] = ((1.0 - alpha) * u[i - 1]) + (alpha * u[i])

    knotVec[-(degree + 1) :] = 1.0

    return knotVec
