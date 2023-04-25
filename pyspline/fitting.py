# Standard Python modules
from typing import Optional

# External modules
import numpy as np
from scipy.sparse import csr_matrix, linalg

# Local modules
from . import libspline, compatibility, projections
from . import parametrizations as param
from .bspline import BSplineCurve, BSplineSurface
from .nurbs import NURBSCurve
from .utils import checkInput, intersect3DLines

def localConicInterp(Q: np.ndarray, T: np.ndarray, kMax: int=2, tol: float=1e-6):
    ks = 0
    ke = kMax

    Pw = None  # Weighted control points
    degree = 2 # Conics are always quadratic

    # Loop over the segments from ks to ke to find a conic fit
    # Accumulate the segments to build the full curve
    # Knot vector is the accumulated chord lengths correspoinding to segment boundaries
    # Need to use double knots to exactly represent conics
    fit_flag = False
    chords = []
    while True:
        flag, Rw = fitWithConic(ks, ke, Q, T[ks], T[ke], tol)

        if flag:
            # Succesfully found a conic fit for this segment
            Qs = compatibility.combineCtrlPnts(np.atleast_2d(Q[ks]))
            Qe = compatibility.combineCtrlPnts(np.atleast_2d(Q[ke]))
            
            if Pw is None:
                Pw = np.vstack((Qs, Rw, Qe))
            else:
                Pw = np.vstack((Pw, Rw, Qe))

            _, alpha1, alpha2, _ = intersect3DLines(Q[ks], T[ks], Q[ke], T[ke])

            if ke == len(Q)-1:
                fit_flag = True
                break  # Fitting is complete
            elif (ke + kMax) > len(Q)-1:
                # Check if we will exceed the length of Q
                ks = ke
                ke = len(Q) - 1
            else:
                # Otherwise increment the end point by kMax
                ks = ke
                ke += kMax

            chords.append(np.linalg.norm(Q[ke] - Q[ks]))
        else:
            # Did not find a conic fit
            if ke - ks > 1:
                # Decrease the interval by half and search again
                ke = ke - (ke - ks) // 2
            else:
                # No fit and we are at the smallest possible interval
                break
    
    chords = np.array(chords)
    intKnots = np.cumsum(chords)
    intKnots /= np.max(intKnots)
    knotVec = np.hstacck((np.zeros(3), ))

    # Return the status of the fitting algorithm and the nurbs curve
    if fit_flag and Pw is not None:
        curve = NURBSCurve(degree, knotVec, Pw)
        return fit_flag, curve
    else:
        return fit_flag, None


def fitWithConic(ks: int, ke: int, Q: np.ndarray, Ts: np.ndarray, Te: np.ndarray, tol: float=1e-12):
    if len(Q) == 2:
        # no interior points to interpolate
        # Need to compute extra points between Qs and Qe
        curve = curveLocalQuadInterp(np.vstack((Q[ks], Q[ke])), np.vstack((Ts, Te)), rational=True)
        return 1, curve.ctrlPntsW

    i, alpha1, alpha2, R = intersect3DLines(Q[ks], Ts, Q[ke], Te)

    if i != 0:
        # No intersection
        if np.any(np.linalg.norm(np.cross(Q[1:], Q[:-1], axis=1), axis=1) != 0):
            return 0, np.array([])  # Not all points are collinear

        else:
            Rw = (Q[ks] + Q[ke]) / 2.0
            Rw = compatibility.combineCtrlPnts(np.atleast_2d(Rw))
            return 1, Rw

    if alpha1 <= 0.0 or alpha2 >= 0.0:
        return 0, np.array([]) # Violates Eq. 9.91 from The NURBS Book

    s = 0.0
    V = Q[ke] - Q[ks]

    for i in range(ks+1, ke):
        # Get conic interpolating each interior point
        V1 = Q[i] - R
        j, alpha1, alpha2, _ = intersect3DLines(Q[ks], V, R, V1)

        if j != 0 or alpha1 <= 0.0 or alpha1 >= 1.0 or alpha2 <= 0.0:
            return 0, np.array([])
        
        # Compute the weight using Algorithm 7.2 from The NURBS Book
        a = np.sqrt(alpha1/(1 - alpha1))
        u = a / (1.0 + a)
        num = (1.0 -u) * (1.0 - u) * np.dot(Q[i] - Q[ks], R - Q[i]) + u * u * np.dot(Q[i]- Q[ke], R - Q[i])
        den = 2 * u * (1 - u) * np.dot(R - Q[i], R-Q[i])
        wi = num / den
        s += wi / (1 + wi)
    
    s = s / (ke - ks - 1)
    w = s / (1 - s)

    if w < 0 or w > 1e6:
        return 0  # Weights are out of bounds
        
    # Create a rational Bezier segment
    ctrlPnts = np.vstack((Q[ks], R, Q[ke]))
    ctrlPntsW = compatibility.combineCtrlPnts(ctrlPnts, np.array([1.0, float(w), 1.0]))
    bezierCurve = NURBSCurve(2, np.array([0, 0, 0, 1, 1, 1]), ctrlPntsW)

    for i in range(ks+1, ke):
        # Project Qi onto the Bezier segment
        _, distance = projections.pointCurve(Q[i], bezierCurve, 20, 1e-12)
        if distance[0] > tol:
            return 0, np.array([])

    Rw = compatibility.combineCtrlPnts(np.atleast_2d(R), np.atleast_1d(w))

    return 1, Rw
    


def curveLocalQuadInterp(Q: np.ndarray, T: Optional[np.ndarray] = None, rational: bool = False, corners: bool=True):
    data = Q.copy()
    n, nDim = Q.shape
    degree = 2 # Quadratic interpolation

    if T is None:
        m = n + 3  # length of the q vector
        q = np.zeros((m, nDim))
        q[2:m-2] = Q[1:] - Q[:-1]
        T = np.zeros((n, nDim))
        V = np.zeros((n, nDim))

        # Equation 9.33 from The NURBS Book
        # q[1] --> q0
        # q[0] --> q-1
        # q[n+1] --> qn+1
        # q[n+2] --> qn+2
        q[1] = 2 * q[2] - q[3]
        q[0] = 2 * q[1] - q[2]
        q[n+1] = 2 * q[n] - q[n-1]
        q[n+2] = 2 * q[n+1] - q[n]

        for k in range(n):
            # Equation 9.31 from The NURBS Book
            j = k + 1
            num = np.linalg.norm(np.cross(q[j-1], q[j]))
            denom1 = np.linalg.norm(np.cross(q[j-1], q[j]))
            denom2 = np.linalg.norm(np.cross(q[j+1], q[j+2]))

            if denom1 + denom2 == 0.0:  # Need to handle collinear cases
                alpha = 1 if corners else 1/2
            else:
                alpha =  num / (denom1 + denom2)

            # Equation 9.30 from The NURBS book
            V[k] = (1 - alpha) * q[j] + alpha * q[j+1]
        
        # Equation 9.29 from The NURBS book
        T = V / np.linalg.norm(V, axis=1)[:, np.newaxis]

    R = []  # Interpolated control points
    Qbar = []
    for k in range(1, n):
        # Try to compute the intersection between lines QTk-1 and QTk
        flag, gammakm1, gammak, Rk = intersect3DLines(Q[k-1], T[k-1], Q[k], T[k])

        if flag == 1:
            # T[k-1] and T[k] are parallel ==> No intersection
            chord = Q[k] - Q[k-1]
           
            # Check the collinear case where T[k-1] and T[k] or both parallel to the line Q[k-1] -- Q[k]
            if np.all(np.cross(chord, T[k-1]) == 0) and np.all(np.cross(chord, T[k] == 0)):
                # Tangents and the chord are collinear
                Rk = 0.5 * (Q[k-1] + Q[k])
                R.append(Rk)
                Qbar.append(Q[k])
            else:
                # Not collinear, but tangents are parallel so we need to interpolate
                gammak = gammak1 = 0.5 * np.linalg.norm(chord)

                # Compute extra parabolic segments
                Rkp = Q[k-1] + gammak * T[k-1]
                Rkp1 = Q[k] - gammak1 * T[k]
                Qkp = (gammak * Rkp1 + gammak1 * Rkp) / (gammak + gammak1)

                R.append(Rkp)
                R.append(Rkp1)
                Qbar.append(Qkp)
                Qbar.append(Q[k])

        else:
            # Tangents are not parallel
            if gammak >= 0 or gammakm1 <= 0:
                # Eq. 9.34 is not satisfied
                chord = Q[k] - Q[k-1]
                alpha = 2/3
                cosThetak = np.dot(chord, T[k]) / (np.linalg.norm(chord) * np.linalg.norm(T[k]))
                cosThetakm1 = np.dot(chord, T[k-1]) / (np.linalg.norm(chord) * np.linalg.norm(T[k-1]))

                beta = 1/2 if rational else 1/4

                gammak = beta * np.linalg.norm(chord) / (alpha * cosThetak + (1-alpha) * cosThetakm1)
                gammak1 = beta * np.linalg.norm(chord) / (alpha * cosThetakm1 + (1-alpha) * cosThetak)

                Rkp = Q[k-1] + gammak * T[k-1]
                Rkp1 = Q[k] - gammak1 * T[k]
                Qkp = (gammak * Rkp1 + gammak1 * Rkp) / (gammak + gammak1)

                R.append(Rkp)
                R.append(Rkp1)
                Qbar.append(Qkp)
                Qbar.append(Q[k])

            else:
                # Eq. 9.34 is satisfied
                R.append(Rk)
                Qbar.append(Q[k])

    R = np.array(R)
    R = np.insert(R, 0, [0, 0, 0], axis=0)
    Qbar = np.array(Qbar)
    Q = np.insert(Qbar, 0, Q[0], axis=0)

    # The length of the control points may have changed due to interpolation, so we need to update "n"
    n = len(Q)

    W = np.ones(n+1) # Initialize to all ones
    if rational:
        # Compute the weights if the rational flag is True
        for k in range(1,n):
            if np.linalg.norm(np.cross(R[k]-Q[k-1], Q[k] - R[k])) < 1e-12:
                # Qk-1, Rk, and Qk are collinear
                W[k] = 1
            
            # Compute the edges of the triangle between Qk-1, Rk, and Qk
            side1 = np.linalg.norm(R[k] - Q[k-1])
            side2 = np.linalg.norm(Q[k] - R[k])

            if side1 - side2 < 1e-12:
                # Triangle is isosceles, use Eq. 7.33 (precise circular arc)
                M = 0.5 * (Q[k-1] + Q[k])
                side = R[k] - Q[k-1] # f = P1 - P2
                base = M - Q[k-1]  # e = M - P2)

                # Wk = cos(theta) = |e| / |f|
                W[k] = np.linalg.norm(base) / np.linalg.norm(side)
            else:
                # Triangle is not isosceles
                M = 0.5 * (Q[k-1] + Q[k])
                vec0 = (M - R[k]) / np.linalg.norm(M - R[k])

                # Find the unit vectors along each of the sides of the triangle
                vec1 = (R[k] - Q[k-1]) / np.linalg.norm(R[k] - Q[k-1])
                vec2 = (Q[k]  - Q[k-1]) / np.linalg.norm(Q[k]  - Q[k-1])

                # Add the two vectors to get the bisecting unit vector
                vec3 = vec1 + vec2

                # Now find the intersection of vec3 and line RkM starting from Qk-1
                _, _, _, S1 = intersect3DLines(Q[k-1], vec3, R[k], vec0)

                # Repeat to find S2
                vec1 = (R[k] - Q[k]) / np.linalg.norm(R[k] - Q[k])
                vec2 = (Q[k-1] - Q[k]) / np.linalg.norm(Q[k-1] - Q[k])
                vec3 = vec1 + vec2
                _, _, _, S2 = intersect3DLines(Q[k], vec3, R[k], vec0)

                # Compute the shoulder point
                S = 0.5 * (S1 + S2)

                # Use Eqs. 7.30 and 7.31 from The NURBS book to compute the weight
                # s = np.linalg.norm(S - M) / np.linalg.norm(R[k] - M)
                s = (S[0] - M[0]) / (R[k][0] - M[0])
                W[k] = s / (1 - s)

    ubar = np.zeros(n)
    ubar[1] = 1
    for k in range(2, n):
        term1 = (ubar[k-1] - ubar[k-2])
        term2 = np.linalg.norm(R[k] - Q[k-1]) / np.linalg.norm(Q[k-1] - R[k-1])
        ubar[k] = ubar[k-1] + term1 * term2
    

    # Create the new control points: P={Q0, R1, R2, ... , Rn, Qn}
    # len(P) = n - 1 + 1 + 1 = n + 1
    P = np.zeros((n+1, nDim))
    P[0] = Q[0]
    P[-1] = Q[-1]
    P[1:-1] = R[1:]

    # Create the new knot vector
    knotVec = np.zeros(len(P) + degree + 1)
    knotVec[-3:] = 1
    knotVec[3:-3] = ubar[1:-1] / ubar[-1]

    if rational:
        Pw = compatibility.combineCtrlPnts(P, W)
        curve = NURBSCurve(degree, knotVec, Pw)
    else:
        curve = BSplineCurve(degree, knotVec, P)
    
    curve.X = data
    return curve


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
