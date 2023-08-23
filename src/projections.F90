! This file contains all the projection functionality used in
! pySpline. There are 5 combinations that can yield single solutions. They are:
! 1. point-curve   (1 dof)
! 2. point-surface (2 dof)
! 3. point-volume  (3 dof)
! 4. curve-curve   (2 dof)
! 5. curve-surface (3 dof)

! Additionally, each combination requires a globlalization function to
! obtain a good starting point for the Newton search if one is not
! already provided. The five combinations of these brute-force
! starting points are also included.

! Define some parameters used for all projection functions
#define LSFailMax 2
#define wolfe .001
#define nLine 20

subroutine pointCurve(points, knotVec, degree, Pw, nIter, eps, u, nCtl, nDim, diff)

    !***DESCRIPTION
    !
    !     Written by Gaetan Kenway
    !
    !     Abstract: pointCurve attempts to solve the point inversion problem
    !
    !     Description of Arguments
    !     Input
    !     points    - Real, array size(nDim) point we are trying to invert
    !     knotVec   - Real, Knot vector. Length nCtl+k
    !     degree    - Integer, degree of B-spline
    !     Pw        - Real, Array of B-spline coefficients and weights. Size (nDim, nCtl)
    !     nCtl      - Integer, Number of control points
    !     nDim      - Integer, spatial dimension of curve
    !     nIter     - Integer, Maximum number of Netwton iterations
    !     eps       - Real - Eculdian Distance Convergence Measure
    !     u         - Real, vector, length(N), guess parameters where C(u)
    !                 is closest to points
    !
    !     Ouput
    !     u         - Real, parameter where C(u) is closest to points
    !     diff      - Real, array, size(nDim)- Distance between points and C(u)

    use precision
    implicit none

    ! Input
    integer, intent(in) :: degree, nCtl, nDim, nIter
    real(kind=realType), intent(in) :: points(3)
    real(kind=realType), intent(in) :: knotVec(nCtl + degree + 1)
    real(kind=realType), intent(in) :: Pw(nDim, nCtl)
    real(kind=realType), intent(in) :: eps

    ! Output
    real(kind=realType), intent(inout) :: u
    real(kind=realType), intent(out) :: diff(3)

    ! Working
    real(kind=realType) :: val(3), deriv(3, 3), step, c, dist, p_diff, w
    real(kind=realType) :: grad, hessian, update, R(3), nDist, fval, nfval, pgrad, newPt
    integer :: m, ii, NLSFail, order
    logical :: flag, cflag, rational

    order = degree + 1
    if (nDim == 3) then
        rational = .False.
    else if (nDim == 4) then
        rational = .True.
    end if

    NLSFail = 0
    iterationLoop: do ii = 1, nIter

        ! We need to check if the curve is rational or not based on the dimension
        if (rational) then
            call evalCurveNURBS(u, knotVec, degree, Pw, nCtl, nDim, 1, val, w)
            call derivEvalCurveNURBS(u, knotVec, degree, Pw, 2, nCtl, nDim, deriv)
        else
            call evalCurve(u, knotVec, degree, Pw, nCtl, nDim, 1, val)
            call derivEvalCurve(u, knotVec, degree, Pw, 2, nCtl, nDim, deriv)
        end if

        ! Distance is R, "function value" fval is what we minimize
        R = val - points
        nDist = NORM2(R)
        fval = 0.5 * nDist**2

        ! Calculate the Gradient
        grad = dot_product(R, deriv(:, 2))

        ! Calculate the Hessian
        hessian = dot_product(deriv(:, 2), deriv(:, 2)) + dot_product(R, deriv(:, 3))

        ! Bounds checking
        flag = .False.
        if (u < knotVec(1) + eps .and. grad >= 0.0) then
            flag = .True.
            u = knotVec(1)
        end if

        if (u > knotVec(nCtl + order) - eps .and. grad <= 0.0) then
            flag = .True.
            u = knotVec(nCtl + order)
        end if

        if (flag) then
            grad = 0.0
            hessian = 1.0
        end if

        ! Check the norm of the gradient
        if (abs(grad) < eps) then
            exit iterationLoop
        end if

        ! "Invert" the hessian
        update = grad / hessian
        update = -update
        pgrad = update * grad !dot_product(update, grad)

        ! Check that this is the descent direction
        if (pgrad >= 0.0) then
            update = -grad / ABS(grad)
            pgrad = update * grad !dot_product(update, grad)
        end if

        step = 1.0
        nDist = 0.0
        lineLoop: do m = 1, nLine
            newPt = u + step * update
            cflag = .False. ! Check if the constraint is applied

            if (newpt > knotVec(nCtl + order)) then
                cflag = .True.
                newPt = knotVec(nCtl + order)
            end if
            if (newPt < knotVec(1)) then
                cflag = .True.
                newpt = knotVec(1)
            end if

            ! Evaluate the new point
            if (rational) then
                call evalCurveNURBS(u, knotVec, degree, Pw, nCtl, nDim, 1, val, w)
            else
                call evalCurve(u, knotVec, degree, Pw, nCtl, nDim, 1, val)
            end if

            ! Distance is R, "function value" fval is what we minimize
            R = val - points
            nDist = NORM2(R)
            nfVal = 0.5 * nDist**2

            ! Check if the new point satisfies the wolfe condition
            if (nfval < fval + pgrad * wolfe * step) then
                dist = ndist
                exit lineloop
            end if

            ! Calculate the new step length
            if (cflag) then
                ! If the constraints are applied - and the new point
                ! doesn't satisfy the Wolfe conditions, it doesn't make
                ! sense to apply a quadratic approximation
                step = 0.25 * step
            else
                ! c = nfval - fval - pgrad * step is always positive since
                ! nfval - fval > pgrad * wolfe * step > pgrad * step
                c = ((nfval - fval) - pgrad * step)
                step = -step * step * pgrad / (2.0 * c)
                ! This update is always less than the original step length
            end if
        end do lineloop

        if (m == nLine + 1) then
            dist = ndist
            nLSFail = nLSFail + 1

            if (NLSFail > LSFailMax) then ! There is nothing more we can do...
                exit iterationLoop
            end if
        else
            NLSFail = 0
            ! Check if there has been no change in the coordinates
            p_diff = ABS(u - newpt)
            if (p_diff < eps) then
                exit iterationLoop
            end if
        end if

        u = newpt
    end do iterationLoop

    diff = R

end subroutine pointCurve

subroutine curveCurve(knotVec1, degree1, Pw1, knotVec2, degree2, Pw2, nCtl1, nCtl2, nDim, nIter, &
                      tol, u, t, d)

    !*** DESCRIPTION
    !
    !     Written by Gaetan Kenway
    !
    !     The function takes the spline defination of a bivariate spline
    !     surface and determines the u, v positions that minimizes the
    !     distance between the point and the surface.

    !     Description of Arguments:
    !     Input:
    !     knotVec1      - Knot Vector for Curve 1
    !     degree1      - Order for Curve 1
    !     Pw1   - Coefficients for Curve 1
    !     knotVec2      - Knot Vector for Curve 2
    !     degree2      - Order for Curve 2
    !     Pw2   - Coefficients for Curve 2
    !     nCtl1      - Number of coefficients on curve 1
    !     nCtl2      - Number of coefficients on curve 2
    !     nIter   - Integer: Maximum number of iterations
    !     tol     - Real: Tolerance for newton iteration
    !     u       - Real: Initial guess for parameter on Curve 1
    !     t       - Real: Initial guess for parameter on Curve 2

    !     Output:
    !     u       - Real: parameter on Curve 1
    !     t       - Real: parameter on Curve 2
    !     d    - Real: Distance between curves
    !

    use precision
    implicit none

    ! Input/Output
    integer, intent(in) :: nCtl1, nCtl2, degree1, degree2, nDim
    real(kind=realType), intent(in) :: knotVec1(nCtl1 + degree1 + 1), knotVec2(nCtl2 + degree2 + 1)
    real(kind=realType), intent(in) :: Pw1(nDim, nCtl1), Pw2(nDim, nCtl2)
    real(kind=realType), intent(inout) :: u, t
    integer, intent(in) :: nIter
    real(kind=realType), intent(in) :: tol
    real(kind=realType), intent(out) :: d(nDim)

    ! Working
    real(kind=realType) :: val1(nDim), val2(nDim), w1, w2
    real(kind=realType) :: ck1(nDim, 3), ck2(nDim, 3)
    integer :: i, m, ii, NLSFail
    real(kind=realType) :: low(2), high(2), pt(2), R(nDim), ndist
    real(kind=realType) :: fval, nfval, dist
    real(kind=realType) :: hessian(2, 2), grad(2), newpt(2), c
    real(kind=realType) :: pgrad, update(2), step, p_diff
    logical :: flag, cflag, rational

    if (nDim == 3) then
        rational = .False.
    else if (nDim == 4) then
        rational = .True.
    end if

    NLSFail = 0

    low(1) = knotVec1(1)
    low(2) = knotVec2(1)
    high(1) = knotVec1(nCtl1 + degree1 + 1)
    high(2) = knotVec2(nCtl2 + degree2 + 1)

    pt(1) = u
    pt(2) = t

    iteration_loop: do ii = 1, nIter

        if (rational) then
            call derivEvalCurveNURBS(pt(1), knotVec1, degree1, Pw1, 2, nCtl1, nDim, ck1)
            call derivEvalCurveNURBS(pt(2), knotVec2, degree2, Pw2, 2, nCtl2, nDim, ck2)
        else
            call derivEvalCurve(pt(1), knotVec1, degree1, Pw1, 2, nCtl1, nDim, ck1)
            call derivEvalCurve(pt(2), knotVec2, degree2, Pw2, 2, nCtl2, nDim, ck2)
        end if

        ! Distance is R, "function value" fval is what we minimize
        R = ck1(:, 1) - ck2(:, 1)
        nDist = NORM2(R)
        fval = 0.5 * nDist**2

        ! Calculate the Gradient
        grad(1) = dot_product(R, ck1(:, 2))
        grad(2) = dot_product(R, -ck2(:, 2))

        ! Calculate the Hessian
        hessian(1, 1) = dot_product(ck1(:, 2), ck1(:, 2)) + dot_product(R, ck1(:, 3))
        hessian(1, 2) = dot_product(ck1(:, 2), -ck2(:, 2))
        hessian(2, 1) = hessian(1, 2)
        hessian(2, 2) = dot_product(-ck2(:, 2), -ck2(:, 2)) + dot_product(R, -ck2(:, 3))

        ! Bounds Checking
        do i = 1, 2
            flag = .False.
            if (pt(i) < low(i) + tol .and. grad(i) >= 0.0) then
                flag = .True.
                pt(i) = low(i)
            end if

            if (pt(i) > high(i) - tol .and. grad(i) <= 0.0) then
                flag = .True.
                pt(i) = high(i)
            end if

            if (flag) then
                grad(i) = 0.0
                hessian(:, i) = 0.0
                hessian(i, :) = 0.0
                hessian(i, i) = 1.0
            end if
        end do

        ! Zero-cosine check...slightly modifed from The NURBS Book
        if (NORM2(grad) < tol) then
            exit iteration_loop
        end if

        ! Invert the hessian, compute the update and the projected gradient
        call solve2by2(hessian, grad, update)

        update = -update
        pgrad = dot_product(update, grad)

        !Check that this is a descent direction -
        !otherwise use the negative gradient
        if (pgrad >= 0.0) then
            update = -grad / NORM2(grad)
            pgrad = dot_product(update, grad)
        end if

        step = 1.0
        nDist = 0.0

        lineloop: do m = 1, nLine
            newpt = pt + step * update
            cflag = .False. ! Check if the constraints are applied
            ! Check if the new point exceeds the bounds
            do i = 1, 2
                if (newpt(i) > high(i)) then
                    cflag = .True.
                    newpt(i) = high(i)
                else if (newpt(i) < low(i)) then
                    cflag = .True.
                    newpt(i) = low(i)
                end if
            end do

            ! Re-evaluate new position
            if (rational) then
                call evalCurveNURBS(newPt(1), knotVec1, degree1, Pw1, nCtl1, nDim, 1, val1, w1)
                call evalCurveNURBS(newPt(2), knotVec2, degree2, Pw2, nCtl2, nDim, 1, val2, w2)
            else
                call evalCurve(newpt(1), knotVec1, degree1, Pw1, nCtl1, nDim, 1, val1)
                call evalCurve(newpt(2), knotVec2, degree2, Pw2, nCtl2, nDim, 1, val2)
            end if

            ! Check if the new function value is lower,
            ! otherwise adjust the step size
            R = val1 - val2
            ndist = NORM2(R)
            nfval = 0.5 * ndist**2

            ! Check if the new point satisfies the wolfe condition
            if (nfval < fval + pgrad * wolfe * step) then
                dist = ndist
                exit lineloop
            end if

            ! Calculate the new step length
            if (cflag) then
                ! If the constraints are applied - and the new point
                ! doesn't satisfy the Wolfe conditions, it doesn't make
                ! sense to apply a quadratic approximation
                step = 0.25 * step
            else
                ! c = nfval - fval - pgrad * step is always positive since
                ! nfval - fval > pgrad * wolfe * step > pgrad * step
                c = ((nfval - fval) - pgrad * step)
                step = -step * step * pgrad / (2.0 * c)
                ! This update is always less than the original step length
            end if
        end do lineloop

        if (m == nLine + 1) then
            dist = ndist
            nLSFail = nLSFail + 1
            if (NLSFail > LSFailMax) then
                exit Iteration_loop
            end if
        else
            NLSFail = 0
            ! Check if there has been no change in the coordinates
            p_diff = NORM2(pt - newpt)
            if (p_diff < tol) then
                exit Iteration_loop
            end if
        end if

        pt = newpt
    end do iteration_loop

    u = pt(1)
    t = pt(2)
    d = R

end subroutine curveCurve

! ------------------------------------------------------------------------------------
!              Globalization Functions
! ------------------------------------------------------------------------------------

subroutine pointCurveStart(points, u, data, nu, ndim, N, uGuess)

    !***DESCRIPTION
    !
    !     Written by Gaetan Kenway
    !
    !     Abstract: point_curve_start uses discrete data to determine a good
    !     starting point for the curve projection algorithm

    !     Description of Arguments
    !     Input
    !     x0      - Real, array size (ndim, N), Points we want to globalize
    !     uu      - Real, array size(nu) u-parameter values defining data
    !     data    - Real, array size(ndim, nu) - Data to compare against
    !     nu      - Integer, number of uu data
    !     ndim    - Integer, Spatial dimension
    !     N       - Integer, number of points to check
    !
    !     Ouput
    !     u       - Real, array size(N) - cloested u-parameters

    use precision
    implicit none

    ! Input
    integer, intent(in) :: nu, ndim, N
    real(kind=realType), intent(in) :: points(ndim, N), u(nu), data(ndim, nu)

    ! Output
    real(kind=realType), intent(out) :: uGuess(N)

    ! Working
    real(kind=realType) :: D
    integer :: ipt, i

    do ipt = 1, N
        D = 1e20
        do i = 1, nu
            if (NORM2(points(:, ipt) - data(:, i)) < D) then
                uGuess(ipt) = u(i)
                D = NORM2(points(:, ipt) - data(:, i))
            end if
        end do
    end do

end subroutine pointCurveStart

subroutine pointSurfaceStart(x0, uu, vv, data, nu, nv, ndim, N, u, v)

    !***DESCRIPTION
    !
    !     Written by Gaetan Kenway
    !
    !     Abstract: point_surface_start uses discrete data to determine a good
    !     starting point for the surface projection algorithm

    !     Description of Arguments
    !     Input
    !     x0      - Real, array size (ndim, N), Points we want to globalize
    !     uu      - Real, array size(nu) u-parameter values defining data
    !     vv      - Real, array size(nv) v-parameter values defining data
    !     data    - Real, array size(ndim, nw, nv, nu) - Data to compare against
    !     nu      - Integer, number of uu data
    !     nv      - Integer, number of vv data
    !     ndim    - Integer, Spatial dimension
    !     N       - Integer, number of points to check
    !
    !     Ouput
    !     u       - Real, array size(N) - cloested u-parameters
    !     v       - Real, array size(N) - cloested v-parameters

    use precision
    implicit none

    ! Input
    integer, intent(in) :: nu, nv, ndim, N
    real(kind=realType), intent(in) :: x0(ndim, N), uu(nu), vv(nv)
    real(kind=realType), intent(in) :: data(ndim, nv, nu)

    ! Output
    real(kind=realType), intent(out) :: u(N), v(N)

    ! Working
    real(kind=realType) :: D
    integer :: ipt, i, j

    do ipt = 1, N
        D = 1e20
        do i = 1, nu
            do j = 1, nv
                if (NORM2(X0(:, ipt) - data(:, j, i)) < D) then
                    u(ipt) = uu(i)
                    v(ipt) = vv(j)
                    D = NORM2(X0(:, ipt) - data(:, j, i))
                end if
            end do
        end do
    end do

end subroutine pointSurfaceStart

subroutine pointVolumeStart(x0, uu, vv, ww, data, nu, nv, nw, ndim, N, u, v, w)

    !***DESCRIPTION
    !
    !     Written by Gaetan Kenway
    !
    !     Abstract: point_volume_start uses discrete data to determine a good
    !     starting point for the volume projection algorithm

    !     Description of Arguments
    !     Input
    !     x0      - Real, array size (ndim, N), Points we want to globalize
    !     uu      - Real, array size(nu) u-parameter values defining data
    !     vv      - Real, array size(nv) v-parameter values defining data
    !     ww      - Real, array size(nw) w-parameter values defining data
    !     data    - Real, array size(ndim, nw, nv, nu) - Data to compare against
    !     nu      - Integer, number of uu data
    !     nv      - Integer, number of vv data
    !     nw      - Integer, number of ww data
    !     ndim    - Integer, Spatial dimension
    !     N       - Integer, number of points to check
    !
    !     Ouput
    !     u       - Real, array size(N) - cloested u-parameters
    !     v       - Real, array size(N) - cloested v-parameters
    !     w       - Real, array size(N) - cloested w-parameters

    use precision
    implicit none

    ! Input
    integer, intent(in) :: nu, nv, nw, ndim, N
    real(kind=realType), intent(in) :: x0(ndim, N), uu(nu), vv(nv), ww(nw)
    real(kind=realType), intent(in) :: data(ndim, nw, nv, nu)

    ! Output
    real(kind=realType), intent(out) :: u(N), v(N), w(N)

    ! Working
    real(kind=realType) :: D
    integer :: ipt, i, j, k

    do ipt = 1, N
        D = 1e20
        do i = 1, nu
            do j = 1, nv
                do k = 1, nw
                    if (NORM2(X0(:, ipt) - data(:, k, j, i)) < D) then
                        u(ipt) = uu(i)
                        v(ipt) = vv(j)
                        w(ipt) = ww(k)
                        D = NORM2(X0(:, ipt) - data(:, k, j, i))
                    end if
                end do
            end do
        end do
    end do

end subroutine pointVolumeStart

subroutine curveCurveStart(data1, uu1, data2, uu2, nu1, nu2, ndim, s1, s2)

    !***DESCRIPTION
    !
    !     Written by Gaetan Kenway
    !
    !     Abstract: point_surface_start uses discrete data to determine a good
    !     starting point for the surface projection algorithm

    !     Description of Arguments
    !     Input
    !     data1   - Real, array size (ndim, nu1), Points from first curve
    !     uu1     - Real, array size (nu1), parameter values from first curve
    !     data2   - Real, array size (ndim, nu2), Points from second curve
    !     uu2     - Real, array size (nu2), parameter values from second curve
    !     nu1     - Integer, number of points in data1
    !     nu2     - Integer, number of points in data2
    !     ndim    - Integer, Spatial dimension
    !
    !     Ouput
    !     s1      - Real, parameter value on curve1
    !     s2      - Real, parameter value on curve2

    use precision
    implicit none

    ! Input
    integer, intent(in) :: nu1, nu2, ndim
    real(kind=realType), intent(in) :: data1(ndim, nu1), uu1(nu1), data2(ndim, nu2), uu2(nu2)

    ! Output
    real(kind=realType), intent(out) :: s1, s2

    ! Working
    real(kind=realType) :: D
    integer :: i, j

    D = 1e20
    do i = 1, nu1
        do j = 1, nu2
            if (NORM2(data1(:, i) - data2(:, j)) < D) then
                s1 = uu1(i)
                s2 = uu2(j)
                D = NORM2(data1(:, i) - data2(:, j))
            end if
        end do
    end do

end subroutine curveCurveStart

subroutine curveSurfaceStart(data1, uu1, data2, uu2, vv2, nu1, nu2, nv2, ndim, s, u, v)

    !***DESCRIPTION
    !
    !     Written by Gaetan Kenway
    !
    !     Abstract: point_surface_start uses discrete data to determine a good
    !     starting point for the surface projection algorithm

    !     Description of Arguments
    !     Input
    !     data1   - Real, array size (ndim, nu1), Points from curve
    !     uu1     - Real, array size (nu1), parameter values from curve
    !     data2   - Real, array size (ndim, nv2, nu2), Points from surface
    !     uu2     - Real, array size (nu2), u parameter values from surface
    !     vv2     - Real, array size (nv2), v parameter values from surface
    !     nu1     - Integer, number of points in data1
    !     nu2     - Integer, number of u-points in data2
    !     nv2     - Integer, number of v-points in data2
    !     ndim    - Integer, Spatial dimension
    !
    !     Ouput
    !     s       - Real, parameter value on curve
    !     u       - Real, u-parameter value on surface
    !     v       - Real, v-parameter value on surface

    use precision
    implicit none

    ! Input
    integer, intent(in) :: nu1, nu2, nv2, ndim
    real(kind=realType), intent(in) :: data1(ndim, nu1), uu1(nu1)
    real(kind=realtype), intent(in) :: data2(ndim, nv2, nu2), uu2(nu2), vv2(nv2)

    ! Output
    real(kind=realType), intent(out) :: s, u, v

    ! Working
    real(kind=realType) :: D
    integer :: i, j, k

    D = 1e20
    do i = 1, nu1
        do j = 1, nu2
            do k = 1, nv2
                if (NORM2(data1(:, i) - data2(:, k, j)) < D) then
                    s = uu1(i)
                    u = uu2(j)
                    v = vv2(k)
                    D = NORM2(data1(:, i) - data2(:, k, j))
                end if
            end do
        end do
    end do
end subroutine curveSurfaceStart

subroutine linePlane(ia, vc, p0, v1, v2, n, sol, pid, n_sol)

    ! Check a line against multiple planes
    !
    ! ia:   The initial point
    ! vc:   The search vector from the initial point
    ! p0:   Vectors to the triangle origins
    ! v1:   Vector along the first triangle direction
    ! v2:   Vector along the second triangle direction
    !  n:   Number of triangles to search
    ! sol:  Solution vector - parametric positions + physical coordiantes
    ! nsol: Number of solutions
    !
    ! Solve for the scalars: alpha, beta, gamma such that:
    !    ia + alpha*vc = p0 + beta*v1 + gamma*v2
    !    ia - p0 = [ - vc ; v1 ; v2 ][ alpha ]
    !                                [ beta  ]
    !                                [ gamma ]
    !
    ! alpha >= 0: The point lies above the initial point
    ! alpha  < 0: The point lies below the initial point
    !
    ! The domain of the triangle is defined by:
    !     beta + gamma = 1
    ! and
    !     0 < beta, gamma < 1

    use precision
    implicit none

    ! Input
    integer, intent(in) :: n
    real(kind=realType), intent(in) :: ia(3), vc(3), p0(3, n), v1(3, n), v2(3, n)

    ! Output
    integer, intent(out) :: n_sol
    real(kind=realType), intent(out) :: sol(6, n)
    integer, intent(out) :: pid(n)

    ! Worling
    integer :: i
    real(kind=realType) :: A(3, 3), rhs(3), x(3)

    A(:, 1) = -vc
    n_sol = 0
    sol(:, :) = 0.0
    do i = 1, n
        A(:, 2) = v1(:, i)
        A(:, 3) = v2(:, i)
        rhs = ia - p0(:, i)

        call solve3by3(A, rhs, x)

        ! if (x(1) .ge. 0.00 .and. x(1) .le. 1.00 .and. &
        !     x(2) .ge. 0.00 .and. x(2) .le. 1.00 .and. &
        !     x(3) .ge. 0.00 .and. x(3) .le. 1.00 .and. &
        !     x(2)+x(3) .le. 1.00) then

        if (x(2) .ge. 0.0 .and. x(2) .le. 1.0 .and. &
            x(3) .ge. 0.0 .and. x(3) .le. 1.0 .and. &
            x(2) + x(3) .le. 1.0) then

            n_sol = n_sol + 1
            sol(1:3, n_sol) = x  ! t, u, v parametric locations
            sol(4:6, n_sol) = ia + x(1) * vc ! Actual point value
            pid(n_sol) = i
        end if
    end do
end subroutine linePlane

subroutine planeLine(ia, vc, p0, v1, v2, n, sol, n_sol)

    ! Check a plane against multiple lines

    use precision
    implicit none

    ! Input
    integer, intent(in) :: n
    real(kind=realType), intent(in) :: ia(3, n), vc(3, n), p0(3), v1(3), v2(3)

    ! Output
    integer, intent(out) :: n_sol
    real(kind=realType), intent(out) :: sol(6, n)

    ! Worling
    integer :: i, ind(n)
    real(kind=realType) :: A(3, 3), rhs(3), x(3)

    n_sol = 0
    sol(:, :) = 0.0
    A(:, 2) = v1(:)
    A(:, 3) = v2(:)

    do i = 1, n

        A(:, 1) = -vc(:, i)
        rhs = ia(:, i) - p0(:)

        call solve3by3(A, rhs, x)

        if (x(1) .ge. 0.00 .and. x(1) .le. 1.00 .and. &
            x(2) .ge. 0.00 .and. x(2) .le. 1.00 .and. &
            x(3) .ge. 0.00 .and. x(3) .le. 1.00 .and. &
            x(2) + x(3) .le. 1.00) then

            n_sol = n_sol + 1
            sol(1:3, n_sol) = x  ! t, u, v parametric locations
            sol(4:6, n_sol) = ia(:, i) + x(1) * vc(:, i) ! Actual point value
            ind(n_sol) = i

        end if
    end do
end subroutine planeLine

subroutine pointPlane(pt, p0, v1, v2, n, sol, n_sol, best_sol)

    use precision
    implicit none

    ! Input
    integer, intent(in) :: n
    real(kind=realType), intent(in) :: pt(3), p0(3, n), v1(3, n), v2(3, n)

    ! Output
    integer, intent(out) :: n_sol, best_sol
    real(kind=realType), intent(out) :: sol(6, n)

    ! Working
    integer :: i, ind(n)
    real(kind=realType) :: A(2, 2), rhs(2), x(2), r(3), D, D0

    n_sol = 0
    sol(:, :) = 0.0
    do i = 1, n
        A(1, 1) = v1(1, i)**2 + v1(2, i)**2 + v1(3, i)**2
        A(1, 2) = v1(1, i) * v2(1, i) + v1(2, i) * v2(2, i) + v1(3, i) + v2(3, i)
        A(2, 1) = A(1, 2)
        A(2, 2) = v2(1, i)**2 + v2(2, i)**2 + v2(3, i)**2
        r = p0(:, i) - pt
        rhs(1) = r(1) * v1(1, i) + r(2) * v1(2, i) + r(3) * v1(3, i)
        rhs(2) = r(1) * v2(1, i) + r(2) * v2(2, i) + r(3) * v2(3, i)

        call solve2by2(A, rhs, x)

        if (x(1) .ge. 0.00 .and. x(1) .le. 1.00 .and. &
            x(2) .ge. 0.00 .and. x(2) .le. 1.00 .and. &
            x(1) + x(2) .le. 1.00) then

            n_sol = n_sol + 1
            sol(2:3, n_sol) = x  ! t, u, v parametric locations
            sol(4:6, n_sol) = 0.0 ! Actual point value
            ind(n_sol) = i
        end if
    end do

    ! Now post-process to get the closest one
    best_sol = 1
    D0 = NORM2(p0(:, ind(1)) + sol(2, ind(1)) * v1(:, ind(1)) + sol(3, ind(1)) * v2(:, ind(1)))

    do i = 1, n_sol
        D = NORM2(p0(:, ind(i)) + sol(2, ind(i)) * v1(:, ind(i)) + sol(3, ind(i)) * v2(:, ind(i)))
        if (D < D0) then
            D0 = D
            best_sol = i
        end if
    end do
end subroutine pointPlane
