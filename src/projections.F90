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

subroutine pointCurve(points, u0, knotVec, degree, Pw, nCtl, nDimPw, nDim, rational, lb, ub, tol, &
                      maxIter, maxIterLS, alpha0, rho, wolfe, printLevel, nPts, u, diff)
    ! Description:
    !     pointCurve attempts to solve the point inversion problem using
    !     gradient-based optimization.
    !
    !     Description of Arguments
    !     Inputs:
    !       points    - Real, array size(nPts), Points we are trying to invert
    !       u0        - Real, vector, size(nPts), Initial guess parameters
    !       knotVec   - Real, array size(nCtl+degree+1), Knot vector
    !       degree    - Integer, Degree of B-spline
    !       Pw        - Real, Array of B-spline coefficients and weights. Size (nDim, nCtl)
    !       nCtl      - Integer, Number of control points
    !       nDimPw    - Integer, Number of dimensions in Pw
    !       nDim      - Integer, Spatial dimension of curve
    !       rational  - Logical, Is the curve rational?
    !       lb        - Real, array size(nPts), Lower bound on u
    !       ub        - Real, array size(nPts), Upper bound on u
    !       tol       - Real, Optimization convergence tolerance
    !       maxIter   - Integer, Maximum number of iterations
    !       maxIterLS - Integer, Maximum number of line search iterations
    !       alpha0    - Real, Initial line search step length
    !       rho       - Real, Line search backtracking factor
    !       printLevel- Integer, Print level
    !       nPts      - Integer, Number of points to invert
    !
    !     Ouputs:
    !       u         - Real, array size(nPts), Parameter where C(u) is closest to points
    !       diff      - Real, Array, size(nDim), Distance between points and C(u)
    use precision
    implicit none

    ! Input
    integer, intent(in) :: degree, nCtl, nDimPw, nDim, nPts, maxIter, maxIterLS
    real(kind=realType), intent(in) :: points(nPts, nDim)
    real(kind=realType), intent(in) :: knotVec(nCtl + degree + 1)
    real(kind=realType), intent(in) :: Pw(nDimPw, nCtl)
    real(kind=realType), intent(in) :: lb(nPts), ub(nPts), u0(nPts)
    real(kind=realType), intent(in) :: tol, alpha0, rho, wolfe
    integer, intent(in) :: printLevel
    logical, intent(in) :: rational

    ! Output
    real(kind=realType), intent(out) :: u(nPts)
    real(kind=realType), intent(out) :: diff(nPts, nDim)

    ! Working
    integer :: ii
    real(kind=realType) :: point(nDim), tempVal(nDim), tempWeight

    ! Loop over all the points and minimize the distance between the
    ! point and the curve
    parameterLoop: do ii = 1, nPts
        point = points(ii, :)
        if (printLevel > 0) write (*, '(a, i3, a, i3, a, e12.4)') 'Point ', ii, ' of ', nPts, ' : ', point
        call optimizer(objective, derivative, u0(ii), u(ii), lb(ii), ub(ii), &
                       alpha0, rho, wolfe, maxIter, maxIterLS, tol, printLevel, 1)
        if (printLevel > 0) write (*, '(a,e12.4,/)') 'u = ', u(ii)

        ! Evaluate the curve at the final point
        if (rational) then
            call evalCurveNURBS(u(ii), knotVec, degree, Pw, nCtl, nDimPw, 1, tempVal, tempWeight)
        else
            call evalCurve(u(ii), knotVec, degree, Pw, nCtl, nDimPw, 1, tempVal)
        end if

        diff(ii, :) = tempVal - point
    end do parameterLoop

contains

    subroutine objective(param, obj)
        implicit none

        ! Input
        real(kind=realType), intent(in) :: param

        ! Output
        real(kind=realType), intent(out) :: obj

        ! Working
        real(kind=realType) :: val(nDim), weight

        if (rational) then
            call evalCurveNURBS(param, knotVec, degree, Pw, nCtl, nDimPw, 1, val, weight)
        else
            call evalCurve(param, knotVec, degree, Pw, nCtl, nDimPw, 1, val)
        end if

        obj = 0.5 * NORM2(val - point)**2

    end subroutine objective

    subroutine derivative(param, gradient, hessian)
        implicit none

        ! Input
        real(kind=realType), intent(in) :: param

        ! Output
        real(kind=realType), intent(out) :: gradient, hessian

        ! Working
        real(kind=realType) :: deriv(nDim, 3)

        if (rational) then
            call derivEvalCurveNURBS(param, knotVec, degree, Pw, 2, nCtl, nDimPw, deriv)
        else
            call derivEvalCurve(param, knotVec, degree, Pw, 2, nCtl, nDimPw, deriv)
        end if

        gradient = dot_product(deriv(:, 1) - point, deriv(:, 2))
        hessian = dot_product(deriv(:, 2), deriv(:, 2)) + dot_product(deriv(:, 1) - point, deriv(:, 3))
    end subroutine derivative
end subroutine pointCurve

subroutine curveCurve(u0, knotVec1, degree1, Pw1, knotVec2, degree2, &
                      Pw2, nCtl1, nCtl2, nDimPw1, nDimPw2, rational1, &
                      rational2, lb, ub, tol, maxIter, maxIterLS, &
                      alpha0, rho, wolfe, printLevel, u)
    ! Description:
    !     curveCurve attempts to solve the curve-curve intersection problem using
    !     gradient-based optimization.
    !
    !     Description of Arguments
    !     Inputs:
    !       u0        - Real, vector, size(2), Initial guess parameters
    !       knotVec1  - Real, array size(nCtl1+degree1+1), Knot vector for curve 1
    !       degree1   - Integer, Degree of B-spline for curve 1
    !       Pw1       - Real, Array of B-spline coefficients and weights for curve 1. Size (nDim, nCtl1)
    !       knotVec2  - Real, array size(nCtl2+degree2+1), Knot vector for curve 2
    !       degree2   - Integer, Degree of B-spline for curve 2
    !       Pw2       - Real, Array of B-spline coefficients and weights for curve 2. Size (nDim, nCtl2)
    !       nCtl1     - Integer, Number of control points for curve 1
    !       nCtl2     - Integer, Number of control points for curve 2
    !       nDimPw1   - Integer, Number of dimensions in Pw1
    !       nDimPw2   - Integer, Number of dimensions in Pw2
    !       rational1 - Logical, Is the curve 1 rational?
    !       rational2 - Logical, Is the curve 2 rational?
    !       lb        - Real, array size(2), Lower bound on u
    !       ub        - Real, array size(2), Upper bound on u
    !       tol       - Real, Optimization convergence tolerance
    !       maxIter   - Integer, Maximum number of iterations
    !       maxIterLS - Integer, Maximum number of line search iterations
    !       alpha0    - Real, Initial line search step length
    !       rho       - Real, Line search backtracking factor
    !       printLevel- Integer, Print level
    !
    !     Ouputs:
    !       u         - Real, array size(2), Parameters where the two curves are closest
    !       diff      - Real, Array, size(nDim), Distance between points and C(u)

    use precision
    implicit none

    ! Input
    integer, intent(in) :: degree1, degree2, nCtl1, nCtl2
    integer, intent(in) :: nDimPw1, nDimPw2, maxIter, maxIterLS
    real(kind=realType), intent(in) :: knotVec1(nCtl1 + degree1 + 1)
    real(kind=realType), intent(in) :: knotVec2(nCtl2 + degree2 + 1)
    real(kind=realType), intent(in) :: Pw1(nDimPw1, nCtl1)
    real(kind=realType), intent(in) :: Pw2(nDimPw2, nCtl2)
    real(kind=realType), intent(in) :: lb(2), ub(2), u0(2)
    real(kind=realType), intent(in) :: tol, alpha0, rho, wolfe
    integer, intent(in) :: printLevel
    logical, intent(in) :: rational1, rational2

    ! Output
    real(kind=realType), intent(out) :: u(2)

    ! Working
    integer :: nDim

    if (rational1) then
        nDim = nDimPw1 - 1
    else
        nDim = nDimPw1
    end if

    ! Minimize the distance between the two curves
    call optimizer(objective, derivative, u0, u, lb, ub, alpha0, rho, wolfe, maxIter, maxIterLS, tol, printLevel, 2)

contains

    subroutine objective(param, obj)
        implicit none

        ! Input
        real(kind=realType), intent(in) :: param(2)

        ! Output
        real(kind=realType), intent(out) :: obj

        ! Working
        real(kind=realType) :: val1(nDimPw1), val2(nDimPw2), w1, w2

        if (rational1 .and. rational2) then
            call evalCurveNURBS(param(1), knotVec1, degree1, Pw1, nCtl1, nDimPw1, 1, val1, w1)
            call evalCurveNURBS(param(2), knotVec2, degree2, Pw2, nCtl2, nDimPw2, 1, val2, w2)
        else if (rational1 .and. .not. rational2) then
            call evalCurveNURBS(param(1), knotVec1, degree1, Pw1, nCtl1, nDimPw1, 1, val1, w1)
            call evalCurve(param(2), knotVec2, degree2, Pw2, nCtl2, nDimPw2, 1, val2)
        else if (rational2 .and. .not. rational1) then
            call evalCurve(param(1), knotVec1, degree1, Pw1, nCtl1, nDimPw1, 1, val1)
            call evalCurveNURBS(param(2), knotVec2, degree2, Pw2, nCtl2, nDimPw2, 1, val2, w2)
        else
            call evalCurve(param(1), knotVec1, degree1, Pw1, nCtl1, nDimPw1, 1, val1)
            call evalCurve(param(2), knotVec2, degree2, Pw2, nCtl2, nDimPw2, 1, val2)
        end if

        obj = 0.5 * NORM2(val1(1:nDim) - val2(1:nDim))**2
        ! print *, "R obj: ", val1(1:nDim) - val2(1:nDim)
    end subroutine objective

    subroutine derivative(param, gradient, hessian)
        implicit none

        ! Input
        real(kind=realType), intent(in) :: param(2)

        ! Output
        real(kind=realType), intent(out) :: gradient(2), hessian(2, 2)

        ! Working
        real(kind=realType) :: deriv1(nDimPw1, 3), deriv2(nDimPw2, 3), R(nDim)

        if (rational1 .and. rational2) then
            call derivEvalCurveNURBS(param(1), knotVec1, degree1, Pw1, 2, nCtl1, nDimPw1, deriv1)
            call derivEvalCurveNURBS(param(2), knotVec2, degree2, Pw2, 2, nCtl2, nDimPw2, deriv2)
        else if (rational1 .and. .not. rational2) then
            call derivEvalCurveNURBS(param(1), knotVec1, degree1, Pw1, 2, nCtl1, nDimPw1, deriv1)
            call derivEvalCurve(param(2), knotVec2, degree2, Pw2, 2, nCtl2, nDimPw2, deriv2)
        else if (rational2 .and. .not. rational1) then
            call derivEvalCurve(param(1), knotVec1, degree1, Pw1, 2, nCtl1, nDimPw1, deriv1)
            call derivEvalCurveNURBS(param(2), knotVec2, degree2, Pw2, 2, nCtl2, nDimPw2, deriv2)
        else
            call derivEvalCurve(param(1), knotVec1, degree1, Pw1, 2, nCtl1, nDimPw1, deriv1)
            call derivEvalCurve(param(2), knotVec2, degree2, Pw2, 2, nCtl2, nDimPw2, deriv2)
        end if

        R = deriv1(1:nDim, 1) - deriv2(1:nDim, 1)

        call mtxPrint("deriv1", deriv1, nDimPw1, 3)
        call mtxPrint("deriv2", deriv2, nDimPw2, 3)

        gradient(1) = dot_product(R, deriv1(1:nDim, 2))
        gradient(2) = dot_product(R, -deriv2(1:nDim, 2))

        hessian(1, 1) = dot_product(deriv1(1:nDim, 2), deriv1(1:nDim, 2)) + dot_product(R, deriv1(1:nDim, 3))
        hessian(1, 2) = dot_product(deriv1(1:nDim, 2), -deriv2(1:nDim, 2))
        hessian(2, 1) = hessian(1, 2)
        hessian(2, 2) = dot_product(-deriv2(1:nDim, 2), -deriv2(1:nDim, 2)) + dot_product(R, -deriv2(1:nDim, 3))
    end subroutine derivative

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
