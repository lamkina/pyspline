subroutine pointInvCurve(xStar, coord, u0, lb, ub, maxIter, tol, printLevel, knotVec, degree, Pw, nCtl, nDim, nPts, uStar)
    use precision
    implicit none

    ! Input
    integer, intent(in) :: degree, nCtl, nDim, nPts, maxIter, printLevel, coord
    real(kind=realType), intent(in) :: xStar(nPts), u0(nPts)
    real(kind=realType), intent(in) :: knotVec(nCtl + degree + 1)
    real(kind=realType), intent(in) :: Pw(nDim, nCtl)
    real(kind=realType), intent(in) :: lb(nPts), ub(nPts)
    real(kind=realType), intent(in) :: tol

    ! Output
    real(kind=realType), intent(out) :: uStar(nPts)

    ! Working
    logical :: rational

    ! Determine if we are using a NURBS or BSpline curve
    if (nDim > 3) then
        rational = .true.
    else
        rational = .false.
    end if

    ! Call the newton solver
    if (rational) then
        call newton(curveResiduals, curveJacobian, u0, uStar, lb, ub, 1.0, 0.5, 0.1, maxIter, 3, tol, nPts, printLevel)
    else
        call newton(curveResidualsNURBS, curveJacobianNURBS, u0, uStar, lb, ub, 1.0, 0.5, 0.1, &
                    maxIter, 3, tol, nPts, printLevel)
    end if

contains

    subroutine curveResiduals(u, r)
        use precision
        implicit none

        ! Input
        real(kind=realType), intent(in) :: u(nPts)

        ! Output
        real(kind=realType) :: r(nPts)

        ! Working
        integer :: i
        real(kind=realType) :: val(nDim, nPts)

        r = 0.0

        call evalCurve(u, knotVec, degree, Pw, nCtl, nDim, nPts, val)

        do i = 1, nPts
            r(i) = val(coord, i) - xStar(i)
        end do

    end subroutine curveResiduals

    subroutine curveResidualsNURBS(u, r)
        use precision
        implicit none

        ! Input
        real(kind=realType), intent(in) :: u(nPts)

        ! Output
        real(kind=realType) :: r(nPts)

        ! Working
        integer :: i
        real(kind=realType) :: val(nDim, nPts), weights(nPts)

        r = 0.0

        call evalCurveNURBS(u, knotVec, degree, Pw, nCtl, nDim, nPts, val, weights)

        do i = 1, nPts
            r(i) = val(coord, i) - xStar(i)
        end do

    end subroutine curveResidualsNURBS

    subroutine curveJacobian(u, jac)
        use precision
        implicit none

        ! Input
        real(kind=realType), intent(in) :: u(nPts)

        ! Output
        real(kind=realType) :: jac(nPts, nPts)

        ! Working
        integer :: i
        real(kind=realType) :: tmp(nDim, 2)

        jac(:, :) = 0.0

        do i = 1, nPts
            call derivEvalCurve(u(i), knotVec, degree, Pw, 1, nCtl, nDim, tmp)
            jac(i, i) = tmp(coord, 2)
        end do
    end subroutine curveJacobian

    subroutine curveJacobianNURBS(u, jac)
        use precision
        implicit none

        ! Input
        real(kind=realType), intent(in) :: u(nPts)

        ! Output
        real(kind=realType) :: jac(nPts, nPts)

        ! Working
        integer :: i
        real(kind=realType) :: tmp(nDim, 2)

        jac(:, :) = 0.0

        do i = 1, nPts
            call derivEvalCurveNURBS(u(i), knotVec, degree, Pw, 1, nCtl, nDim, tmp)
            jac(i, i) = tmp(coord, 2)
        end do
    end subroutine curveJacobianNURBS

end subroutine pointInvCurve

subroutine pointInvSurface(xStar, coords, u0, lb, ub, maxIter, tol, printLevel, uKnotVec, vKnotVec, &
                           uDegree, vDegree, Pw, nCtlu, nCtlv, nDim, nPts, uStar)
    use precision
    implicit none

    ! Input
    integer, intent(in) :: uDegree, vDegree, nCtlu, nCtlv, nDim, nPts, maxIter, printLevel, coords(2)
    real(kind=realType), intent(in) :: xStar(nPts), u0(nPts)
    real(kind=realType), intent(in) :: uKnotVec(nCtlu + uDegree + 1), vKnotVec(nCtlv + vDegree + 1)
    real(kind=realType), intent(in) :: Pw(nDim, nCtlv, nCtlu)
    real(kind=realType), intent(in) :: lb(nPts), ub(nPts)
    real(kind=realType), intent(in) :: tol

    ! Output
    real(kind=realType), intent(out) :: uStar(nPts)

    ! Working
    logical :: rational

    ! Determine if we are using a NURBS or BSpline curve
    if (nDim > 3) then
        rational = .true.
    else
        rational = .false.
    end if

    ! Call the newton solver
    if (rational) then
        call newton(surfaceResidualsNURBS, surfaceJacobianNURBS, u0, uStar, lb, ub, 1.0, 0.5, 0.1, &
                    maxIter, 3, tol, nPts, printLevel)
    else
        call newton(surfaceResiduals, surfaceJacobian, u0, uStar, lb, ub, 1.0, 0.5, 0.1, &
                    maxIter, 3, tol, nPts, printLevel)
    end if
contains
    subroutine surfaceResiduals(u, r)
        use precision
        implicit none

        ! Input
        real(kind=realType), intent(in) :: u(nPts)

        ! Output
        real(kind=realType), intent(out) :: r(nPts)

        ! Working
        integer :: i
        real(kind=realType) :: tmp(nDim)

        do i = 1, nPts, 2

            call evalSurface(u(i), u(i + 1), uKnotVec, vKnotVec, uDegree, vDegree, Pw, &
                             nCtlu, nCtlv, nDim, 1, 1, tmp)

            r(i) = tmp(coords(1)) - xStar(i)
            r(i + 1) = tmp(coords(2)) - xStar(i + 1)
        end do
    end subroutine surfaceResiduals

    subroutine surfaceResidualsNURBS(u, r)
        use precision
        implicit none

        ! Input
        real(kind=realType), intent(in) :: u(nPts)

        ! Output
        real(kind=realType), intent(out) :: r(nPts)

        ! Working
        integer :: i
        real(kind=realType) :: tmp(nDim)

        do i = 1, nPts, 2

            call evalSurfaceNURBS(u(i), u(i + 1), uKnotVec, vKnotVec, uDegree, vDegree, Pw, &
                                  nCtlu, nCtlv, nDim, 1, 1, tmp)

            r(i) = tmp(coords(1)) - xStar(i)
            r(i + 1) = tmp(coords(2)) - xStar(i + 1)
        end do

    end subroutine surfaceResidualsNURBS

    subroutine surfaceJacobian(u, jac)
        use precision
        implicit none

        ! Input
        real(kind=realType), intent(in) :: u(nPts)

        ! Output
        real(kind=realType), intent(out) :: jac(nPts, nPts)

        ! Working
        integer :: i
        real(kind=realType) :: skl(nDim, 2, 2)

        jac(:, :) = 0.0

        do i = 1, nPts, 2
            call derivEvalSurface(u(i), u(i + 1), uKnotVec, vKnotVec, uDegree, vDegree, Pw, &
                                  1, nCtlu, nCtlv, nDim, skl)

            jac(i, i) = skl(coords(1), 1, 2)
            jac(i, i + 1) = skl(coords(1), 2, 1)
            jac(i + 1, i) = skl(coords(2), 1, 2)
            jac(i + 1, i + 1) = skl(coords(2), 2, 1)
        end do
    end subroutine surfaceJacobian

    subroutine surfaceJacobianNURBS(u, jac)
        use precision
        implicit none

        ! Input
        real(kind=realType), intent(in) :: u(nPts)

        ! Output
        real(kind=realType), intent(out) :: jac(nPts, nPts)

        ! Working
        integer :: i
        real(kind=realType) :: skl(nDim - 1, 2, 2)

        jac(:, :) = 0.0

        do i = 1, nPts, 2
            call derivEvalSurfaceNURBS(u(i), u(i + 1), uKnotVec, vKnotVec, uDegree, vDegree, Pw, &
                                       nCtlu, nCtlv, nDim, 1, skl)
            jac(i, i) = skl(coords(1), 1, 2)
            jac(i, i + 1) = skl(coords(1), 2, 1)
            jac(i + 1, i) = skl(coords(2), 1, 2)
            jac(i + 1, i + 1) = skl(coords(2), 2, 1)
        end do
    end subroutine surfaceJacobianNURBS
end subroutine pointInvSurface

subroutine pointInvArc(inputLength, ua, ub, startA, startB, degree, knotVec, Pw, rational, nDim, nCtl, &
                       maxIter, tol, printLevel, status, u)
    use precision
    implicit none

    ! Input
    real(8), intent(in) :: inputLength
    integer, intent(in) :: degree, maxIter, nCtl, nDim, printLevel
    logical, intent(in) :: rational, startA, startB
    real(kind=realType), intent(in) :: ua, ub, tol
    real(kind=realType), intent(in) :: knotVec(nCtl + degree + 1), Pw(nDim, nCtl)

    ! Output
    integer, intent(out) :: status
    real(kind=realType), intent(out) :: u

    call brent(subArcLength, ua, ub, tol, maxIter, printLevel, u, status)

contains
    subroutine subArcLength(t, residual)
        implicit none

        ! Input
        real(kind=realType), intent(in) :: t

        ! Output
        real(kind=realType), intent(out) :: residual

        ! Working
        real(kind=realType) :: dist

        if (startA) then
            call arcLength(ua, t, degree, knotVec, Pw, nDim, nCtl, rational, 32, dist)
        else if (startB) then
            call arcLength(ub, t, degree, knotVec, Pw, nDim, nCtl, rational, 32, dist)
        end if

        residual = dist - inputLength
    end subroutine subArcLength
end subroutine pointInvArc
