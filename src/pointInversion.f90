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
    call newton(curveResiduals, curveJacobian, u0, uStar, lb, ub, 1.0, 0.5, 0.1, maxIter, 3, tol, nPts, printLevel)

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
        real(kind=realType) :: val(nDim, nPts), weights(nPts)

        r = 0.0

        if (rational) then
            call evalCurveNURBS(u, knotVec, degree, Pw, nCtl, nDim, nPts, val, weights)
        else
            call evalCurve(u, knotVec, degree, Pw, nCtl, nDim, nPts, val)
        end if

        do i = 1, nPts
            r(i) = val(coord, i) - xStar(i)
        end do

    end subroutine curveResiduals

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
            if (rational) then
                call derivEvalCurveNURBS(u(i), knotVec, degree, Pw, 1, nCtl, nDim, tmp)
            else
                call derivEvalCurve(u(i), knotVec, degree, Pw, 1, nCtl, nDim, tmp)
            end if
            jac(i, i) = tmp(coord, 2)
        end do
    end subroutine curveJacobian

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
    call newton(surfaceResiduals, surfaceJacobian, u0, uStar, lb, ub, 1.0, 0.5, 0.1, maxIter, 3, tol, nPts, printLevel)

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

            if (rational) then
                call evalSurfaceNURBS(u(i), u(i + 1), uKnotVec, vKnotVec, uDegree, vDegree, Pw, &
                                      nCtlu, nCtlv, nDim, 1, 1, tmp)
            else
                call evalSurface(u(i), u(i + 1), uKnotVec, vKnotVec, uDegree, vDegree, Pw, &
                                 nCtlu, nCtlv, nDim, 1, 1, tmp)
            end if

            r(i) = tmp(coords(1)) - xStar(i)
            r(i + 1) = tmp(coords(2)) - xStar(i + 1)
        end do
    end subroutine surfaceResiduals

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

        do i = 1, nPts, 2
            if (rational) then
                call derivEvalSurfaceNURBS(u(i), u(i + 1), uKnotVec, vKnotVec, uDegree, vDegree, Pw, &
                                           nCtlu, nCtlv, nDim, 1, skl)
            else
                call derivEvalSurface(u(i), u(i + 1), uKnotVec, vKnotVec, uDegree, vDegree, Pw, &
                                      1, nCtlu, nCtlv, nDim, skl)
            end if
            jac(i, i) = skl(coords(1), 1, 2)
            jac(i, i + 1) = skl(coords(2), 1, 2)
            jac(i + 1, i) = skl(coords(1), 2, 1)
            jac(i + 1, i + 1) = skl(coords(2), 2, 1)
        end do
    end subroutine surfaceJacobian
end subroutine pointInvSurface
