subroutine arcLength(u1, u2, degree, knotVec, Pw, nDim, nCtl, rational, nGauss, length)
    use precision
    implicit none

    ! Input
    integer, intent(in) :: degree, nDim, nCtl, nGauss
    real(kind=realType), intent(in) :: Pw(nDim, nCtl), u1, u2
    real(kind=realType), intent(in) :: knotVec(nCtl + degree + 1)
    logical, intent(in) :: rational

    ! Output
    real(kind=realType), intent(out) :: length
    integer :: nDimDeriv

    if (rational) then
        nDimDeriv = nDim - 1
    else
        nDimDeriv = nDim
    end if

    call gaussianQuadrature(curvePoint, nGauss, u1, u2, length)

contains

    subroutine curvePoint(u, val)
        implicit none
        ! Inputs
        real(kind=realType), intent(in) :: u

        ! Outputs
        real(kind=realType), intent(out) :: val

        ! Working
        real(kind=realType) :: deriv(nDimDeriv, 2)

        if (rational) then
            call derivEvalCurveNURBS(u, knotVec, degree, Pw, 1, nCtl, nDim, deriv)
        else
            call derivEvalCurve(u, knotVec, degree, Pw, 1, nCtl, nDim, deriv)
        end if

        val = sqrt(deriv(1, 2)**2 + deriv(2, 2)**2)

    end subroutine curvePoint

end subroutine arcLength
