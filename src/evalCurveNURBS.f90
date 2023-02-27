!> Description:
!>  Evaluates a NURBS curve at the given parameter values using the provided knot vector,
!>  control points, and weights
!>
!> Inputs:
!>  u         - Real, The parameter values to evaluate the curve at
!>  knotvec   - Real, The knot vector of the NURBS curve, length (nctl + degree + 1)
!>  degree    - Integer, The degree of the NURBS curve
!>  Pw        - Real, The array of control points and weights of the NURBS curve, size (ndim + 1, nctl)
!>  nctl      - Integer, The number of control points of the NURBS curve
!>  ndim      - Integer, The dimension of the control points (number of coordinates)
!>  npts      - Integer, The number of parameter values to evaluate
!>
!> Outputs:
!>  val       - Real, The output array of evaluated points on the curve
!>
!> Notes:
!>  This subroutine assumes that the last coordinate of each control point in the Pw array is its weight.
subroutine evalCurveNURBS(u, knotvec, degree, Pw, nctl, ndim, npts, val)
    use precision
    implicit none

    ! Input
    integer, intent(in) :: degree, nctl, ndim, npts
    real(kind=realType), intent(in) :: u(npts)
    real(kind=realType), intent(in) :: knotvec(nctl + degree + 1)
    real(kind=realType), intent(in) :: Pw(ndim, nctl)

    ! Output
    real(kind=realType), intent(out) :: val(ndim, npts)

    ! Working
    integer :: i, j, istart, ileft
    real(kind=realType) :: B(degree + 1)

    val(:, :) = 0.0
    do i = 1, npts
        call findSpan(u(i), degree, knotvec, nctl, ileft)
        call basis(u(i), degree, knotvec, ileft, nctl, B)
        istart = ileft - degree
        do j = 1, degree + 1
            val(:, i) = val(:, i) + B(j) * Pw(:, istart + j - 1)
        end do
        val(:, i) = val(:, i) / val(ndim, i)
    end do
end subroutine evalCurveNURBS
