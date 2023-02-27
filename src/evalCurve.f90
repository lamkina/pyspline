!> Description:
!>  Evaluates a BSpline curve at the given parameter values using the provided knot vector,
!>  control points, and weights
!>
!> Inputs:
!>  u        - Real, The parameter values at which to evaluate the curve.
!>  knotvec  - Real, The knot vector, length (nctl + degree + 1)
!>  degree   - Integer, The degree of the curve.
!>  P        - Real, The control points of the bspline, shape (ndim, nctl)
!>  nctl     - Integer, The number of control points
!>  ndim     - Integer, The dimension of the curve (typically 1, 2, or 3)
!>  npts     - Integer, The number of parameteric points to evaluate
!>
!> Outputs:
!>  val:     - Real, The output array of evaluated points on the curve
subroutine evalCurve(u, knotvec, degree, P, nctl, ndim, npts, val)
    use precision
    implicit none
    ! Input
    integer, intent(in) :: degree, nctl, ndim, npts
    real(kind=realType), intent(in) :: u(npts)
    real(kind=realType), intent(in) :: knotvec(nctl + degree + 1)
    real(kind=realType), intent(in) :: P(ndim, nctl)

    ! Output
    real(kind=realType), intent(out) :: val(ndim, npts)

    ! Working
    integer :: i, j, istart, ileft
    real(kind=realType) :: B(degree)

    val(:, :) = 0.0
    do i = 1, npts
        call findSpan(u(i), degree, knotvec, nctl, ileft)
        call basis(u(i), degree, knotvec, ileft, nctl, B)
        istart = ileft - degree
        do j = 1, degree + 1
            val(:, i) = val(:, i) + B(j) * P(:, istart + j)
        end do
    end do
end subroutine evalCurve
