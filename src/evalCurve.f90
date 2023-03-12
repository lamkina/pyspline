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
    ! NOTE: We use 0-based indexing to be consistent with algorithms
    ! in The NURBS Book.
    use precision
    implicit none
    ! Input
    integer, intent(in) :: degree, nctl, ndim, npts
    real(kind=realType), intent(in) :: u(0:npts - 1)
    real(kind=realType), intent(in) :: knotvec(0:nctl + degree)
    real(kind=realType), intent(in) :: P(0:ndim - 1, 0:nctl - 1)

    ! Output
    real(kind=realType), intent(out) :: val(0:ndim - 1, 0:npts - 1)

    ! Working
    integer :: i, j, istart, ileft
    real(kind=realType) :: B(0:degree)

    val(:, :) = 0.0
    do i = 0, npts - 1
        call findSpan(u(i), degree, knotvec, nctl, ileft)
        call basis(u(i), degree, knotvec, ileft, nctl, B)
        istart = ileft - degree
        do j = 0, degree
            val(:, i) = val(:, i) + B(j) * P(:, istart + j)
        end do
    end do
end subroutine evalCurve

subroutine derivEvalCurve(u, knotVec, degree, P, nCtl, nDim, order, ck)
    ! NOTE: We use 0-based indexing to be consistent with algorithms
    ! in The NURBS Book.
    use precision
    implicit none
    ! Input
    integer, intent(in) :: degree, nctl, ndim, order
    real(kind=realType), intent(in) :: u
    real(kind=realType), intent(in) :: knotvec(0:nctl + degree)
    real(kind=realType), intent(in) :: P(0:ndim - 1, 0:nctl - 1)

    ! Output
    real(kind=realType), intent(out) :: ck(0:ndim - 1, 0:order)

    ! Working
    integer :: du, k, j, span
    real(kind=realType) :: Bd(0:min(degree, order), 0:degree)

    du = min(degree, order)

    ck(:, :) = 0.0

    call findSpan(u, degree, knotVec, nCtl, span)
    call derivBasis(u, degree, knotVec, span, nCtl, order, Bd)

    do k = 0, du
        ck(:, k) = 0.0
        do j = 0, degree
            ck(:, k) = ck(:, k) + Bd(k, j) * P(:, span - degree + j)
        end do
    end do
end subroutine derivEvalCurve
