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
    real(kind=realType), intent(in) :: u(0:npts - 1)
    real(kind=realType), intent(in) :: knotvec(0:nctl + degree)
    real(kind=realType), intent(in) :: Pw(0:ndim - 1, 0:nctl - 1)

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
            val(:, i) = val(:, i) + B(j) * Pw(:, istart + j)
        end do
        val(:, i) = val(:, i) / val(ndim - 1, i)
    end do
end subroutine evalCurveNURBS

subroutine derivEvalCurveNURBS(u, knotVec, degree, Pw, nCtl, nDim, order, ck)
    ! NOTE: We use 0-based indexing to be consistent with algorithms
    ! in The NURBS Book.
    ! NOTE: The control points are the spatial dimension + 1 to account for
    ! the weights of the rational curve.
    use precision
    implicit none
    ! Input
    integer, intent(in) :: degree, nCtl, nDim, order
    real(kind=realType), intent(in) :: u
    real(kind=realType), intent(in) :: knotvec(0:nCtl + degree)
    real(kind=realType), intent(in) :: Pw(0:nDim - 1, 0:nCtl - 1)

    ! Output
    real(kind=realType), intent(out) :: ck(0:nDim - 2, 0:order)

    ! Working
    integer :: k, i, binCoeff
    real(kind=realType) :: v(0:nDim - 2), ckw(0:nDim - 1, 0:order)

    ! First we need to evaluate the derivative of the weighted control points using
    ! `derivEvalCurve` for a non-rational B-Spline.
    ! This will get A(u) and w(u) and we store them in `ckw`
    call derivEvalCurve(u, knotVec, degree, Pw, nCtl, nDim, order, ckw)

    ! Next we use Algorithm 4.2 from The NURBS Book to compute the true derivatives `ck`
    ck(:, :) = 0.0
    do k = 0, order
        v = ckw(0:nDim - 2, k)
        do i = 1, k + 1
            call bin(k, i, binCoeff)
            v = v - (binCoeff * ckw(nDim - 1, i) * ck(:, k - i))
        end do
        ck(:, k) = v / ckw(nDim - 1, 0)
    end do
    
end subroutine derivEvalCurveNURBS
