!> Description:
!>  Finds the knot span of a given value within a knot vector for a B-spline curve or surface
!>
!> Inputs:
!>  u       - Real, The parameter value for which to find the knot span
!>  degree  - Integer, The degree of the B-spline curve or surface
!>  nCtl    - Integer, The number of control points for the B-spline curve or surface
!>  knotVec - Real, The knot vector for the B-spline curve or surface, length (nctl + degree + 1)
!>
!> Outputs:
!>  span    - Integer, The knot span that contains the input parameter value u within the knot vector
subroutine findSpan(u, degree, knotVec, nCtl, span)
    ! NOTE: We use 0-based indexing to be consistent with algorithms
    ! in The NURBS Book.
    use precision
    implicit none

    ! Inputs
    integer, intent(in) :: degree, nCtl
    real(kind=realType), intent(in) :: u
    real(kind=realType), intent(in) :: knotVec(0:nCtl + degree)

    ! Outputs
    integer, intent(out) :: span

    ! Working
    integer :: low, high, mid, n
    real(kind=realType) :: tol

    tol = 1e-6
    n = nCtl - 1

    if ((abs(knotVec(n + 1) - u) <= tol)) then
        span = n
        return
    end if

    ! Perform a binary search to find the knot interval containing u
    low = degree
    high = nCtl

    mid = (low + high) / 2
    do while ((u < knotVec(mid) .or. u >= knotVec(mid + 1)))
        if (u < knotVec(mid)) then
            high = mid
        else
            low = mid
        end if
        mid = (low + high) / 2
    end do

    span = mid
end subroutine findSpan
