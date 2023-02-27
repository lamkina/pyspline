!> Description:
!>  Finds the knot span of a given value within a knot vector for a B-spline curve or surface
!>
!> Inputs:
!>  u       - Real, The parameter value for which to find the knot span
!>  degree  - Integer, The degree of the B-spline curve or surface
!>  nctl    - Integer, The number of control points for the B-spline curve or surface
!>  knotvec - Real, The knot vector for the B-spline curve or surface, length (nctl + degree + 1)
!>
!> Outputs:
!>  span    - Integer, The knot span that contains the input parameter value u within the knot vector
subroutine findSpan(u, degree, knotvec, nctl, span)
    use precision
    implicit none
    integer, intent(in) :: degree, nctl
    real(kind=realType), intent(in) :: u
    real(kind=realType), intent(in) :: knotvec(nctl + degree + 1)
    integer, intent(out) :: span

    integer :: low, high, mid
    real(kind=realType) :: tol

    tol=1e-6

    if ((abs(knotvec(nctl+1) - u) <= tol)) then
        span = nctl
        return
    end if

    ! Check if u is outside the knot vector range
    ! if (u < knotvec(degree + 2)) then
    !     span = degree
    !     return
    ! elseif (u >= knotvec(nctl + 1)) then
    !     span = nctl + 1
    !     return
    ! end if

    ! Perform a binary search to find the knot interval containing u
    low = degree + 1
    high = nctl + 1

    mid = (low + high) / 2
    do while ((u < knotvec(mid) .or. u >= knotvec(mid + 1)))
        if (u < knotvec(mid)) then
            high = mid
        else
            low = mid
        end if
        mid = (low + high) / 2
    end do

    span = mid
end subroutine findSpan
