!> Description:
!>   Given a knot vector, a degree, and a parameter value u, compute the
!>   basis functions for the corresponding B-spline curve segment. The basis
!>   functions are computed using the Cox-de Boor recursion formula.
!>
!> Inputs:
!>   degree    - Integer, degree of the B-spline basis functions
!>   span      - Integer, index of the knot span containing the parameter value u
!>   nctl      - Integer, number of control points in the curve
!>   u         - Real, parameter value
!>   knotvec   - Real, knot vector (array of size nctl + degree + 1)
!>
!> Outputs:
!>   B         - Real, array containing the basis functions (of size degree + 1)
!>
subroutine basis(u, degree, knotvec, span, nctl, B)
    use precision
    implicit none

    ! Inputs
    integer, intent(in) :: degree, span, nctl
    real(kind=realType), intent(in) :: u
    real(kind=realType), intent(in) :: knotvec(nctl + degree + 1)

    ! Outputs
    real(kind=realType), intent(out) :: B(degree + 1)

    ! Working
    integer :: j, r
    real(kind=realType) :: left(degree + 1), right(degree + 1), saved, temp

    B(1) = 1.0

    do j = 2, degree + 1
        left(j) = u - knotvec(span + 2 - j)
        right(j) = knotvec(span - 1 + j) - u
        saved = 0.0
        do r = 1, j - 1
            temp = B(r) / (right(r + 1) + left(j + 1 - r))
            B(r) = saved + right(r + 1) * temp
            saved = left(j + 1 - r) * temp
        end do
        B(j) = saved
    end do
end subroutine basis
