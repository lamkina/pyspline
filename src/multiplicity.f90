subroutine multiplicity(u, knotvec, nctl, degree, mult)
    use precision
    implicit none

    ! Input
    integer, intent(in) :: degree, nctl
    real(kind=realType), intent(in) :: u
    real(kind=realType), intent(in) :: knotvec(0:nctl + degree)

    ! Output
    integer, intent(out) :: mult

    ! Working
    real(kind=realType) :: tol
    integer :: i

    tol = 1e-8

    mult = 0
    do i = 0, nctl + degree
        if (abs(u - knotvec(i)) <= tol) then
            mult = mult + 1
        end if
    end do
end subroutine multiplicity
