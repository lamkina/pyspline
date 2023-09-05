subroutine solve2by2(A, b, x)

    use precision
    implicit none

    ! Solve a 2 x 2 system  -- With NO checking
    real(kind=realType), intent(in) :: A(2, 2), b(2)
    real(kind=realType), intent(out) :: x(2)
    real(kind=realType) :: idet, det

    det = A(1, 1) * A(2, 2) - A(1, 2) * A(2, 1)
    if (det == 0) then
        X = B
    else
        idet = 1.0 / det
        X(1) = idet * (B(1) * A(2, 2) - B(2) * A(1, 2))
        X(2) = idet * (A(1, 1) * B(2) - B(1) * A(2, 1))
    end if

end subroutine solve2by2

subroutine solve3by3(A, b, x)

    use precision
    implicit none

    ! Solve a 3 x 3 system  -- With NO checking
    real(kind=realType), intent(in) :: A(3, 3), b(3)
    real(kind=realType), intent(out) :: x(3)
    real(kind=realType) :: idet

    idet = 1/(A(1,1)*(A(3,3)*A(2,2)-A(3,2)*A(2,3))-A(2,1)*(A(3,3)*A(1,2)-A(3,2)*A(1,3))+A(3,1)*(A(2,3)*A(1,2)-A(2,2)*A(1,3)))
    x(1) = idet*( b(1)*(A(3,3)*A(2,2)-A(3,2)*A(2,3)) - b(2)*(A(3,3)*A(1,2)-A(3,2)*A(1,3)) + b(3)*(A(2,3)*A(1,2)-A(2,2)*A(1,3)))
    x(2) = idet*(-b(1)*(A(3,3)*A(2,1)-A(3,1)*A(2,3)) + b(2)*(A(3,3)*A(1,1)-A(3,1)*A(1,3)) - b(3)*(A(2,3)*A(1,1)-A(2,1)*A(1,3)))
    x(3) = idet*( b(1)*(A(3,2)*A(2,1)-A(3,1)*A(2,2)) - b(2)*(A(3,2)*A(1,1)-A(3,1)*A(1,2)) + b(3)*(A(2,2)*A(1,1)-A(2,1)*A(1,2)))

    ! | a11 a12 a13 |-1             |   a33a22-a32a23  -(a33a12-a32a13)   a23a12-a22a13  |
    ! | a21 a22 a23 |    =  1/DET * | -(a33a21-a31a23)   a33a11-a31a13  -(a23a11-a21a13) |
    ! | a31 a32 a33 |               |   a32a21-a31a22  -(a32a11-a31a12)   a22a11-a21a12  |

    ! DET  =  a11(a33a22-a32a23)-a21(a33a12-a32a13)+a31(a23a12-a22a13)

end subroutine solve3by3

function dotproduct(x1, x2, n)
    use precision
    implicit none

    integer, intent(in) :: n
    integer :: i
    real(kind=realType), intent(in) :: x1(n), x2(n)

    real(kind=realType) :: dotproduct
    dotproduct = 0.0
    do i = 1, n
        dotproduct = dotproduct + x1(i) * x2(i)
    end do

end function dotproduct

subroutine cross(a, b, cross_prod)
    use precision
    implicit none

    real(kind=realType), dimension(3), intent(in) :: a, b
    real(kind=realType), dimension(3), intent(out) :: cross_prod

    cross_prod(1) = a(2) * b(3) - a(3) * b(2)
    cross_prod(2) = a(3) * b(1) - a(1) * b(3)
    cross_prod(3) = a(1) * b(2) - a(2) * b(1)

end subroutine cross

subroutine lusolve(L, U, P, x, b, n)
    use precision
    implicit none

    ! Inputs
    integer, intent(in) :: n
    real(kind=realType), intent(in) :: L(n, n), U(n, n), P(n, n)
    real(kind=realType), intent(in) :: b(n)

    ! Outputs
    real(kind=realType), intent(out) :: x(n)

    ! Working
    real(kind=realType) :: tmp, z(n), y(n)
    integer :: i, j

    x(:) = 0.0
    y(:) = 0.0

    z = matmul(P, b)

    ! Do the forward substitution
    do i = 1, n
        tmp = z(i)
        do j = 1, i - 1
            tmp = tmp - L(i, j) * y(j)
        end do
        y(i) = tmp / L(i, i)
    end do

    ! Do the back substitution
    do i = n, 1, -1
        tmp = y(i)
        do j = i + 1, n
            tmp = tmp - U(i, j) * x(j)
        end do
        x(i) = tmp / U(i, i)
    end do
end subroutine lusolve

subroutine ludecomp(A, L, U, P, n)
    ! in situ decomposition, corresponds to LAPACK's dgebtrf
    use precision
    implicit none

    ! Inputs
    integer, intent(in) :: n
    real(kind=realType), intent(in) :: A(n, n)

    ! Outputs
    real(kind=realType), intent(out) :: L(n, n)
    real(kind=realType), intent(out) :: U(n, n)
    real(kind=realType), intent(out) :: P(n, n)

    ! Working
    real(kind=realType) :: AA(n, n)
    integer :: ipiv(n)
    integer :: i, j, k, kmax

    forall (j=1:n, i=1:n)
        AA(i, j) = A(i, j)
        U(i, j) = 0d0
        P(i, j) = merge(1, 0, i .eq. j)
        L(i, j) = merge(1d0, 0d0, i .eq. j)
    end forall

    ipiv = [(i, i=1, n)]
    do k = 1, n - 1
        kmax = maxloc(abs(A(ipiv(k:), k)), 1) + k - 1
        if (kmax /= k) then
            ipiv([k, kmax]) = ipiv([kmax, k])
            AA([k, kmax], :) = AA([kmax, k], :)
        end if
        AA(k + 1:, k) = AA(k + 1:, k) / AA(k, k)
        forall (j=k + 1:n) AA(k + 1:, j) = AA(k + 1:, j) - AA(k, j) * AA(k + 1:, k)
    end do

    do i = 1, n
        L(i, :i - 1) = AA(i, :i - 1)
        U(i, i:) = AA(i, i:)
    end do

    P(ipiv, :) = P
end subroutine ludecomp

subroutine mtxPrint(title, matrix, rows, cols)
    use precision
    implicit none

    integer, intent(in) :: rows, cols
    real(kind=realType), dimension(rows, cols), intent(in) :: matrix
    character(len=*), intent(in), optional :: title
    integer :: i, j

    if (present(title)) then
        write (*, '(A)', advance="no") title
    end if

    write (*, *) ! Move to the next line after the title

    do i = 1, rows
        do j = 1, cols
            write (*, '(F8.4, "  ")', advance="no") matrix(i, j)
        end do
        write (*, *) ! Move to the next line after each row
    end do
end subroutine mtxPrint

