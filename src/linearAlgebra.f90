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