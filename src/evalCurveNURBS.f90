subroutine evalCurveNURBS(s, t, k, Pw, nctl, ndim, n, val)

    !***DESCRIPTION
    !
    !   Written by Andrew Lamkin
    !
    !   A vectorized version of the "CurvePoint" algorithm A4.1 from
    !   the NURBS book.
    !
    !   Description of Arguments:
    !   Input
    !   s       - Real, Vector of parameteric coordinates, length n
    !   t       - Real, Knot vector, length nctl+k
    !   k       - Integer, Degree of B-spline
    !   Pw      - Array of weighted control point coefficients, size (ndim+1, nctl)
    !             One column of Pw=(w*x, w*y, w*z, w) where x, y, and z
    !             are the locations of the control point coefficients and
    !             "w" is the weight.
    !
    !   Output
    !   val       - Real, Array of evaluated points, size (ndim, n)

    use precision
    implicit none

    ! Input
    integer, intent(in) :: k, nctl, ndim, n
    real(kind=realType), intent(in) :: s(n)
    real(kind=realType), intent(in) :: t(nctl + k)
    real(kind=realType), intent(in) :: Pw(ndim + 1, nctl)

    ! Output
    real(kind=realType), intent(out) :: val(ndim, n)

    ! Working
    integer :: i, j, idim, istart, ileft
    real(kind=realType) :: basisu(k)

    val(:, :) = 0.0
    do i = 1, n
        call findSpan(s(i), k, t, nctl, ileft)
        call basis(t, nctl, k, s(i), ileft, basisu)
        istart = ileft - k
        do j = 1, k
            do idim = 1, ndim
                val(idim, i) = (val(idim, i) + basisu(j) * Pw(idim, istart + j)) / Pw(4, istart + j)
            end do
        end do
    end do
end subroutine evalCurveNURBS
