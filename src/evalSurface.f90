!> Description:
!>  Evaluates a surface represented by a B-spline using the provided knot vectors,
!>  control points, and degrees at the given parameter values.
!>
!> Inputs:
!>  u         - Real, The parameter values in the u-direction at which to evaluate the surface, shape (m, n)
!>  v         - Real, The parameter values in the v-direction at which to evaluate the surface, shape (m, n)
!>  uknotvec  - Real, The knot vector in the u-direction, length (nctlu + udegree + 1)
!>  vknotvec  - Real, The knot vector in the v-direction, length (nctlv + vdegree + 1)
!>  udegree   - Integer, The degree of the curve in the u-direction.
!>  vdegree   - Integer, The degree of the curve in the v-direction.
!>  P         - Real, The control points of the surface, shape (ndim, nctlv, nctlu)
!>  nctlu     - Integer, The number of control points in the u-direction.
!>  nctlv     - Integer, The number of control points in the v-direction.
!>  ndim      - Integer, The dimension of the surface (typically 1, 2, or 3).
!>  m         - Integer, The number of parameteric points to evaluate in each direction.
!>  n         - Integer, The number of parameteric points to evaluate in each direction.
!>
!> Outputs:
!>  val       - Real, The output array of evaluated points on the surface, shape (ndim, m, n).
subroutine evalSurface(u, v, uknotvec, vknotvec, udegree, vdegree, P, nctlu, nctlv, ndim, n, m, val)
    use precision
    implicit none

    ! Input
    integer, intent(in) :: udegree, vdegree, nctlu, nctlv, ndim, n, m
    real(kind=realType), intent(in) :: u(m, n), v(m, n)
    real(kind=realType), intent(in) :: uknotvec(nctlu + udegree + 1), vknotvec(nctlv + vdegree + 1)
    real(kind=realType), intent(in) :: P(ndim, nctlv, nctlu)

    ! Output
    real(kind=realType), intent(out) :: val(ndim, m, n)

    ! Working
    integer :: istartu, istartv, ii, jj, i, j
    integer :: ileftu, ileftv
    real(kind=realType) :: Bu(udegree), Bv(vdegree)

    val(:, :, :) = 0.0
    do i = 1, n
        do j = 1, m
            ! U
            call findSpan(u(j, i), udegree, uknotvec, nctlu, ileftu)
            call basis(u(j, i), udegree, uknotvec, ileftu, nctlu, Bu)
            istartu = ileftu - udegree

            ! V
            call findSpan(v(j, i), vdegree, vknotvec, nctlv, ileftv)
            call basis(v(j, i), vdegree, vknotvec, ileftv, nctlv, Bv)
            istartv = ileftv - vdegree

            do ii = 1, udegree
                do jj = 1, vdegree
                    val(:, j, i) = val(:, j, i) + Bu(ii) * Bv(jj) * P(:, istartv + jj, istartu + ii)
                end do
            end do
        end do
    end do

end subroutine evalSurface
