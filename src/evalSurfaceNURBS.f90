!> Evaluate a NURBS surface at a set of parametric locations
!! @param u Array of u-parameters, of size (m x n)
!! @param v Array of v-parameters, of size (m x n)
!! @param uknotvec Knot vector in the u-direction, length (nctlu + udegree + 1)
!! @param vknotvec Knot vector in the v-direction, length (nctlv + vdegree + 1)
!! @param udegree Degree of the NURBS surface in the u-direction
!! @param vdegree Degree of the NURBS surface in the v-direction
!! @param Pw Array of control points and weights, of size (ndim+1, nctlv, nctlu)
!! @param nctlu Number of control points in the u-direction
!! @param nctlv Number of control points in the v-direction
!! @param ndim Number of spatial dimensions of the NURBS surface
!! @param m Number of points to evaluate in the u-direction
!! @param n Number of points to evaluate in the v-direction
!! @param[out] val Array of evaluated points, of size (ndim x m x n)
subroutine evalSurfaceNURBS(u, v, uknotvec, vknotvec, udegree, vdegree, Pw, nctlu, nctlv, ndim, m, n, val)

    use precision
    implicit none

    ! Input
    integer, intent(in) :: udegree, vdegree, nctlu, nctlv, ndim, n, m
    real(kind=realType), intent(in) :: u(m, n), v(m, n)
    real(kind=realType), intent(in) :: uknotvec(nctlu + udegree + 1), vknotvec(nctlv + vdegree + 1)
    real(kind=realType), intent(in) :: Pw(ndim + 1, nctlv, nctlu)

    ! Output
    real(kind=realType), intent(out) :: val(ndim, m, n)

    ! Working
    integer :: idim, istartu, istartv, ii, jj, i, j
    integer :: ileftu, ileftv
    real(kind=realType) :: Bu(udegree), Bv(vdegree)

    val(:, :, :) = 0.0
    do i = 1, n
        do j = 1, m
            ! U
            call findSpan(u(j, i), udegree, uknotvec, nctlu, ileftu)
            call basis(uknotvec, nctlu, udegree, u(j, i), ileftu, Bu)
            istartu = ileftu - udegree

            ! V
            call findSpan(v(j, i), vdegree, vknotvec, nctlv, ileftv)
            call basis(vknotvec, nctlv, vdegree, v(j, i), ileftv, Bv)
            istartv = ileftv - vdegree

            do ii = 1, udegree
                do jj = 1, vdegree
                    do idim = 1, ndim
                        val(idim, j, i) = val(idim, j, i) + Bu(ii) * Bv(jj) * &
                                          Pw(idim, istartv + jj, istartu + ii) / Pw(4, istartv + jj, istartu + ii)
                    end do
                end do
            end do
        end do
    end do

end subroutine evalSurfaceNURBS
