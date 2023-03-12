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
    ! NOTE: We use 0-based indexing to be consistent with algorithms
    ! in The NURBS Book.
    use precision
    implicit none

    ! Input
    integer, intent(in) :: udegree, vdegree, nctlu, nctlv, ndim, n, m
    real(kind=realType), intent(in) :: u(0:m - 1, 0:n - 1), v(0:m - 1, 0:n - 1)
    real(kind=realType), intent(in) :: uknotvec(0:nctlu + udegree), vknotvec(0:nctlv + vdegree)
    real(kind=realType), intent(in) :: Pw(0:ndim - 1, 0:nctlv - 1, 0:nctlu - 1)

    ! Output
    real(kind=realType), intent(out) :: val(0:ndim - 1, 0:m - 1, 0:n - 1)

    ! Working
    integer :: istartu, istartv, ii, jj, i, j
    integer :: ileftu, ileftv
    real(kind=realType) :: Bu(0:udegree), Bv(0:vdegree)

    val(:, :, :) = 0.0
    do i = 0, n - 1
        do j = 0, m - 1
            ! Get the span and evaluate the basis functions in the u-directions
            call findSpan(u(j, i), udegree, uknotvec, nctlu, ileftu)
            call basis(uknotvec, nctlu, udegree, u(j, i), ileftu, Bu)
            istartu = ileftu - udegree

            ! Get the span and evaluate the basis functions in the v-directions
            call findSpan(v(j, i), vdegree, vknotvec, nctlv, ileftv)
            call basis(vknotvec, nctlv, vdegree, v(j, i), ileftv, Bv)
            istartv = ileftv - vdegree

            ! Loop over the basis functions up to the u and v degrees and evaluate each point
            do ii = 0, udegree
                do jj = 0, vdegree
                    val(:, j, i) = val(:, j, i) + Bu(ii) * Bv(jj) * Pw(:, istartv + jj, istartu + ii)
                end do
            end do

            ! Divide by the control point weights
            val(:, j, i) = val(:, j, i) / val(nDim - 1, j, i)
        end do
    end do

end subroutine evalSurfaceNURBS
