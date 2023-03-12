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
    ! NOTE: We use 0-based indexing to be consistent with algorithms
    ! in The NURBS Book.
    use precision
    implicit none

    ! Input
    integer, intent(in) :: udegree, vdegree, nctlu, nctlv, ndim, n, m
    real(kind=realType), intent(in) :: u(0:m - 1, 0:n - 1), v(0:m - 1, 0:n - 1)
    real(kind=realType), intent(in) :: uknotvec(0:nctlu + udegree), vknotvec(0:nctlv + vdegree)
    real(kind=realType), intent(in) :: P(0:ndim - 1, 0:nctlv - 1, 0:nctlu - 1)

    ! Output
    real(kind=realType), intent(out) :: val(0:ndim - 1, 0:m - 1, 0:n - 1)

    ! Working
    integer :: istartu, istartv, ii, jj, i, j
    integer :: ileftu, ileftv
    real(kind=realType) :: Bu(0:udegree), Bv(0:vdegree)

    val(:, :, :) = 0.0
    do i = 0, n - 1
        do j = 0, m - 1
            ! U
            call findSpan(u(j, i), udegree, uknotvec, nctlu, ileftu)
            call basis(u(j, i), udegree, uknotvec, ileftu, nctlu, Bu)
            istartu = ileftu - udegree

            ! V
            call findSpan(v(j, i), vdegree, vknotvec, nctlv, ileftv)
            call basis(v(j, i), vdegree, vknotvec, ileftv, nctlv, Bv)
            istartv = ileftv - vdegree

            do ii = 0, udegree
                do jj = 0, vdegree
                    val(:, j, i) = val(:, j, i) + Bu(ii) * Bv(jj) * P(:, istartv + jj, istartu + ii)
                end do
            end do
        end do
    end do

end subroutine evalSurface

subroutine derivEvalSurface(u, v, uknotvec, vknotvec, udegree, vdegree, P, nctlu, nctlv, ndim, order, skl)
    ! NOTE: We use 0-based indexing to be consistent with algorithms
    ! in The NURBS Book.
    use precision
    implicit none

    ! Input
    integer, intent(in) :: udegree, vdegree, nctlu, nctlv, ndim, order
    real(kind=realType), intent(in) :: u, v
    real(kind=realType), intent(in) :: uknotvec(0:nctlu + udegree), vknotvec(0:nctlv + vdegree)
    real(kind=realType), intent(in) :: P(0:ndim - 1, 0:nctlv - 1, 0:nctlu - 1)

    ! Output
    real(kind=realType), intent(out) :: skl(0:nDim-1, 0:order, 0:order)

    ! Working
    integer :: istartu, istartv, ii, jj, i, j, du, dv, dd, k
    integer :: ileftu, ileftv
    real(kind=realType) :: Bdu(0:min(udegree, order), 0:udegree), Bdv(0:min(vdegree, order), 0:vdegree)
    real(kind=realType) :: temp(0:vdegree, 0:nDim)

    ! Initialize the derivatives to zeros
    skl(:, :, :) = 0.0

    ! Get the highest available derivative order
    ! (Can only be as big as the degree in each parameteric direction)
    du = min(udegree, order)
    dv = min(vdegree, order)

    ! Evaluate the span and basis function derivatives in the u-direction
    call findSpan(u, udegree, uknotvec, nctlu, ileftu)
    call derivBasis(u, udegree, uknotvec, ileftu, nctlu, du, Bdu)
    istartu = ileftu - udegree

    ! Evaluate the spand and basis derivatives in the v-direction
    call findSpan(v, vdegree, vknotvec, nctlv, ileftv)
    call derivBasis(v, vdegree, vknotvec, ileftv, nctlv, dv, Bdv)
    istartv = ileftv - vdegree

    do k = 0, du
        temp(:, :) = 0.0
        do i = 0, vdegree
            do j = 0, udegree
                temp(i, :) = temp(i, :) + Bdu(k, j) * P(:, istartv + i, istartu + j)
            end do
        end do

        dd = min(order - k, dv)
        do ii = 0, dd
            do jj = 0, vdegree
                skl(:, ii, k) = skl(:, ii, k) + Bdv(ii, jj) * temp(jj, :)
            end do
        end do
    end do

end subroutine derivEvalSurface
