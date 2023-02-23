subroutine evalSurfaceNURBS(u, v, tu, tv, ku, kv, Pw, nctlu, nctlv, ndim, n, m, val)

    !***DESCRIPTION
    !
    !   Written by Andrew Lamkin
    !
    !   Abstract evalSurfaceNURBS evaluates many points on a rational
    !   BSpline surface.
    !
    !   Description of Arguments
    !   Inputs
    !   u       - Real, u coordinate, size(m,n)
    !   v       - Real, v coordinate, size(m,n)
    !   tu      - Real, knot vector in u, size(nctlu+ku)
    !   tv      - Real, knot vector in v, size(nctlv+dv)
    !   ku      - Integer, order of B-spline in u
    !   kv      - Integer, order of B-spline in v
    !   Pw      - Real, array of B-spline coefficients, size (ndim+1, nctlv, nctlu)
    !   nctlu   - Integer, number of control points in u
    !   nctlv   - Integer, number of control points in v
    !   ndim    - Integer, spatial dimension
    !
    !   Output
    !   val     - Real, evaluated point(s), size (ndim, m, n)

    use precision
    implicit none

    ! Input
    integer, intent(in) :: ku, kv, nctlu, nctlv, ndim, n, m
    real(kind=realType), intent(in) :: u(m, n), v(m, n)
    real(kind=realType), intent(in) :: tu(nctlu + ku), tv(nctlv + kv)
    real(kind=realType), intent(in) :: Pw(ndim + 1, nctlv, nctlu)

    ! Output
    real(kind=realType), intent(out) :: val(ndim, m, n)

    ! Working
    integer :: idim, istartu, istartv, i, j, ii, jj
    integer :: ileftu, ileftv
    real(kind=realType) :: basisu(ku), basisv(kv)

    val(:, :, :) = 0.0
    do ii = 1, n
        do jj = 1, m
            ! U
            call findSpan(u(jj, ii), ku, tu, nctlu, ileftu)
            call basis(tu, nctlu, ku, u(jj, ii), ileftu, basisu)
            istartu = ileftu - ku

            ! V
            call findSpan(v(jj, ii), kv, tv, nctlv, ileftv)
            call basis(tv, nctlv, kv, v(jj, ii), ileftv, basisv)
            istartv = ileftv - kv

            do i = 1, ku
                do j = 1, kv
                    do idim = 1, ndim
                        val(idim, jj, ii) = val(idim, jj, ii) + basisu(i) * basisv(j) * &
                                            Pw(idim, istartv + j, istartu + i) / Pw(4, istartv + j, istartu + i)
                    end do
                end do
            end do
        end do
    end do

end subroutine evalSurfaceNURBS
