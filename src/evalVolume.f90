subroutine evalVolume(u, v, w, uKnotVec, vKnotVec, wKnotVec, uDegree, vDegree, wDegree, P, &
                       nCtlu, nCtlv, nCtlw, nDim, n, m, l, val)

    !***DESCRIPTION
    !
    !     Written by Gaetan Kenway
    !
    !     Abstract eval_volume evaluates (possibly) many points on the
    !              b-spline volume
    !
    !     Description of Arguments
    !     Input
    !     u       - Real, u coordinate, size(l, m, n)
    !     v       - Real, v coordinate, size(l, m, n)
    !     w       - Real, w coordinate, size(l, m, n)
    !     tu      - Real, Knot vector in u. Length nctlu+ku
    !     tv      - Real, Knot vector in v. Length nctlv+kv
    !     tw      - Real, Knot vector in w. Length nctlv+kw
    !     ku      - Integer, order of B-spline in u
    !     kv      - Integer, order of B-spline in v
    !     kw      - Integer, order of B-spline in w
    !     coef    - Real, Array of B-spline coefficients
    !                 Size (ndim, nctlw, nctlv, nctlu)
    !     nctlu   - Integer, Number of control points in u
    !     nctlv   - Integer, Number of control points in v
    !     nctlw   - Integer, Number of control points in w
    !     ndim    - Integer, Spatial Dimension
    !
    !     Ouput
    !     val     - Real, Evaluated points, size (ndim, l, m, n)

    use precision
    implicit none

    ! Input
    integer, intent(in) :: uDegree, vDegree, wDegree, nCtlu, nCtlv, nCtlw
    integer, intent(in) :: nDim, n, m, l
    real(kind=realType), intent(in) :: u(l, m, n), v(l, m, n), w(l, m, n)
    real(kind=realType), intent(in) :: uKnotVec(nctlu + uDegree+1), vKnotVec(nctlv + vDegree+1), wKnotVec(nctlw + wDegree + 1)
    real(kind=realType), intent(in) :: P(nDim, nCtlw, nCtlv, nCtlu)

    ! Output
    real(kind=realType), intent(out) :: val(nDim, l, m, n)

    ! Working
    integer :: idim, istartu, istartv, istartw
    integer :: i, j, k, ii, jj, kk
    integer :: ileftu, ileftv, ileftw
    real(kind=realType) :: basisu(uDegree + 1), basisv(vDegree + 1), basisw(wDegree + 1)

    val(:, :, :, :) = 0.0
    do ii = 1, n
        do jj = 1, m
            do kk = 1, l
                ! U
                call findSpan(u(kk, jj, ii), uDegree, uKnotVec, nCtlu, ileftu)
                call basis(u(kk, jj, ii), uDegree, uKnotVec, ileftu, nCtlu, basisu)
                istartu = ileftu - uDegree + 1

                ! V
                call findSpan(v(kk, jj, ii), vDegree, vKnotVec, nCtlv, ileftv)
                call basis(v(kk, jj, ii), vDegree, vKnotVec, ileftv, nCtlv, basisv)
                istartv = ileftv - vDegree + 1

                ! W
                call findSpan(w(kk, jj, ii), wDegree, wKnotVec, nCtlw, ileftw)
                call basis(w(kk, jj, ii), wDegree, wKnotVec, ileftw, nCtlw, basisw)
                istartw = ileftw - wDegree + 1

                do i = 1, uDegree + 1
                    do j = 1, vDegree + 1
                        do k = 1, wDegree + 1
                            do idim = 1, ndim
                                val(idim, kk, jj, ii) = val(idim, kk, jj, ii) + &
                                                        basisu(i) * basisv(j) * basisw(k) * &
                                                        P(idim, istartw + k, istartv + j, istartu + i)
                            end do
                        end do
                    end do
                end do
            end do
        end do
    end do
end subroutine evalVolume

subroutine derivEvalVolume(u, v, w, uKnotVec, vKnotVec, wKnotVec, uDegree, vDegree, wDegree, P, order, &
                             nCtlu, nCtlv, nCtlw, nDim, vlmn)

    !***DESCRIPTION
    !
    !     Written by Gaetan Kenway
    !
    !     Abstract eval_volume_deriv evaluates the first derivative on a
    !              volume
    !
    !     Description of Arguments
    !     Input
    !     u       - Real, u coordinate
    !     v       - Real, v coordinate
    !     w       - Real, w coordinate
    !     tu      - Real, Knot vector in u. Length nctlu+ku
    !     tv      - Real, Knot vector in v. Length nctlv+kv
    !     tw      - Real, Knot vector in w. Length nctlv+kw
    !     ku      - Integer, order of B-spline in u
    !     kv      - Integer, order of B-spline in v
    !     kw      - Integer, order of B-spline in w
    !     coef    - Real, Array of B-spline coefficients
    !                 Size (ndim, nctlw, nctlv, nctlu)
    !     nctlu   - Integer, Number of control points in u
    !     nctlv   - Integer, Number of control points in v
    !     nctlw   - Integer, Number of control points in w
    !     ndim    - Integer, Spatial Dimension
    !
    !     Ouput
    !     val     - Real, Evaluated derivatvie, size(ndim, 3)

    use precision
    implicit none

    ! Input
    integer, intent(in) :: uDegree, vDegree, wDegree, nCtlu, nCtlv, nCtlw, nDim, order
    real(kind=realType), intent(in) :: u, v, w
    real(kind=realType), intent(in) :: uKnotVec(nCtlu + uDegree + 1), vKnotVec(nCtlv + vDegree + 1), wKnotVec(nCtlw + wDegree + 1)
    real(kind=realType), intent(in) :: P(nDim, nCtlw, nCtlv, nCtlu)

    ! Output
    real(kind=realType), intent(out) :: vlmn(0:nDim-1, 0:order, 0:order, 0:order)

    ! Working
    integer :: istartu, istartv, istartw, du, dv, dw, i, j, k, l, m, n
    integer :: ileftu, ileftv, ileftw
    real(kind=realType) :: Bdu(0:min(uDegree, order), 0:uDegree), Bdv(0:min(vDegree, order), 0:vDegree), Bdw(0:min(wDegree, order), 0:wDegree)

    ! Initialize the derivatives to zeros
    vlmn(:, :, :, :) = 0.0

    ! Get the highest available derivaitve order
    ! (Can only be as big as the degree in each parameteric direction)
    du = min(uDegree, order)
    dv = min(vDegree, order)
    dw = min(wDegree, order)

    ! Evaluate the span and basis functions in the u, v, and w directions
    call findSpan(u, uDegree, uKnotVec, nCtlu, ileftu)
    call derivBasis(u, uDegree, uKnotVec, ileftu, nCtlu, du, Bdu)
    istartu = ileftu - uDegree

    call findSpan(v, vDegree, vKnotVec, nCtlv, ileftv)
    call derivBasis(v, vDegree, vKnotVec, ileftv, nCtlv, dv, Bdv)
    istartv = ileftv - vDegree

    call findSpan(w, wDegree, wKnotVec, nCtlw, ileftw)
    call derivBasis(w, wDegree, wKnotVec, ileftw, nCtlw, dw, Bdw)
    istartw = ileftw - wDegree

    ! Loop over the u, v, and w degree starting at 0 (4 values).
    do i = 0, uDegree
        do j = 0, vDegree
            do k = 0, wDegree
                ! Loop over the requested derivative order
                do l=0,du
                    do m=0,dv
                        do n=0,dw
                            ! Sum the contributions from each partial derivative
                            vlmn(:, l, m, n) = vlmn(:, l, m, n) + Bdu(l, i) * Bdv(m, j) * Bdw(n, k) * P(:, istartw + k, istartv + j, istartu + i)
                        end do
                    end do
                end do
            end do
        end do 
    end do
end subroutine derivEvalVolume
