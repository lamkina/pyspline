subroutine buildSurfaceCoeffMatrix(u, v, uKnotVec, vKnotVec, uDegree, vDegree, nCtlu, nCtlv, nu, nv, vals, &
                                 rowPtr, colInd)

    use precision
    implicit none

    ! Input
    integer, intent(in) :: uDegree, vDegree, nCtlu, nCtlv, nu, nv
    real(kind=realType), intent(in) :: u(nv, nu), v(nv, nu)
    real(kind=realType), intent(in) :: uKnotVec(nCtlu + uDegree + 1), vKnotVec(nCtlv + vDegree + 1)

    ! Output
    real(kind=realType), intent(out) :: vals(nu * nv * (uDegree + 1) * (vDegree + 1))
    integer, intent(out) :: colInd(nu * nv * (uDegree + 1) * (vDegree + 1)), rowPtr(nu * nv + 1)

    ! Working
    real(kind=realType) :: basisu(uDegree + 1), basisv(vDegree + 1)
    integer :: ileftu, ileftv
    integer :: i, j, ii, jj, counter, ku, kv

    ku = uDegree + 1
    kv = vDegree + 1
    counter = 1
    do i = 1, nu
        do j = 1, nv
            ! Get u interval
            call findSpan(u(j, i), uDegree, uKnotVec, nCtlu, ileftu)
            call basis(u(j, i), uDegree, uKnotVec, ileftu, nCtlu, basisu)
            
            ! Convert to 1 based index
            ileftu = ileftu + 1

            ! Get v interval
            call findSpan(v(j, i), vDegree, vKnotVec, nCtlv, ileftv)
            call basis(v(j, i), vDegree, vKnotVec, ileftv, nCtlv, basisv)
            
            ! Convert to 1 based index
            ileftv = ileftv + 1

            rowPtr((i - 1) * nv + j) = counter - 1
            do ii = 1, ku
                do jj = 1, kv
                    colInd(counter) = (ileftu - ku + ii - 1) * nCtlv + (ileftv - kv + jj - 1)
                    vals(counter) = basisu(ii) * basisv(jj)
                    counter = counter + 1

                end do
            end do
        end do
    end do
    rowPtr(nu * nv + 1) = counter - 1

end subroutine buildSurfaceCoeffMatrix

subroutine surfaceParamCorr(uKnotVec, vKnotVec, uDegree, vDegree, u, v, P, nCtlu, nCtlv, nDim, nu, nv, X, rms)

    ! Do Hoschek parameter correction
    use precision
    implicit none

    ! Input/Output
    integer, intent(in) :: uDegree, vDegree, nCtlu, nCtlv, nDim, nu, nv
    real(kind=realType), intent(in) :: uKnotVec(uDegree + nCtlu + 1), vKnotVec(vDegree + nCtlv + 1)
    real(kind=realType), intent(inout) :: u(nv, nu), v(nv, nu)
    real(kind=realType), intent(in) :: P(nDim, nCtlv, nCtlu)
    real(kind=realType), intent(in) :: X(nDim, nv, nu)
    real(kind=realType), intent(out) :: rms

    ! Working
    integer :: i, j, jj, maxInnerIter, ku, kv
    real(kind=realType) :: D(nDim), D2(nDim)
    real(kind=realType) :: val(nDim), deriv(nDim, 3, 3)
    real(kind=realType) :: uTilde, vTilde
    real(kind=realType) :: A(2, 2), ki(2), delta(2)

    ku = uDegree + 1
    kv = vDegree + 1

    maxInnerIter = 10
    rms = 0.0

    do i = 2, nu - 1
        do j = 2, nv - 2
            call evalSurface(u(j, i), v(j, i), uKnotVec, vKnotVec, uDegree, vDegree, P, nCtlu, nCtlv, nDim, 1, 1, val)
            call derivEvalSurface(u(j, i), v(j,i), uKnotVec, vKnotVec, uDegree, vDegree, P, 2, nCtlu, nCtlv, nDim, deriv)

            D = val - X(:, j, i)

            A(1, 1) = NORM2(deriv(:, 2, 1))**2 + dot_product(D, deriv(:, 3, 1))
            A(1, 2) = dot_product(deriv(:, 2, 1), deriv(:,1, 2)) + dot_product(D, deriv(:, 2, 2))
            A(2, 1) = A(1, 2)
            A(2, 2) = NORM2(deriv(:, 1, 2))**2 + dot_product(D, deriv(:, 1, 3))

            ki(1) = -dot_product(D, deriv(:, 2, 1))
            ki(2) = -dot_product(D, deriv(:, 1, 2))

            call solve2by2(A, ki, delta)

            if (j .eq. 1 .or. j .eq. nv) then
                delta(1) = 0.0
            end if
            if (i .eq. 1 .or. i .eq. nu) then
                delta(2) = 0.0
            end if
            innerLoop: do jj = 1, maxInnerIter
                uTilde = u(j, i) + delta(1)
                vTilde = v(j, i) + delta(2)

                call evalSurface(uTilde, vTilde, uKnotVec, vKnotVec, uDegree, vDegree, P, nCtlu, nCtlv, nDim, 1, 1, val)
                D2 = val - X(:, j, i)
                if (NORM2(D) .ge. NORM2(D2)) then
                    u(j, i) = uTilde
                    v(j, i) = vTilde
                    exit innerLoop
                else
                    delta = delta * 0.5
                end if
            end do innerLoop
        end do
    end do

    ! Lets redo the full RMS
    rms = 0.0

    do i = 1, nu
        do j = 1, nv
            call evalSurface(u(j, i), v(j, i), uKnotVec, vKnotVec, ku-1, kv-1, P, nCtlu, nCtlv, nDim, 1, 1, val)
            D = val - X(:, j, i)
            rms = rms + dot_product(D, D)
        end do
    end do
    rms = sqrt(rms / (nu * nv))
end subroutine surfaceParamCorr

! function compute_rms_surface(tu, tv, ku, kv, u, v, coef, nctlu, nctlv, ndim, nu, nv, X)
!     ! Do Hoschek parameter correction
!     use precision
!     implicit none

!     ! Input/Output
!     integer, intent(in) :: ku, kv, nctlu, nctlv, ndim, nu, nv
!     real(kind=realType), intent(in) :: tu(ku + nctlu), tv(kv + nctlv)
!     real(kind=realType), intent(inout) :: u(nv, nu), v(nv, nu)
!     real(kind=realType), intent(in) :: coef(ndim, nctlv, nctlu)
!     real(kind=realType), intent(in) :: X(ndim, nv, nu)

!     ! Working
!     integer :: i, j, idim
!     real(kind=realType) :: val(ndim), D(ndim)
!     real(kind=realType) :: compute_rms_surface

!     compute_rms_surface = 0.0
!     do i = 1, nu
!         do j = 1, nv
!             call eval_surface(u(j, i), v(j, i), tu, tv, ku, kv, coef, nctlu, nctlv, ndim, val)
!             D = val - X(:, j, i)
!             do idim = 1, ndim
!                 compute_rms_surface = compute_rms_surface + D(idim)**2
!             end do
!         end do
!     end do
!     compute_rms_surface = sqrt(compute_rms_surface / (nu * nv))

! end function compute_rms_surface

