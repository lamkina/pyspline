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
subroutine evalSurfaceNURBS(u, v, uKnotVec, vKnotVec, uDegree, vDegree, Pw, nCtlu, nCtlv, nDim, m, n, val)
    ! NOTE: We use 0-based indexing to be consistent with algorithms
    ! in The NURBS Book.
    use precision
    implicit none

    ! Input
    integer, intent(in) :: uDegree, vDegree, nCtlu, nCtlv, nDim, n, m
    real(kind=realType), intent(in) :: u(0:m - 1, 0:n - 1), v(0:m - 1, 0:n - 1)
    real(kind=realType), intent(in) :: uKnotVec(0:nCtlu + uDegree), vKnotVec(0:nCtlv + vDegree)
    real(kind=realType), intent(in) :: Pw(0:nDim - 1, 0:nCtlv - 1, 0:nCtlu - 1)

    ! Output
    real(kind=realType), intent(out) :: val(0:nDim - 1, 0:m - 1, 0:n - 1)

    ! Working
    integer :: istartu, istartv, ii, jj, i, j
    integer :: ileftu, ileftv
    real(kind=realType) :: Bu(0:uDegree), Bv(0:vDegree)

    val(:, :, :) = 0.0
    do i = 0, n - 1
        do j = 0, m - 1
            ! U
            call findSpan(u(j, i), uDegree, uKnotVec, nCtlu, ileftu)
            call basis(u(j, i), uDegree, uKnotVec, ileftu, nCtlu, Bu)
            istartu = ileftu - uDegree

            ! V
            call findSpan(v(j, i), vDegree, vKnotVec, nCtlv, ileftv)
            call basis(v(j, i), vDegree, vKnotVec, ileftv, nCtlv, Bv)
            istartv = ileftv - vDegree

            ! Loop over the basis functions up to the u and v degrees and evaluate each point
            do ii = 0, uDegree
                do jj = 0, vDegree
                    val(:, j, i) = val(:, j, i) + Bu(ii) * Bv(jj) * Pw(:, istartv + jj, istartu + ii)
                end do
            end do

            ! Divide by the control point weights
            val(:, j, i) = val(:, j, i) / val(nDim - 1, j, i)
        end do
    end do

end subroutine evalSurfaceNURBS

subroutine derivEvalSurfaceNURBS(u, v, uKnotVec, vKnotVec, uDegree, vDegree, Pw, nCtlu, nCtlv, nDim, order, skl)
    use precision
    implicit none

    ! Input
    integer, intent(in) :: uDegree, vDegree, nCtlu, nCtlv, nDim, order
    real(kind=realType), intent(in) :: u, v
    real(kind=realType), intent(in) :: uKnotVec(0:nCtlu+uDegree), vKnotVec(0:nCtlv+vDegree)
    real(kind=realType), intent(in) :: Pw(0:nDim-1, 0:nCtlv-1, 0:nCtlu-1)

    ! Output
    real(kind=realType), intent(out) :: skl(0:nDim-2, 0:order, 0:order)

    ! Working
    real(kind=realType) :: sklw(0:nDim-1, 0:order, 0:order)
    real(kind=realType) :: temp(0:nDim-2), temp2(0:nDim-2)
    integer :: k, l, j, i, binCoeff

    ! First we need to call `derivEvalSurface` to evaluate the derivative of the weighted control points.
    ! This will get A(u) and w(u) and store them in `sklw`
    call derivEvalSurface(u, v, uKnotVec, vKnotVec, uDegree, vDegree, Pw, nCtlu, nCtlv, nDim, order, sklw)

    ! Use Algorithm A4.4 from The NURBS Book to compute the true derivatives `skl`
    do k = 0, order
        do l = 0, order
            temp = sklw(:, l, k)
            do j = 1, l+1
                call bin(l, j, binCoeff)
                temp = temp - (binCoeff * sklw(nDim-1, j, 0) * skl(:, l-j, k))
            end do
            do i = 1, k+1
                call bin(k, i, binCoeff)
                temp = temp - (binCoeff * sklw(nDim-1, 0, i) * skl(:, l, k-i))
                temp2(:) = 0.0
                do j = 1, l+1
                    call bin(l, j, binCoeff)
                    temp2 = temp2 + (binCoeff * sklw(nDim-1, j, i) * skl(:, l-j, k-i))
                end do
                call bin(k, i, binCoeff)
                temp = temp - (binCoeff * temp2)
            end do
            skl(:, l, k) = temp(0:nDim-2) / skl(nDim-1, 0, 0)
        end do
    end do
    
end subroutine derivEvalSurfaceNURBS
