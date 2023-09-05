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
subroutine evalSurface(u, v, uKnotVec, vKnotVec, uDegree, vDegree, P, nCtlu, nCtlv, nDim, n, m, val)
    ! NOTE: We use 0-based indexing to be consistent with algorithms
    ! in The NURBS Book.
    use precision
    implicit none

    ! Input
    integer, intent(in) :: uDegree, vDegree, nCtlu, nCtlv, nDim, n, m
    real(kind=realType), intent(in) :: u(0:m - 1, 0:n - 1), v(0:m - 1, 0:n - 1)
    real(kind=realType), intent(in) :: uKnotVec(0:nCtlu + uDegree), vKnotVec(0:nCtlv + vDegree)
    real(kind=realType), intent(in) :: P(0:nDim - 1, 0:nCtlv - 1, 0:nCtlu - 1)

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

            do ii = 0, uDegree
                do jj = 0, vDegree
                    val(:, j, i) = val(:, j, i) + Bu(ii) * Bv(jj) * P(:, istartv + jj, istartu + ii)
                end do
            end do
        end do
    end do

end subroutine evalSurface

subroutine derivEvalSurface(u, v, uKnotVec, vKnotVec, uDegree, vDegree, P, order, nCtlu, nCtlv, nDim, skl)
    ! NOTE: We use 0-based indexing to be consistent with algorithms
    ! in The NURBS Book.
    use precision
    implicit none

    ! Input
    integer, intent(in) :: uDegree, vDegree, nCtlu, nCtlv, nDim, order
    real(kind=realType), intent(in) :: u, v
    real(kind=realType), intent(in) :: uKnotVec(0:nCtlu + uDegree), vKnotVec(0:nCtlv + vDegree)
    real(kind=realType), intent(in) :: P(0:nDim - 1, 0:nCtlv - 1, 0:nCtlu - 1)

    ! Output
    real(kind=realType), intent(out) :: skl(0:nDim - 1, 0:order, 0:order)

    ! Working
    integer :: istartu, istartv, ii, jj, i, j, du, dv, dd, k
    integer :: ileftu, ileftv
    real(kind=realType) :: Bdu(0:min(uDegree, order), 0:uDegree), Bdv(0:min(vDegree, order), 0:vDegree)
    real(kind=realType) :: temp(0:vDegree, 0:nDim)

    ! Initialize the derivatives to zeros
    skl(:, :, :) = 0.0

    ! Get the highest available derivative order
    ! (Can only be as big as the degree in each parameteric direction)
    du = min(uDegree, order)
    dv = min(vDegree, order)

    ! Evaluate the span and basis function derivatives in the u-direction
    call findSpan(u, uDegree, uKnotVec, nCtlu, ileftu)
    call derivBasis(u, uDegree, uKnotVec, ileftu, nCtlu, du, Bdu)
    istartu = ileftu - uDegree

    ! Evaluate the spand and basis derivatives in the v-direction
    call findSpan(v, vDegree, vKnotVec, nCtlv, ileftv)
    call derivBasis(v, vDegree, vKnotVec, ileftv, nCtlv, dv, Bdv)
    istartv = ileftv - vDegree

    do k = 0, du
        temp(:, :) = 0.0
        do i = 0, vDegree
            do j = 0, uDegree
                temp(i, :) = temp(i, :) + Bdu(k, j) * P(:, istartv + i, istartu + j)
            end do
        end do

        dd = min(order - k, dv)
        do ii = 0, dd
            do jj = 0, vDegree
                skl(:, ii, k) = skl(:, ii, k) + Bdv(ii, jj) * temp(jj, :)
            end do
        end do
    end do

end subroutine derivEvalSurface

subroutine evalSurfaceNormals(u, v, uKnotVec, vKnotVec, uDegree, vDegree, P, nCtlu, nCtlv, nDim, n, m, normals)
    use precision
    implicit none

    ! Input
    integer, intent(in) :: uDegree, vDegree, nCtlu, nCtlv, nDim, n, m
    real(kind=realType), intent(in) :: u(0:m - 1, 0:n - 1), v(0:m - 1, 0:n - 1)
    real(kind=realType), intent(in) :: uKnotVec(0:nCtlu + uDegree), vKnotVec(0:nCtlv + vDegree)
    real(kind=realType), intent(in) :: P(0:nDim - 1, 0:nCtlv - 1, 0:nCtlu - 1)

    ! Output
    real(kind=realType), intent(out) :: normals(0:nDim - 1, 0:m - 1, 0:n - 1)

    ! Working
    real(kind=realType) :: skl(0:nDim - 1, 0:1, 0:1)
    real(kind=realType) :: tmp(0:nDim - 1)
    integer :: i, j

    do j = 0, m - 1
        do i = 0, n - 1
            call derivEvalSurface(u(j, i), v(j, i), uKnotVec, vKnotVec, uDegree, vDegree, P, 1, nCtlu, nCtlv, nDim, skl)
            call cross(skl(:, 0, 1), skl(:, 1, 0), tmp)
            normals(:, j, i) = tmp(:) / norm2(tmp)
        end do
    end do

end subroutine evalSurfaceNormals
