
subroutine buildCurveCoeffMatrix(u, ud, knotVec, degree, nctl, n, nd, vals, row_ptr, col_ind)
    use precision
    implicit none
    ! Input
    integer, intent(in) :: degree, nctl, n, nd
    real(kind=realType), intent(in) :: knotVec(nctl + degree + 1), u(n), ud(nd)
    ! Output
    real(kind=realType), intent(inout) :: vals((n + nd) * (degree +1))
    integer, intent(inout) :: row_ptr(n + nd + 1)
    integer, intent(inout) :: col_ind((n + nd) * (degree + 1))
    real(kind=realType) :: basisu(degree + 1), basisud(degree + 1, degree + 1)
    integer :: i, j, counter, ileft, k

    k = degree + 1 ! Initialize the order
    counter = 1
    do i = 1, n ! Do the values first
        call findSpan(u(i), degree, knotVec, nctl, ileft)
        call basis(u(i), degree, knotVec, ileft, nctl, basisu)
        ! Convert to 1 based indexing
        ileft = ileft + 1

        row_ptr(i) = counter - 1
        do j = 1, k
            col_ind(counter) = ileft - k + j - 1
            vals(counter) = basisu(j)
            counter = counter + 1
        end do
    end do
    do i = 1, nd ! Do the derivatives next
        call findSpan(ud(i), degree, knotVec, nctl, ileft)
        call derivBasis(ud(i), degree, knotVec, ileft, nctl, 1, basisud)
        ! Convert to 1 based indexing
        ileft = ileft + 1

        row_ptr(i + n) = counter - 1
        do j = 1, k
            col_ind(counter) = ileft - k + j - 1
            vals(counter) = basisud(2, j)
            counter = counter + 1
        end do
    end do
    row_ptr(n + nd + 1) = counter - 1
end subroutine buildCurveCoeffMatrix

subroutine buildCurveConJac(Aval, ArowPtr, AcolInd, Bval, BrowPtr, BcolInd, Cval, CrowPtr, CcolInd, &
                      Am, An, Cm, Annz, Bnnz, Cnnz, Jval, JcolInd, JrowPtr)
    use precision
    implicit none
    ! Input
    integer, intent(in) :: Am, An, Cm, Annz, Bnnz, Cnnz
    real(kind=realType), intent(in) :: Aval(Annz), Bval(Bnnz), Cval(Cnnz)
    integer, intent(in) :: AcolInd(Annz), BcolInd(Bnnz), CcolInd(Cnnz)
    integer, intent(in) :: ArowPtr(Am + 1), BrowPtr(Am + 1), CrowPtr(Cm + 1)

    ! Output
    real(kind=realType), intent(out) :: Jval(Annz + Bnnz + Cnnz)
    integer, intent(out) :: JcolInd(Annz + Bnnz + Cnnz)
    integer, intent(out) :: JrowPtr(Am + Cm + 1)

    ! Local
    integer :: i, j, counter
    ! This functions assembes the following CSR matrix:
    ! J = [A    B]
    !     [C    0]

    ! Now assmeble the full jacobain
    counter = 1
    JrowPtr(1) = 0
    do i = 1, Am
        do j = 1, ArowPtr(i + 1) - ArowPtr(i)
            Jval(counter) = Aval(ArowPtr(i) + j)
            JcolInd(counter) = AcolInd(ArowPtr(i) + j)
            counter = counter + 1
        end do
        do j = 1, BrowPtr(i + 1) - BrowPtr(i)
            Jval(counter) = Bval(BrowPtr(i) + j)
            JcolInd(counter) = BcolInd(BrowPtr(i) + j) + An
            counter = counter + 1
        end do
        JrowPtr(i + 1) = counter - 1
    end do
    do i = 1, Cm
        do j = 1, CrowPtr(i + 1) - CrowPtr(i)
            Jval(counter) = Cval(CrowPtr(i) + j)
            JcolInd(counter) = CcolInd(CrowPtr(i) + j)
            counter = counter + 1
        end do
        JrowPtr(i + 1 + am) = counter - 1
    end do
end subroutine buildCurveConJac

subroutine polyLength(X, n, ndim, length)
    ! Compute the length of the spatial polygon
    use precision
    implicit none

    !Input
    integer, intent(in) :: n, ndim
    real(kind=realType), intent(in) :: X(ndim, n)

    ! Ouput
    real(kind=realType), intent(out) :: length

    ! Working
    integer :: i, idim
    real(kind=realType) :: dist

    length = 0.0
    do i = 1, n - 1
        dist = 0.0
        do idim = 1, ndim
            dist = dist + (X(idim, i) - X(idim, i + 1))**2
        end do
        length = length + sqrt(dist)
    end do

end subroutine polyLength

subroutine curveParamCorr(knotVec, degree, u, coef, nCtl, nDim, length, n, X)

    ! Do Hoschek parameter correction
    use precision
    implicit none
    ! Input/Output
    integer, intent(in) :: degree, nCtl, nDim, n
    real(kind=realType), intent(in) :: knotVec(nCtl + degree + 1)
    real(kind=realType), intent(inout) :: u(n)
    real(kind=realType), intent(in) :: coef(nDim, nCtl)
    real(kind=realType), intent(in) :: X(nDim, n)
    real(kind=realType), intent(in) :: length
    ! Working
    integer :: i, j, k, maxInnerIter
    real(kind=realType) :: D(nDim), D2(nDim)
    ! "deriv" has second dimension = 2 because derivEvalCurve returns the
    ! evaluation of the curve in the first column, and then the first derivative
    ! in the second column.
    real(kind=realType) :: val(nDim), deriv(nDim, 2)
    real(kind=realType) :: c, sTilde

    k = degree + 1 ! Set the order of the spline
    maxInnerIter = 10
    do i = 2, n - 1
        call evalCurve(u(i), knotVec, degree, coef, nCtl, nDim, 1, val)
        call derivEvalCurve(u(i), knotVec, degree, coef, 1, nCtl, nDim, deriv)

        D = X(:, i) - val

        ! Be sure to use the second column of deriv to get the first derivative
        c = dot_product(D, deriv(:, 2)) / NORM2(deriv(:, 2))

        inner_loop: do j = 1, maxInnerIter

            sTilde = u(i) + c * (knotVec(nCtl + k) - knotVec(1)) / length
            call evalCurve(sTilde, knotVec, degree, coef, nCtl, nDim, 1, val)
            D2 = X(:, i) - val
            if (NORM2(D) .ge. NORM2(D2)) then
                u(i) = sTilde
                exit inner_loop
            else
                c = c * 0.5
            end if
        end do inner_loop
    end do

end subroutine curveParamCorr

function computeRMSCurve(knotVec, degree, u, P, nCtl, nDim, n, X)
    ! Compute the rms
    use precision
    implicit none
    ! Input/Output
    integer, intent(in) :: degree, nCtl, nDim, n
    real(kind=realType), intent(in) :: knotVec(degree + nCtl + 1)
    real(kind=realType), intent(in) :: u(n)
    real(kind=realType), intent(in) :: P(nDim, nCtl)
    real(kind=realType), intent(in) :: X(nDim, n)
    real(kind=realType) :: computeRMSCurve

    ! Working
    integer :: i, idim
    real(kind=realType) :: val(nDim), D(nDim)

    computeRMSCurve = 0.0
    do i = 1, n
        call evalCurve(u(i), knotVec, degree, P, nCtl, nDim, 1, val)
        D = val - X(:, i)
        do idim = 1, nDim
            computeRMSCurve = computeRMSCurve + D(idim)**2
        end do
    end do
    computeRMSCurve = sqrt(computeRMSCurve / n)
end function computeRMSCurve
