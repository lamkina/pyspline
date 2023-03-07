subroutine insertKnot(u, r, knotVec, degree, ctrlPts, nctl, ndim, knotVecNew, ctrlPtsNew, spanNew)

    !***DESCRIPTION
    !
    !     Written by Gaetan Kenway
    !
    !     Abstract insertKnot inserts a knot u into the curve, r times
    !     Adapted from "The NURBS Book" Algorithm 5.1
    !     Description of Arguments
    !     Input
    !     u       - Real, location of knot to insert
    !     r       - Integer, Insert r times
    !     t       - Real,Knot vector. Length nctl+k
    !     k       - Integer,order of B-spline
    !     coef    - Real,Array of B-spline coefficients  Size (ndim,nctl)
    !     nctl    - Integer,Number of control points
    !     ndim    - Integer, dimension of curve

    !     Ouput
    !     t_new    - Real, vector of lenght(nctl+k+1+r)
    !     coef_new - Real, Array of new cofficients size(ndim,nctl+r)
    !     ileft    - Integer of position of knot insertion

    use precision
    implicit none

    ! Input
    integer, intent(inout) :: r
    integer, intent(in) :: degree, nctl, ndim
    real(kind=realType), intent(in) :: u
    real(kind=realType), intent(in) :: knotVec(nctl + degree + 1)
    real(kind=realType), intent(in) :: ctrlPts(ndim, nctl)

    ! Output
    real(kind=realType), intent(out) :: knotVecNew(nctl + degree + 1 + r)
    real(kind=realType), intent(out) :: ctrlPtsNew(ndim, nctl + r)
    integer, intent(out) :: spanNew

    ! Working
    integer :: mult, i, j, L
    real(kind=realType) :: alpha, temp(ndim, degree)

    ! Find the span and multiplicity of the new knot
    call findSpan(u, degree, knotVec, nctl, spanNew)
    call multiplicity(u, knotVec, nctl, degree, mult)

    ! We need to make sure that the requested multipliity r, plus
    ! the actual multiplicity of this know is less than (degree-1)
    if (mult + r + 1 > degree) then
        r = degree - 1 - mult
    end if

    ! If we *can't* insert the knot, we MUST copy t and coef to t_new
    ! and coef_new and return
    if (r == 0) then
        ctrlPtsNew(:, 1:nctl) = ctrlPts(:, 1:nctl)
        knotVecNew(1:nctl + degree + 1) = knotVec(1:nctl + degree + 1)
        return
    end if

    ! --- Load the new knot vector ---
    ! copy everything before the knot insertion point
    do i = 1, spanNew
        knotVecNew(i) = knotVec(i)
    end do

    ! Add the value of the new knot r times
    do i = 1, r
        knotVecNew(spanNew + i) = u
    end do

    ! Copy the rest of the knot vector from the knot insertion point to the end
    do i = spanNew + 1, nctl + degree + 1
        knotVecNew(i + r) = knotVec(i)
    end do

    ! --- Save unaltered control points ---
    do i = 1, spanNew - degree + 2
        ctrlPtsNew(:, i) = ctrlPts(:, i)
    end do

    do i = spanNew - mult, nctl + r
        ctrlPtsNew(:, i + r) = ctrlPts(:, i)
    end do

    do i = 0, degree - 1 - mult
        temp(:, i + 1) = ctrlPts(:, spanNew - (degree - 1) + i)
    end do

    ! Insert the knot r times
    do j = 1, r
        L = spanNew - degree + j
        do i = 1, degree - j - mult + 1
            alpha = (u - knotVec(L + i)) / (knotVec(i + spanNew + 1) - knotVec(L + i))
            temp(:, i) = alpha * temp(:, i + 1) + (1.0 - alpha) * temp(:, i)
        end do
        ctrlPtsNew(:, L) = temp(:, 1)
        ctrlPtsNew(:, spanNew + r - j - mult) = temp(:, degree - j - mult)
    end do

    do i = L + 1, spanNew - mult + 1
        ctrlPtsNew(:, i) = temp(:, i - L + 1)
    end do

    call findSpan(u, degree, knotVecNew, nctl + r, spanNew)
end subroutine insertKnot

