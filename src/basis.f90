!> Description:
!>   Given a knot vector, a degree, and a parameter value u, compute the
!>   basis functions for the corresponding B-spline curve segment. The basis
!>   functions are computed using the Cox-de Boor recursion formula.
!>
!> Inputs:
!>   degree    - Integer, degree of the B-spline basis functions
!>   span      - Integer, index of the knot span containing the parameter value u
!>   nctl      - Integer, number of control points in the curve
!>   u         - Real, parameter value
!>   knotvec   - Real, knot vector (array of size nctl + degree + 1)
!>
!> Outputs:
!>   B         - Real, array containing the basis functions (of size degree + 1)
!>
subroutine basis(u, degree, knotVec, span, nCtl, B)
    ! NOTE: We use 0-based indexing to be consistent with algorithms
    ! in The NURBS Book.
    use precision
    implicit none

    ! Inputs
    integer, intent(in) :: degree, span, nCtl
    real(kind=realType), intent(in) :: u
    real(kind=realType), intent(in) :: knotVec(0:nCtl + degree)

    ! Outputs
    real(kind=realType), intent(out) :: B(0:degree)

    ! Working
    integer :: j, r
    real(kind=realType) :: left(0:degree), right(0:degree), saved, temp

    B(0) = 1.0

    do j = 1, degree
        left(j) = u - knotVec(span + 1 - j)
        right(j) = knotVec(span + j) - u
        saved = 0.0
        do r = 0, j - 1
            temp = B(r) / (right(r + 1) + left(j - r))
            B(r) = saved + right(r + 1) * temp
            saved = left(j - r) * temp
        end do
        B(j) = saved
    end do
end subroutine basis

subroutine derivBasis(u, degree, knotVec, span, nCtl, order, Bd)
    ! NOTE: We use 0-based indexing to be consistent with algorithms
    ! in The NURBS Book.
    use precision
    implicit none

    ! Inputs
    integer, intent(in) :: degree, nCtl, span, order
    real(kind=realType), intent(in) :: u
    real(kind=realType), intent(in) :: knotVec(0:nCtl + degree)

    ! Outputs
    real(kind=realType), intent(out) :: Bd(0:min(degree, order), 0:degree)

    ! Working
    integer :: r, j, k, s1, s2, rk, pk, j1, j2
    real(kind=realType) :: saved, temp, d
    real(kind=realType) :: a(0:1, 0:degree)
    real(kind=realType) :: left(0:degree), right(0:degree)
    real(kind=realType) :: ndu(0:degree, 0:degree)

    ndu(0, 0) = 1.0

    do j = 1, degree
        left(j) = u - knotVec(span + 1 - j)
        right(j) = knotVec(span + j) - u
        saved = 0.0

        do r = 0, j - 1
            ! Lower triangle
            ndu(j, r) = right(r + 1) + left(j - r)
            temp = ndu(r, j - 1) / ndu(j, r)

            ! Upper triangle
            ndu(r, j) = saved + right(r + 1) * temp
            saved = left(j - r) * temp
        end do

        ndu(j, j) = saved
    end do

    ! Load the basis functions
    do j = 0, degree
        Bd(0, j) = ndu(j, degree)
    end do

    ! This section computes the derivatives using Eq. 2.9 from The NURBS book
    do r = 0, degree ! Loop over the function index
        ! Indexing for alternating rows in the array
        s1 = 0
        s2 = 1
        a(0, 0) = 1.0

        ! Loop to compute the kth derivative
        do k = 1, order
            d = 0.0
            rk = r - k
            pk = degree - k
            if (r >= k) then
                a(s2, 0) = a(s1, 0) / ndu(pk + 1, rk)
                d = a(s2, 0) * ndu(rk, pk)
            end if
            if (rk >= -1) then
                j1 = 1
            else
                j1 = -rk
            end if

            if (r - 1 <= pk) then
                j2 = k - 1
            else
                j2 = degree - r
            end if

            do j = j1, j2
                a(s2, j) = (a(s1, j) - a(s1, j - 1)) / ndu(pk + 1, rk + j)
                d = d + a(s2, j) * ndu(rk + j, pk)
            end do

            if (r <= pk) then
                a(s2, k) = -a(s1, k - 1) / ndu(pk + 1, r)
                d = d + a(s2, k) * ndu(r, pk)
            end if

            Bd(k, r) = d

            ! Switch rows
            j = s1
            s1 = s2
            s2 = j
        end do
    end do

    ! Multiply through by the correct factors
    ! Eq. 2.9 from The NURBS Book
    r = degree
    do k = 1, order
        do j = 0, degree
            Bd(k, j) = Bd(k, j) * r
            r = r * (degree - k)
        end do
    end do

end subroutine derivBasis
