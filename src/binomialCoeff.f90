! Computes the binomial coefficient C(n,k) using the formula
! C(n,k) = n! / (k! * (n-k)!)
! where n! is the factorial of n.
! Inputs:
!   n: integer, the total number of items.
!   k: integer, the number of items to choose.
! Outputs:
!   binomial_coefficient: integer, the computed value of C(n,k).
subroutine bin(n, k, binCoeff)
    implicit none
    
    ! Input
    integer, intent(in) :: n, k

    ! Output
    integer, intent(out) :: binCoeff

    ! Working
    integer :: i, num, denom
  
    num = 1
    denom = 1
  
    ! compute the numerator and denominator
    do i = 1, k
      num = num * (n - i + 1)
      denom = denom * i
    end do
  
    ! compute the final result
    binCoeff = num / denom
  
  end subroutine bin
  