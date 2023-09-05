subroutine newton(f, fp, x0, x, lb, ub, alpha0, rho, c, maxIter, maxIterLS, tol, nPts, printLevel)
    use precision
    implicit none

    ! Input
    integer, intent(in) :: nPts
    real(kind=realType), intent(in) :: x0(nPts), lb(nPts), ub(nPts)
    external :: f, fp
    integer, intent(in) :: maxIter, printLevel, maxIterLS
    real(kind=realType), intent(in) :: tol, alpha0, rho, c

    ! Output
    real(kind=realType), intent(out) :: x(nPts)

    ! Working
    real(kind=realType) :: deltaX(nPts)
    real(kind=realType) :: resid(nPts), jac(nPts, nPts)
    real(kind=realType) :: L(nPts, nPts), U(nPts, nPts), P(nPts, nPts)
    real(kind=realType) :: phi0, phi, lsCheck, alpha, alphaOld
    integer :: iter, lsIter

    ! Initial values
    alphaOld = 0.0
    x = x0

    do iter = 1, maxIter
        call f(x, resid)
        call fp(x, jac)
        phi0 = norm2(resid)
        alpha = alpha0

        if (printLevel > 0) then
            write (*, '(A, I3, A, ES15.8)') "NL Newton: ", iter, " | Residual Norm: ", phi0
        end if

        ! Convergence check
        if (phi0 < tol) then
            if (printLevel > 0) then
                write (*, '(A, I3, A)') "NL Newton converged in ", iter, " iterations."
            end if
            return
        end if

        ! Solve the linear system for the newton step
        if (nPts > 1) then
            resid = -1 * resid
            ! call solve_linear_system(jac, resid, deltaX, nPts)
            call ludecomp(jac, L, U, P, nPts)
            call lusolve(L, U, P, deltaX, resid, nPts)
            resid = -1 * resid

        else
            deltaX(1) = -resid(1) / jac(1, 1)
        end if

        ! Update the states
        x = x + alpha * deltaX

        ! Enforce the bounds
        deltaX = enforceBounds()

        ! Linesearch
        call f(x, resid)
        phi = norm2(resid)
        lsIter = 1

        if (printLevel > 0) then
            write (*, '(A, I3, A, F5.2)') "    LS Armijo: ", lsIter, ": Alpha: ", alpha
        end if

        lsCheck = phi0 - (c * alpha * phi0)
        if (phi <= lsCheck) then
            cycle
        end if

        do lsIter = 1, maxIterLS
            alphaOld = alpha
            alpha = alpha * rho
            x = x + deltaX * (alpha - alphaOld)

            call f(x, resid)
            phi = norm2(resid)

            if (printLevel > 0) then
                write (*, '(A, I3, A, F5.2)') "    LS Armijo: ", lsIter, ": Alpha: ", alpha
            end if

            lsCheck = phi0 - (c * alpha * phi0)
            if (phi <= lsCheck) then
                cycle
            end if
        end do
    end do

    if (printLevel > 0) then
        write (*, '(A, I3, A)') "NL Newton did not converge in ", maxIter, " iterations."
    end if

contains

    function enforceBounds() result(deltaXnew)
        use precision
        implicit none

        real(kind=realType) :: changeLower(nPts), changeUpper(nPts), change(nPts), deltaXnew(nPts)
        integer :: i

        ! Calculate the lower and upper change
        do i = 1, nPts
            changeLower(i) = max(x(i), lb(i)) - x(i)
            changeUpper(i) = min(x(i), ub(i)) - x(i)
        end do

        ! Calculate the total change
        change = changeLower + changeUpper

        ! Update the states and the step length
        x = x + change
        deltaXnew = deltaX + change / alpha
    end function enforceBounds
end subroutine newton

