subroutine optimizer(f, df, x0, x, lb, ub, alpha0, rho, wolfe, maxIter, maxIterLS, tol, printLevel, nx)
    ! Description:
    !   Simple gradient-based optimization algorithm with a backtracking line search.
    !   The line search uses bounds enforcement to satisfy the design variable bounds.
    !
    !   Description of Arguments
    !   Inputs:
    !     f         - external, The objective function
    !     df        - external, The gradient of the objective function
    !     x0        - Real, array size(nx), The initial design variables
    !     lb        - Real, array size(nx), The lower bounds on the design variables
    !     ub        - Real, array size(nx), The upper bounds on the design variables
    !     alpha0    - Real, The initial step length
    !     rho       - Real, Line search backtracking factor
    !     wolfe     - Real, The sufficient decrease factor
    !     maxIter   - Integer, The maximum number of iterations
    !     maxIterLS - Integer, The maximum number of line search iterations
    !     tol       - Real, The convergence tolerance
    !     printLevel- Integer, The print level
    !     nx        - Integer, The number of design variables
    !
    !   Outputs:
    !     x         - Real, array size(nx), The optimized design variables
    use precision
    implicit none

    ! Inputs
    integer, intent(in) :: nx, maxIter, maxIterLS
    real(kind=realType), dimension(nx) :: x0, lb, ub
    external :: f, df
    real(kind=realType), intent(in) :: alpha0, rho, tol, wolfe
    integer, intent(in) :: printLevel

    ! Outputs
    real(kind=realType), dimension(nx), intent(out) :: x

    ! Local variables
    real(kind=realType), dimension(nx) :: gradient, update
    real(kind=realType), dimension(nx, nx) :: hessian, L, U, P
    real(kind=realType) :: phi0, phi, alpha, alphaOld, dphi0, obj
    integer :: iter, lsIter

    ! Initialize values
    x = x0

    iterationLoop: do iter = 1, maxIter
        ! Initialize line search values
        alpha = alpha0
        alphaOld = 0.0

        ! Evaluate objective function and gradient
        call f(x, obj)
        call df(x, gradient, hessian)

        phi0 = obj ! Copy the objective function value into phi0

        ! Check for convergence using an infinity norm
        if (printLevel > 0) write (*, '(a, i3, a, e12.4, a, 4f5.2)') 'Iteration: ', &
            iter, '| Tol: ', maxval(abs(gradient)), '| u:', x
        if (maxval(abs(gradient)) < tol) exit iterationLoop

        ! Compute the search direction
        if (nx == 1) then
            update = gradient(1) / hessian(1, 1)
        else
            call ludecomp(hessian, L, U, P, nx)
            call lusolve(L, U, P, update, gradient, nx)
        end if

        update = -update

        if (nx == 1) then
            dphi0 = update(1) * gradient(1)
        else
            dphi0 = dot_product(update, gradient)
        end if

        ! Check if the search direction is a descent direction
        if (dphi0 >= 0.0) then
            if (nx == 1) then
                update(1) = -gradient(1) / abs(gradient(1))
                dphi0 = update(1) * gradient(1)
            else
                update = -gradient / norm2(gradient) ! Replace with steepest descent direction
                dphi0 = dot_product(update, gradient)
            end if
        end if

        ! Update the states
        x = x + alpha * update

        ! Enforce the bounds on the design variables
        update = enforceBounds()

        ! Evaluate objective function and gradient
        call f(x, obj)
        phi = obj ! Copy the objective function value into phi

        ! Check the sufficient decrease condition
        ! If the condition is satisfied then we can skip the line search
        ! and iterate again.
        if (phi < phi0 + dphi0 * wolfe * alpha) cycle iterationLoop

        ! Perform line search
        lsLoop: do lsIter = 1, maxIterLS
            alphaOld = alpha
            alpha = alpha * rho
            x = x + update * (alpha - alphaOld) ! Update the design variables
            call f(x, obj)
            phi = obj ! Copy the objective function value into phi
            ! Check the sufficient decrease condition
            if (phi < phi0 + dphi0 * wolfe * alpha) exit lsLoop
        end do lsLoop
    end do iterationLoop

contains

    function enforceBounds() result(updateNew)
        implicit none

        ! Local variables
        real(kind=realType) :: changeLower(nx), changeUpper(nx), change(nx), updateNew(nx)
        integer :: j

        ! Calculate the lower and upper change
        do j = 1, nx
            changeLower(j) = max(x(j), lb(j)) - x(j)
            changeUpper(j) = min(x(j), ub(j)) - x(j)
        end do

        ! Calculate the total change
        change = changeLower + changeUpper

        ! Update the states and the step length
        x = x + change
        updateNew = updateNew + change / alpha
    end function enforceBounds

end subroutine optimizer
