subroutine brent(f, a, b, tol, maxIter, printLevel, root, status)
    use precision
    implicit none

    ! Input
    real(kind=realType), intent(in) :: a, b, tol
    integer, intent(in) :: maxIter, printLevel
    external :: f

    ! Output
    real(kind=realType), intent(out) :: root
    integer, intent(out) :: status

    ! Working
    real(kind=realType) :: fa, fb, c, fc, s, fs, d, bb, aa, swapTemp
    integer :: iter
    logical :: mflag

    call f(a, fa)
    call f(b, fb)

    if (fa * fb > 0.0) then
        status = -1
        return
    end if

    aa = a
    bb = b

    ! Swap if |f(a)| < |f(b)|
    if (abs(fa) < abs(fb)) then
        aa = b
        bb = a

        swapTemp = fa
        fa = fb
        fb = swapTemp
    end if

    c = aa
    fc = fa
    s = 0.0
    fs = 0.0
    d = 0.0
    mflag = .true.

    iter = 0
    do while (iter < maxIter)
        if (fa /= fc .and. fb /= fc) then
            s = (aa*fb*fc / ((fa-fb)*(fa-fc))) + (bb*fa*fc / ((fb-fa)*(fb-fc))) + (c*fa*fb / ((fc-fa)*(fc-fb)))
        else
            s = (fa * b - fb * a) / (fa - fb)
        end if

        if (((s - (3 * (aa + bb) / 4)) * (s - bb) >= 0) &
            .or. (mflag .and. abs(s - bb) >= abs(bb - c) / 2) &
            .or. (.not. mflag .and. abs(s - bb) >= abs(c - d) / 2) &
            .or. (mflag .and. abs(bb - c) < tol) &
            .or. (.not. mflag .and. abs(c - d) < tol)) then
            s = (aa + bb) / 2.0
            mflag = .true.
        else
            mflag = .false.
        end if
        call f(s, fs)
        d = c
        c = bb
        fc = fb

        if (fa * fs < 0.0) then
            bb = s
            fb = fs
        else
            aa = s
            fa = fs
        end if

        ! Swap if |f(a)| < |f(b)|
        if (abs(fa) < abs(fb)) then
            swapTemp = aa
            aa = bb
            bb = swapTemp

            swapTemp = fa
            fa = fb
            fb = swapTemp
        end if

        if (printLevel > 0) write (*, '(a, i3, a, e12.4, a, f5.2, a, f5.2, a, e12.4)') &
            'Iteration:', iter, '| tol:', abs(fb), '| u:', bb, "| b-a:", abs(bb - aa)

        if (fb == 0.0 .or. fs == 0.0 .or. abs(bb - aa) < tol) then
            status = 0
            root = bb
            return
        end if

        iter = iter + 1
    end do

    if (abs(fb) > tol) then
        root = bb
        status = -2
    end if
end subroutine brent
