subroutine df_enzyme_heat_step(n, alpha, dt, dx, x, y) bind(C)
    use, intrinsic :: iso_c_binding, only: c_double, c_int
    implicit none

    integer(c_int), value, intent(in) :: n
    real(c_double), value, intent(in) :: alpha, dt, dx
    real(c_double), intent(in) :: x(n)
    real(c_double), intent(out) :: y(n)
    real(c_double) :: ratio
    integer :: i

    ratio = alpha*dt/(dx*dx)
    y(1) = x(1)
    do i = 2, n - 1
        y(i) = x(i) + ratio*(x(i - 1) - 2.0_c_double*x(i) + x(i + 1))
    end do
    y(n) = x(n)
end subroutine df_enzyme_heat_step

function df_enzyme_heat_objective(n, alpha, dt, dx, x, y_bar) &
        result(value) bind(C)
    use, intrinsic :: iso_c_binding, only: c_double, c_int
    implicit none

    integer(c_int), value, intent(in) :: n
    real(c_double), value, intent(in) :: alpha, dt, dx
    real(c_double), intent(in) :: x(n), y_bar(n)
    real(c_double) :: ratio, value
    integer :: i

    ratio = alpha*dt/(dx*dx)
    value = y_bar(1)*x(1) + y_bar(n)*x(n)
    do i = 2, n - 1
        value = value + y_bar(i)*(x(i) &
            + ratio*(x(i - 1) - 2.0_c_double*x(i) + x(i + 1)))
    end do
end function df_enzyme_heat_objective
