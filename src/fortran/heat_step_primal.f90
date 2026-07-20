module heat_step_primal_m
    use, intrinsic :: iso_c_binding, only: c_double, c_int
    use, intrinsic :: iso_fortran_env, only: dp => real64
    implicit none
    private

    public :: heat_point_c, heat_step, heat_step_c

contains

    pure function heat_point_c(left, center, right, ratio) result(value) &
            bind(C, name="df_heat_point")
        real(c_double), value, intent(in) :: left, center, right, ratio
        real(c_double) :: value

        value = center + ratio*(left - 2.0_c_double*center + right)
    end function heat_point_c

    pure subroutine heat_step(n, alpha, dt, dx, x, y)
        integer, intent(in) :: n
        real(dp), intent(in) :: alpha, dt, dx
        real(dp), intent(in) :: x(n)
        real(dp), intent(out) :: y(n)
        real(dp) :: ratio
        integer :: i

        ratio = alpha*dt/(dx*dx)
        y(1) = x(1)
        do i = 2, n - 1
            y(i) = x(i) + ratio*(x(i - 1) - 2.0_dp*x(i) + x(i + 1))
        end do
        y(n) = x(n)
    end subroutine heat_step

    subroutine heat_step_c(n, alpha, dt, dx, x, y) bind(C, name="df_heat_step")
        integer(c_int), value, intent(in) :: n
        real(c_double), value, intent(in) :: alpha, dt, dx
        real(c_double), intent(in) :: x(n)
        real(c_double), intent(out) :: y(n)

        call heat_step(n, alpha, dt, dx, x, y)
    end subroutine heat_step_c

end module heat_step_primal_m
