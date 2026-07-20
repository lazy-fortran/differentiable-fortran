module heat_step_analytic_m
    use, intrinsic :: iso_c_binding, only: c_double, c_int
    use, intrinsic :: iso_fortran_env, only: dp => real64
    use heat_step_primal_m, only: heat_step
    implicit none
    private

    public :: heat_step_jvp, heat_step_jvp_c, heat_step_vjp, heat_step_vjp_c

contains

    pure subroutine heat_step_jvp(n, alpha, dt, dx, x, alpha_dot, dt_dot, &
            dx_dot, x_dot, y, y_dot)
        integer, intent(in) :: n
        real(dp), intent(in) :: alpha, dt, dx
        real(dp), intent(in) :: x(n)
        real(dp), intent(in) :: alpha_dot, dt_dot, dx_dot
        real(dp), intent(in) :: x_dot(n)
        real(dp), intent(out) :: y(n), y_dot(n)
        real(dp) :: laplacian, ratio, ratio_dot
        integer :: i

        ratio = alpha*dt/(dx*dx)
        ratio_dot = dt*alpha_dot/(dx*dx) + alpha*dt_dot/(dx*dx) &
            - 2.0_dp*alpha*dt*dx_dot/(dx*dx*dx)

        call heat_step(n, alpha, dt, dx, x, y)
        y_dot(1) = x_dot(1)
        do i = 2, n - 1
            laplacian = x(i - 1) - 2.0_dp*x(i) + x(i + 1)
            y_dot(i) = x_dot(i) + ratio_dot*laplacian &
                + ratio*(x_dot(i - 1) - 2.0_dp*x_dot(i) + x_dot(i + 1))
        end do
        y_dot(n) = x_dot(n)
    end subroutine heat_step_jvp

    pure subroutine heat_step_vjp(n, alpha, dt, dx, x, y_bar, x_bar, &
            alpha_bar, dt_bar, dx_bar)
        integer, intent(in) :: n
        real(dp), intent(in) :: alpha, dt, dx
        real(dp), intent(in) :: x(n), y_bar(n)
        real(dp), intent(out) :: x_bar(n)
        real(dp), intent(out) :: alpha_bar, dt_bar, dx_bar
        real(dp) :: laplacian_sum, ratio
        integer :: i

        ratio = alpha*dt/(dx*dx)
        x_bar = 0.0_dp
        x_bar(1) = y_bar(1)
        x_bar(n) = y_bar(n)
        laplacian_sum = 0.0_dp

        do i = 2, n - 1
            x_bar(i - 1) = x_bar(i - 1) + ratio*y_bar(i)
            x_bar(i) = x_bar(i) + (1.0_dp - 2.0_dp*ratio)*y_bar(i)
            x_bar(i + 1) = x_bar(i + 1) + ratio*y_bar(i)
            laplacian_sum = laplacian_sum &
                + y_bar(i)*(x(i - 1) - 2.0_dp*x(i) + x(i + 1))
        end do

        alpha_bar = laplacian_sum*dt/(dx*dx)
        dt_bar = laplacian_sum*alpha/(dx*dx)
        dx_bar = -2.0_dp*laplacian_sum*alpha*dt/(dx*dx*dx)
    end subroutine heat_step_vjp

    subroutine heat_step_jvp_c(n, alpha, dt, dx, x, alpha_dot, dt_dot, &
            dx_dot, x_dot, y, y_dot) &
            bind(C, name="df_heat_step_jvp")
        integer(c_int), value, intent(in) :: n
        real(c_double), value, intent(in) :: alpha, dt, dx
        real(c_double), intent(in) :: x(n)
        real(c_double), value, intent(in) :: alpha_dot, dt_dot, dx_dot
        real(c_double), intent(in) :: x_dot(n)
        real(c_double), intent(out) :: y(n), y_dot(n)

        call heat_step_jvp(n, alpha, dt, dx, x, alpha_dot, dt_dot, dx_dot, &
            x_dot, y, y_dot)
    end subroutine heat_step_jvp_c

    subroutine heat_step_vjp_c(n, alpha, dt, dx, x, y_bar, x_bar, &
            alpha_bar, dt_bar, dx_bar) &
            bind(C, name="df_heat_step_vjp")
        integer(c_int), value, intent(in) :: n
        real(c_double), value, intent(in) :: alpha, dt, dx
        real(c_double), intent(in) :: x(n), y_bar(n)
        real(c_double), intent(out) :: x_bar(n)
        real(c_double), intent(out) :: alpha_bar, dt_bar, dx_bar

        call heat_step_vjp(n, alpha, dt, dx, x, y_bar, x_bar, alpha_bar, &
            dt_bar, dx_bar)
    end subroutine heat_step_vjp_c

end module heat_step_analytic_m
