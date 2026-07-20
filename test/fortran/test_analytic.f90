program test_analytic
    use, intrinsic :: iso_fortran_env, only: dp => real64
    use heat_step_analytic_m, only: heat_step_jvp, heat_step_vjp
    use heat_step_primal_m, only: heat_step
    implicit none

    integer, parameter :: n = 8
    real(dp), parameter :: tolerance = 2.0e-8_dp
    real(dp) :: alpha, alpha_bar, alpha_dot, dt, dt_bar, dt_dot
    real(dp) :: dx, dx_bar, dx_dot, epsilon
    real(dp) :: adjoint_left, adjoint_right
    real(dp) :: x(n), x_bar(n), x_dot(n), x_minus(n), x_plus(n)
    real(dp) :: expected_jvp(n), y(n), y_bar(n), y_dot(n), y_minus(n), y_plus(n)
    integer :: i

    alpha = 0.7_dp
    dt = 0.02_dp
    dx = 0.2_dp
    alpha_dot = 0.13_dp
    dt_dot = -0.01_dp
    dx_dot = 0.03_dp
    epsilon = 1.0e-7_dp

    do i = 1, n
        x(i) = sin(0.3_dp*real(i - 1, dp)) + 0.1_dp*real(i - 1, dp)
        x_dot(i) = cos(0.2_dp*real(i - 1, dp))
        y_bar(i) = (-1.0_dp)**i/(1.0_dp + real(i, dp))
    end do

    call heat_step_jvp(n, alpha, dt, dx, x, alpha_dot, dt_dot, dx_dot, &
        x_dot, y, y_dot)

    x_plus = x + epsilon*x_dot
    x_minus = x - epsilon*x_dot
    call heat_step(n, alpha + epsilon*alpha_dot, dt + epsilon*dt_dot, &
        dx + epsilon*dx_dot, x_plus, y_plus)
    call heat_step(n, alpha - epsilon*alpha_dot, dt - epsilon*dt_dot, &
        dx - epsilon*dx_dot, x_minus, y_minus)
    expected_jvp = (y_plus - y_minus)/(2.0_dp*epsilon)
    call assert_vector_close("JVP finite difference", y_dot, expected_jvp, tolerance)

    call heat_step_vjp(n, alpha, dt, dx, x, y_bar, x_bar, alpha_bar, &
        dt_bar, dx_bar)
    adjoint_left = dot_product(y_dot, y_bar)
    adjoint_right = dot_product(x_dot, x_bar) + alpha_dot*alpha_bar &
        + dt_dot*dt_bar + dx_dot*dx_bar
    call assert_scalar_close("adjoint identity", adjoint_left, adjoint_right, &
        1.0e-12_dp)

contains

    subroutine assert_scalar_close(label, actual, expected, allowed)
        character(*), intent(in) :: label
        real(dp), intent(in) :: actual, expected, allowed

        if (abs(actual - expected) > allowed) then
            write (*, '(a,2es24.15)') trim(label)//": ", actual, expected
            error stop 1
        end if
    end subroutine assert_scalar_close

    subroutine assert_vector_close(label, actual, expected, allowed)
        character(*), intent(in) :: label
        real(dp), intent(in) :: actual(:), expected(:), allowed

        call assert_scalar_close(label, maxval(abs(actual - expected)), 0.0_dp, &
            allowed)
    end subroutine assert_vector_close

end program test_analytic
