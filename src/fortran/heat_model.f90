module heat_model_m
    use, intrinsic :: iso_fortran_env, only: dp => real64
    use heat_step_analytic_m, only: heat_step_jvp
    use heat_step_primal_m, only: heat_step
    implicit none
    private

    type, public :: heat_model_t
        private
        real(dp) :: alpha = 1.0_dp
        real(dp) :: dx = 1.0_dp
    contains
        procedure, public :: get_alpha
        procedure, public :: get_dx
        procedure, public :: set_alpha
        procedure, public :: set_dx
        procedure, public :: step
        procedure, public :: step_jvp
    end type heat_model_t

    public :: new_heat_model

contains

    pure function new_heat_model(alpha, dx) result(model)
        real(dp), intent(in) :: alpha, dx
        type(heat_model_t) :: model

        model%alpha = alpha
        model%dx = dx
    end function new_heat_model

    pure function get_alpha(self) result(alpha)
        class(heat_model_t), intent(in) :: self
        real(dp) :: alpha

        alpha = self%alpha
    end function get_alpha

    pure function get_dx(self) result(dx)
        class(heat_model_t), intent(in) :: self
        real(dp) :: dx

        dx = self%dx
    end function get_dx

    pure subroutine set_alpha(self, alpha)
        class(heat_model_t), intent(inout) :: self
        real(dp), intent(in) :: alpha

        self%alpha = alpha
    end subroutine set_alpha

    pure subroutine set_dx(self, dx)
        class(heat_model_t), intent(inout) :: self
        real(dp), intent(in) :: dx

        self%dx = dx
    end subroutine set_dx

    pure subroutine step(self, x, dt, y)
        class(heat_model_t), intent(in) :: self
        real(dp), contiguous, intent(in) :: x(:)
        real(dp), intent(in) :: dt
        real(dp), intent(out) :: y(size(x))

        call heat_step(size(x), self%alpha, dt, self%dx, x, y)
    end subroutine step

    pure subroutine step_jvp(self, x, dt, x_dot, dt_dot, y, y_dot)
        class(heat_model_t), intent(in) :: self
        real(dp), contiguous, intent(in) :: x(:)
        real(dp), intent(in) :: dt, x_dot(size(x)), dt_dot
        real(dp), intent(out) :: y(size(x)), y_dot(size(x))

        call heat_step_jvp(size(x), self%alpha, dt, self%dx, x, 0.0_dp, &
            dt_dot, 0.0_dp, x_dot, y, y_dot)
    end subroutine step_jvp

end module heat_model_m
