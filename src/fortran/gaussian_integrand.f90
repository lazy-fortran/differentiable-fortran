module gaussian_integrand_m
    use, intrinsic :: iso_c_binding, only: c_double
    implicit none
    private

    public :: gaussian_integrand

contains

    function gaussian_integrand(x) result(value) bind(C, name="df_gaussian")
        real(c_double), value, intent(in) :: x
        real(c_double) :: value

        value = exp(-x*x)
    end function gaussian_integrand

end module gaussian_integrand_m
