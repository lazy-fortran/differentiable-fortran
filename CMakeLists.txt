cmake_minimum_required(VERSION 3.22)

project(differentiable_fortran LANGUAGES C Fortran)

include(CTest)
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_Fortran_STANDARD 2018)
set(CMAKE_Fortran_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_Fortran_MODULE_DIRECTORY "${PROJECT_BINARY_DIR}/modules")

add_library(differentiable_fortran SHARED
    src/fortran/heat_step_primal.f90
    src/fortran/heat_step_analytic.f90
    src/fortran/heat_model.f90
    src/fortran/gaussian_integrand.f90
)
set_target_properties(differentiable_fortran PROPERTIES
    OUTPUT_NAME differentiable_fortran
)

if(BUILD_TESTING)
    add_executable(test_analytic test/fortran/test_analytic.f90)
    target_link_libraries(test_analytic PRIVATE differentiable_fortran)
    add_test(NAME analytic COMMAND test_analytic)
endif()

install(TARGETS differentiable_fortran
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)
install(FILES contract/heat_step.h DESTINATION include)
