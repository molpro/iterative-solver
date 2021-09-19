cmake_minimum_required(VERSION 3.19)
project(Experiment1 CXX)

set(CMAKE_CXX_STANDARD 17)

add_executable(LinearAlgebra-examples-experiment1 experiment1.cpp)

if (NOT TARGET molpro::LinearAlgebra)
    include(FetchContent)
    FetchContent_Declare(LinearAlgebra GIT_REPOSITORY https://gitlab.com/molpro/LinearAlgebra)
    set(DEPENDENCYMANAGER_FETCHCONTENT ON)
    set(MPI OFF)
    set(FORTRAN OFF)
    FetchContent_MakeAvailable(LinearAlgebra)
endif ()

target_link_libraries(LinearAlgebra-examples-experiment1 PRIVATE molpro::LinearAlgebra)
