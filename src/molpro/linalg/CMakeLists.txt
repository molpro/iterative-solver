# Extra checking for HDF5 and global arrays will go here
LibraryManager_Append(${PROJECT_NAME}
        SOURCES itsolv/IterativeSolver-double.cpp itsolv/IterativeSolver-complex-double.cpp options.cpp
        PUBLIC_HEADER IterativeSolverC.h itsolv/helper.h itsolv/helper-implementation.h itsolv/ArrayHandlers.h itsolv/Statistics.h options.h
        )
if (FORTRAN)
    LibraryManager_Append(${PROJECT_NAME} SOURCES IterativeSolverF.F90 Iterative_Solver_Problem.F90 Iterative_Solver_PSpace.F90 Iterative_Solver_Matrix_Problem.F90)
endif ()

add_subdirectory(itsolv)

target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_17)
set_target_properties(${PROJECT_NAME} PROPERTIES CXX_EXTENSIONS OFF)

if (MPI)
    message(STATUS "Building iterative-solver on MPI")
    LibraryManager_Append(${PROJECT_NAME} SOURCES IterativeSolverCMPI.cpp)
    target_compile_definitions(${PROJECT_NAME} PUBLIC HAVE_MPI_H)
    target_link_libraries(${PROJECT_NAME} PUBLIC MPI::MPI_CXX)
else ()
    message(STATUS "Building serial iterative-solver")
endif ()

if (TARGET ppidd::ppidd)
    message(STATUS "Building iterative-solver on PPIDD library")
    target_link_libraries(${PROJECT_NAME} PUBLIC ppidd::ppidd)
    target_compile_definitions(${PROJECT_NAME} PUBLIC HAVE_PPIDD_H)
endif ()

include(DependencyManager)
DependencyManager_Populate(profiler)
DependencyManager_Populate(utilities)
target_link_libraries(${PROJECT_NAME} PUBLIC molpro::profiler molpro::utilities)

LibraryManager_LAPACK(${PROJECT_NAME} PUBLIC Intel10_64lp_seq Intel10_64lp)
if (MKL_TYPE STREQUAL "lp64")
    message(STATUS "Eigen will use MKL")
    target_compile_definitions(${PROJECT_NAME} PUBLIC EIGEN_USE_MKL_ALL)
    target_compile_definitions(${PROJECT_NAME} PUBLIC EIGEN_MKL_NO_DIRECT_CALL)
endif ()

if (NOT TARGET Eigen3::Eigen)
    set(EIGEN_MIN_VERSION 3.4.0)
    set(EIGEN_FETCH_TAG 3.4.0)
    message(VERBOSE "Eigen3::Eigen is not yet a target")

    # Note: no version arg to find_package — Eigen's ConfigVersion file uses
    # strict-major-version compatibility, so a request for 3.4.0 rejects an
    # installed 5.x. We accept anything Eigen3Config provides and validate
    # the minimum version ourselves below.
    find_package(Eigen3 QUIET)

    if ((Eigen3_FOUND OR EIGEN3_FOUND) AND Eigen3_VERSION VERSION_LESS ${EIGEN_MIN_VERSION})
        message(STATUS "System Eigen3 ${Eigen3_VERSION} is older than required ${EIGEN_MIN_VERSION}; ignoring")
        set(Eigen3_FOUND FALSE)
        set(EIGEN3_FOUND FALSE)
    endif ()

    if (Eigen3_FOUND OR EIGEN3_FOUND)
        message(STATUS "Eigen3 (${Eigen3_VERSION}) found on system: ${Eigen3_INCLUDE_DIRS}")
    else ()
        message(STATUS "Eigen3::Eigen not found on system, and will be downloaded")
        include(FetchContent)
        FetchContent_Declare(eigen3
                GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
                GIT_TAG ${EIGEN_FETCH_TAG}
                )
        FetchContent_MakeAvailable(eigen3)
    endif ()
    if (NOT TARGET Eigen3::Eigen)
        add_library(Eigen3::Eigen ALIAS eigen)
    endif ()
else ()
    message(STATUS "iterative-solver: Eigen3::Eigen target already defined")
endif ()
set(CMAKE_MODULE_PATH "${CMAKE_MODULE_PATH};${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake")
message("CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH}")

target_link_libraries(${PROJECT_NAME} PUBLIC Eigen3::Eigen)
target_compile_definitions(${PROJECT_NAME} PUBLIC -DEIGEN_MPL2_ONLY)
if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "PGI")
    target_compile_definitions(${PROJECT_NAME} PUBLIC -DEIGEN_DONT_VECTORIZE)
endif ()


include(LibraryManagerUtils)
print_target_properties(BLAS::BLAS DEBUG)
print_target_properties(LAPACK::LAPACK DEBUG)
print_target_properties(Eigen3::Eigen DEBUG)
print_target_properties(ga::ga DEBUG)
print_target_properties(ppidd::ppidd DEBUG)
print_target_properties(${PROJECT_NAME} DEBUG)

add_subdirectory(array)
