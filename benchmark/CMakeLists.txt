foreach (t benchmark)
    add_executable(${t}.exe ${t}.cpp)
    target_link_libraries(${t}.exe PUBLIC ${PROJECT_NAME} Profiler)
    if (MPI)
        #        target_link_libraries(${t}.exe PUBLIC MPI::MPI_CXX)
        # in principle this should be configured at runtime but it seems cmake/ctest support is not yet there to do it simply
        if (NOT DEFINED ENV{MPIPROCS})
            set(ENV{MPIPROCS} 1)
        endif ()
        message(DEBUG "add_test(${t} ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} $ENV{MPIPROCS} ${MPIEXEC_PREFLAGS} ${CMAKE_CURRENT_BINARY_DIR}/${t}.exe ${MPIEXEC_POSTFLAGS})")
        add_test(${t} ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} $ENV{MPIPROCS} ${MPIEXEC_PREFLAGS} "${CMAKE_CURRENT_BINARY_DIR}/${t}.exe" ${MPIEXEC_POSTFLAGS})
    else ()
        add_test(${t} ${t}.exe)
    endif ()
endforeach ()

if (MPI)
    foreach (t LinearEigensystemBenchmark-mpi)
        add_executable(${t}.exe ${t}.cpp)
        target_link_libraries(${t}.exe PUBLIC ${PROJECT_NAME} Profiler)
        if (NOT DEFINED ENV{MPIPROCS})
            set(ENV{MPIPROCS} 1)
        endif ()
        message(DEBUG "add_test(${t} ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} $ENV{MPIPROCS} ${MPIEXEC_PREFLAGS} ${CMAKE_CURRENT_BINARY_DIR}/${t}.exe ${MPIEXEC_POSTFLAGS})")
        add_test(${t} ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} $ENV{MPIPROCS} ${MPIEXEC_PREFLAGS} "${CMAKE_CURRENT_BINARY_DIR}/${t}.exe" ${MPIEXEC_POSTFLAGS})
    endforeach ()
endif ()

#
#include(FetchContent)
#FetchContent_Declare(
#        multiprecision
#        GIT_REPOSITORY https://github.com/boostorg/multiprecision.git
#        GIT_TAG boost-1.74.0
#        GIT_SHALLOW ON
#)
#FetchContent_MakeAvailable(multiprecision)
find_package(Boost)

if (Boost_FOUND)
    add_executable(quad_precision_benchmark.exe quad_precision_benchmark.cpp)
    #    target_include_directories(quad_precision_benchmark.exe PUBLIC ${CMAKE_BINARY_DIR}/_deps/multiprecision-src/include)
    target_link_libraries(quad_precision_benchmark.exe PUBLIC Profiler Boost::headers)
endif ()
