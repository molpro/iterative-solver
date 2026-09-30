LibraryManager_Add(${PROJECT_NAME} NAMESPACE molpro)

target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_20)

# MPI has to be called in the same scope as LibraryManager_Install, if pkg-config file is needed
# Otherwise MPI::MPI_CXX target is not available.
if (MPI)
    find_package(MPI COMPONENTS CXX)
    # Some MPI compiler wrappers (e.g. an MPICH built with LTO) leak -flto/-ffat-lto-objects into
    # the compile flags they report. Mixed with a non-LTO final link, this silently drops
    # weak/inline symbols and breaks linking, so strip them from the cache variable that
    # FindMPI(CXX) uses to populate MPI::MPI_CXX -- profiler and utilities call find_package(MPI)
    # again themselves, so the fix has to live in the cache, not just the imported target.
    if (MPI_CXX_COMPILE_OPTIONS)
        set(_mpi_cxx_compile_options "${MPI_CXX_COMPILE_OPTIONS}")
        list(FILTER _mpi_cxx_compile_options EXCLUDE REGEX "^-(flto|ffat-lto-objects)")
        set(MPI_CXX_COMPILE_OPTIONS "${_mpi_cxx_compile_options}" CACHE STRING "" FORCE)
        unset(_mpi_cxx_compile_options)
        if (TARGET MPI::MPI_CXX)
            set_target_properties(MPI::MPI_CXX PROPERTIES INTERFACE_COMPILE_OPTIONS "${MPI_CXX_COMPILE_OPTIONS}")
        endif ()
    endif ()
endif ()

if (PPIDD)
    if (FORTRAN)
        set(PPIDD_FORTRAN ON)
    else ()
        set(PPIDD_FORTRAN OFF)
    endif ()
    DependencyManager_Populate(ppidd)
endif ()

# Its source files are conventionally in directories named by C++ namespace
add_subdirectory(molpro)

LibraryManager_Install(${PROJECT_NAME} PKG_CONFIG)
LibraryManager_Export(${PROJECT_NAME})
