LibraryManager_Append(${PROJECT_NAME}
        SOURCES temp_file.cpp iterable_lingalg.cpp select_max_dot.cpp
        PUBLIC_HEADER Distribution.h select_max_dot.h temp_file.h TempHandle.h
        BlockReader.h BufferedReader.h Task.h iterable_lingalg.h)

if (LINEARALGEBRA_ARRAY_HDF5)
    LibraryManager_Append(${PROJECT_NAME}
            SOURCES temp_hdf5_handle.cpp
            PUBLIC_HEADER temp_hdf5_handle.h)
endif ()

if (MPI)
    LibraryManager_Append(${PROJECT_NAME}
            SOURCES DistrFlags.cpp LockMPI3.cpp
            PUBLIC_HEADER DistrFlags.h gather_all.h LockMPI3.h ScopeLock.h)
    if (LINEARALGEBRA_ARRAY_HDF5)
        LibraryManager_Append(${PROJECT_NAME}
                SOURCES temp_phdf5_handle.cpp
                PUBLIC_HEADER temp_phdf5_handle.h)
    endif ()
endif ()
