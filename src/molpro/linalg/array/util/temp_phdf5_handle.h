#ifndef LINEARALGEBRA_SRC_MOLPRO_LINALG_ARRAY_UTIL_TEMP_PHDF5_HANDLE_H
#define LINEARALGEBRA_SRC_MOLPRO_LINALG_ARRAY_UTIL_TEMP_PHDF5_HANDLE_H
#include <molpro/linalg/array/PHDF5Handle.h>
#include <string>

namespace molpro::linalg::array::util {
/*!
 * @brief Returns a handle to a temporary file that will be erased on its destruction
 *
 * The handle has no group assigned to it.
 *
 * @param base_name base name of the file
 * @param comm mpi communicator
 */
PHDF5Handle temp_phdf5_handle(const std::string &base_name, MPI_Comm comm);

/*!
 * @brief Returns copy of the handle with a temporary group assigned and opened
 * @note When creating a new group in a parallel hdf5 file that group might not be registered until the file is closed.
 * @param handle handle to modify
 * @param base_name base name of the temporary group. It can be absolute path, or relative to group name of handle
 * @param comm new mpi communicator. If null than the communicator from handle will be used.
 */
PHDF5Handle temp_phdf5_handle_group(const PHDF5Handle &handle, const std::string &base_name,
                                    MPI_Comm comm = MPI_COMM_NULL);

} // namespace molpro::linalg::array::util

#endif // LINEARALGEBRA_SRC_MOLPRO_LINALG_ARRAY_UTIL_TEMP_PHDF5_HANDLE_H
