#ifndef LINEARALGEBRA_SRC_MOLPRO_LINALG_ARRAY_SERIALDISKARRAY_H
#define LINEARALGEBRA_SRC_MOLPRO_LINALG_ARRAY_SERIALDISKARRAY_H

#include <fstream>
#include <iostream>
#include <filesystem>
#include <molpro/linalg/array/Span.h>

namespace molpro::linalg::array {

namespace fs = std::filesystem;
/*!
 * @brief Serial array storing the buffer on disk using temporary local file.
 *
 * On construction, file and corresponding stream object are being created and file is then closed (to be opened using
 * open_access()). On destruction, file is being deleted.
 *
 * @warning Only local operations will be currently supported, if RMA operations are requested, exception will be thrown.
 *
 */
class SerialDiskArray {
public:
  using serial_disk_array = void; //!< a compile time tag that this is a serial disk array
  using value_type = double;
  using index_type = unsigned long int;
  
protected:
  index_type m_dimension = 0;              //!< number of elements in the array
  fs::path m_dir = fs::current_path();
  mutable std::fstream m_file;
  //! creates a file, opens it and @returns m_file fstream
  std::fstream make_file();
  
public:
  SerialDiskArray();
  SerialDiskArray(const SerialDiskArray &source);
  SerialDiskArray(SerialDiskArray &&source) noexcept;
  explicit SerialDiskArray(const Span<value_type> &source);
  
  SerialDiskArray &operator=(const SerialDiskArray &source);
  SerialDiskArray &operator=(const Span<value_type> &source);
  
  ~SerialDiskArray();
  
  friend void swap(SerialDiskArray &x, SerialDiskArray &y) noexcept; //!TODO: is it needed?
  
  void copy(const SerialDiskArray &x);
  
  bool empty() const;
  
  //! @returns element at given index
  value_type at(index_type ind) const;
  //! Writes value at a given index
  void set(index_type ind, value_type val);
  //! Reads requested values into a buffer
  void get(index_type lo, index_type hi, value_type *buf) const;
  //! @returns a vector of values read from file
  std::vector<value_type> get(index_type lo, index_type hi) const;
  //! Writes requested values to file
  void put(index_type lo, index_type hi, const value_type *data);
  //! Accumulates requested values with data from buffer
  void acc(index_type lo, index_type hi, const value_type *data);
  //! @returns a vector of values with specified indices
  std::vector<value_type> gather(const std::vector<index_type> &indices) const;
  //! Writes values at requested indices from provided buffer
  void scatter(const std::vector<index_type> &indices, const std::vector<value_type> &data);
  //! Performs gather() for provided indices, accumulates them with data from buffer and then performs scatter()
  void scatter_acc(std::vector<index_type> &indices, const std::vector<value_type> &data);
 
  //!TODO: Should it have it's own dot(), scal() and axpy() functions, or should we deal with them at the Handler level?
  //!TODO: How about buffer windows? Again at (lazy) handler level?
};

} // namespace molpro::linalg::array

#endif // LINEARALGEBRA_SRC_MOLPRO_LINALG_ARRAY_SERIALDISKARRAY_H
