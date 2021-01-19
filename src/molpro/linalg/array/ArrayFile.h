#ifndef LINEARALGEBRA_SRC_MOLPRO_LINALG_ARRAY_ARRAYFILE_H
#define LINEARALGEBRA_SRC_MOLPRO_LINALG_ARRAY_ARRAYFILE_H
#include <filesystem>
#include <fstream>
#include <map>
#include <string>
#include <string_view>
#include <vector>

#include <molpro/linalg/array/Span.h>
#include <molpro/linalg/array/util/BlockReader.h>
namespace molpro::linalg::array {

/*!
 * @brief Array stored in a temporary file using a binary format.
 *
 * Restriction to a temporary file can be relaxed later.
 *
 * Key responsibilities:
 * - create a file and make sure it is erased on destruction
 * - implement put and get I/O operations
 *
 */
class ArrayFile {
public:
  using index_type = unsigned long int;
  using value_type = double;
  static constexpr size_t default_block_size = 1e6;
  ArrayFile() = delete;
  ArrayFile(size_t dimension, size_t block_size = default_block_size);
  ArrayFile(std::string_view directory, size_t dimension, size_t block_size = default_block_size);
  ~ArrayFile();

  // FIXME remove until copy is implemented
  ArrayFile(const ArrayFile& other) = delete;
  // FIXME Delete until we decide whether resizing is permissible
  ArrayFile& operator=(const ArrayFile& other) = delete;

  ArrayFile(ArrayFile&& other) noexcept = default;
  ArrayFile& operator=(ArrayFile&& other) noexcept = default;

  //! number of elements in the array
  size_t size() const;
  std::string directory() const { return m_dir.string(); }

  //! @returns true if other array's size and blocking distribution are the same as this
  bool compatible(const ArrayFile& other);

  //! Reads requested values
  void get(index_type lo, index_type hi, Span<value_type> buf) const;

  //! @returns a vector of values read from file
  std::vector<value_type> get(index_type lo, index_type hi) const;

  //! Writes requested values to file
  void put(index_type lo, index_type hi, const Span<value_type>& data);

  void fill(value_type value);

protected:
  size_t m_dim = 0; //!< number of elements in the array
  std::filesystem::path m_dir;
  mutable std::fstream m_file;
  util::BlockReader<ArrayFile> m_block_reader;
  //! creates a file and opens it. @returns file stream
  std::fstream make_file();
};

} // namespace molpro::linalg::array

#endif // LINEARALGEBRA_SRC_MOLPRO_LINALG_ARRAY_ARRAYFILE_H
