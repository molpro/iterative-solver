#ifndef LINEARALGEBRA_SRC_MOLPRO_LINALG_ARRAY_DISTRARRAYFILE_H
#define LINEARALGEBRA_SRC_MOLPRO_LINALG_ARRAY_DISTRARRAYFILE_H

#include <filesystem>
#include <fstream>
#include <iostream>

#include <molpro/linalg/array/DistrArrayDisk.h>

namespace molpro::linalg::array {

class ArrayFile;

/*!
 * @brief Distributed array storing the buffer on disk using temporary local files.
 *
 * On construction, file and corresponding stream object are being created and file is then closed (to be opened using
 * open_access()). On destruction, file is being deleted.
 *
 * @warning Only local operations will be currently supported, if RMA operations are requested, exception will be
 * thrown.
 *
 */
class DistrArrayFile : public DistrArrayDisk {
protected:
  std::unique_ptr<ArrayFile> m_local_array;           //! File array storing the local section
  static constexpr size_t m_default_block_size = 1e6; //! Default block size when buffering local section from disk

public:
  DistrArrayFile() = delete;
  DistrArrayFile(const DistrArrayFile &source);
  DistrArrayFile(DistrArrayFile &&source) noexcept;
  explicit DistrArrayFile(size_t dimension, MPI_Comm comm, size_t block_size = m_default_block_size,
                          const std::string &directory = ".");
  explicit DistrArrayFile(std::unique_ptr<Distribution> distribution, MPI_Comm comm,
                          size_t block_size = m_default_block_size, const std::string &directory = ".");
  explicit DistrArrayFile(const DistrArray &source);

  DistrArrayFile &operator=(const DistrArrayFile &source) = delete;
  DistrArrayFile &operator=(DistrArrayFile &&source) noexcept;

  friend void swap(DistrArrayFile &x, DistrArrayFile &y) noexcept;

  //! Flushes the buffer if file access is open
  ~DistrArrayFile() override;

  bool compatible(const DistrArrayFile &source) const;

  //! Dummy
  void erase() override {}
  //! @returns element at given index
  value_type at(index_type ind) const override;
  //! Writes value at a given index
  void set(index_type ind, value_type val) override;
  //! Reads requested values
  void get(index_type lo, index_type hi, value_type *buf) const override;
  //! @returns a vector of values read from file
  std::vector<value_type> get(index_type lo, index_type hi) const override;
  //! Writes requested values to file
  void put(index_type lo, index_type hi, const value_type *data) override;
  //! @warning below functions have to be implemented being pure virtuals, but will return exceptions
  //! taken we don't do collectives here
  void acc(index_type lo, index_type hi, const value_type *data) override;
  std::vector<value_type> gather(const std::vector<index_type> &indices) const override;
  void scatter(const std::vector<index_type> &indices, const std::vector<value_type> &data) override;
  void scatter_acc(std::vector<index_type> &indices, const std::vector<value_type> &data) override;
  std::vector<value_type> vec() const override;

  virtual void fill(value_type a);

  virtual void zero();
  virtual void scal(value_type a);
  virtual void add(const DistrArrayFile &y);
  virtual void add(const DistrArray &y);
  virtual void add(value_type a);
  virtual void sub(const DistrArrayFile &y);
  virtual void sub(const DistrArray &y);
  virtual void sub(value_type a);
  virtual void times(const DistrArrayFile &y);
  virtual void times(const DistrArray &y);
  virtual void times(const DistrArrayFile &y, const DistrArrayFile &z);
  virtual void times(const DistrArray &y, const DistrArray &z);
  virtual void axpy(value_type a, const DistrArrayFile &y);
  virtual void axpy(value_type a, const DistrArray &y);
  virtual void axpy(value_type a, const SparseArray &y);
  [[nodiscard]] virtual value_type dot(const DistrArrayFile &y) const;
  [[nodiscard]] virtual value_type dot(const DistrArray &y) const;
  [[nodiscard]] virtual value_type dot(const SparseArray &y) const;

  // TODO implement this. The logic is already in ArrayFile, it just needs to be moved out.
  //  [[nodiscard]] std::map<size_t, value_type> select_max_dot(size_t n, const DistrArrayFile &y) const;
  //  [[nodiscard]] std::map<size_t, value_type> select_max_dot(size_t n, const DistrArray &y) const;
  //  [[nodiscard]] std::map<size_t, value_type> select_max_dot(size_t n, const SparseArray &y) const;
};

} // namespace molpro::linalg::array

#endif // LINEARALGEBRA_SRC_MOLPRO_LINALG_ARRAY_DISTRARRAYFILE_H
