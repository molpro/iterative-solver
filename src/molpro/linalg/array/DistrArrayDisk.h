#ifndef LINEARALGEBRA_SRC_MOLPRO_LINALG_ARRAY_DISTRARRAYDISK_H
#define LINEARALGEBRA_SRC_MOLPRO_LINALG_ARRAY_DISTRARRAYDISK_H

#include <future>
#include <molpro/Profiler.h>
#include <molpro/linalg/array/DistrArray.h>
#include <molpro/linalg/array/Span.h>

namespace molpro::linalg::array {
/*!
 * @brief Distributed array located primarily on disk
 *
 * This class stores the full array on disk and implements RMA and more efficient linear algebra operations.
 *
 * RMA operations read/write directly to disk. Care must be taken that they do not overlap with local buffer
 * modifications.
 *
 * LocalBuffer reads the whole local section into memory. This might be prohivitively expensive for large arrays, so
 * care must be taken.
 *
 * BufferManager reads the local section in chunks using a separate thread for I/O. This is more memory efficient
 * and allows overlap of communication and computation. The walk through the local section is via iterators.
 *
 * IO can be done in a separate thread using util::Task.
 *
 * @code{.cpp}
 * #include <molpro/linalg/array/util.h>
 * auto da = DistrArrayDisk{...};
 * da.put(lo,hi, data); // Normal put same as in the base class is guaranteed to finish
 * auto t = util::Task::create([&](){da.put(lo, hi, data)}); // does I/O in a new thread
 * // do something time consuming
 * t.wait(); // wait for the thread to finish the I/O
 * @endcode
 *
 */
class DistrArrayDisk : public DistrArray {
public:
  using disk_array = void; //!< a compile time tag that this is a distributed disk array
protected:
  bool m_allocated = false;                     //!< Flags that the memory view buffer has been allocated
  std::unique_ptr<Distribution> m_distribution; //!< describes distribution of array among processes
  size_t m_buffer_size = 8192;                  //!< buffer size for paged access via BufferManager
  using DistrArray::DistrArray;

  DistrArrayDisk(std::unique_ptr<Distribution> distr, MPI_Comm commun);
  DistrArrayDisk();
  DistrArrayDisk(const DistrArray& source);
  DistrArrayDisk(DistrArrayDisk&& source) noexcept;
  ~DistrArrayDisk() override;

public:
  //! Erase the array from disk.
  virtual void erase() = 0;
  [[nodiscard]] const Distribution& distribution() const override;
  [[nodiscard]] value_type dot(const DistrArrayDisk& y) const;
  [[nodiscard]] value_type dot(const DistrArray& y) const override;
  [[nodiscard]] value_type dot(const SparseArray& y) const override;
  void set_buffer_size(size_t buffer_size) { m_buffer_size = buffer_size; }
  void copy(const DistrArray& y) override;

protected:
  //! Reads the whole local buffer from disk into memory. By default the buffer is written to disk on destruction,
  //! unless do_dump is false.
  class LocalBufferDisk : public DistrArray::LocalBuffer {
  public:
    explicit LocalBufferDisk(DistrArrayDisk& source);
    explicit LocalBufferDisk(DistrArrayDisk& source, const span::Span<value_type>& buffer);
    ~LocalBufferDisk() override;
    //! If true, than buffer is dumped to file on destruction.
    bool do_dump = true;

  protected:
    std::vector<value_type> m_snapshot_buffer; //!< when external buffer is not provided
    DistrArrayDisk& m_source;                  //!< keep a handle on source to dump data to disk
  };

public:
  [[nodiscard]] std::unique_ptr<LocalBuffer> local_buffer() override;
  [[nodiscard]] std::unique_ptr<const LocalBuffer> local_buffer() const override;
  //! Access local section, reading it into the provided buffer
  [[nodiscard]] std::unique_ptr<LocalBuffer> local_buffer(const span::Span<value_type>& buffer);
  //! Read-only access to the local section, reading it into the provided buffer
  [[nodiscard]] std::unique_ptr<const LocalBuffer> local_buffer(const span::Span<value_type>& buffer) const;

public:
};

double dot(const DistrArrayDisk& x, const DistrArrayDisk& y);
double dot(const DistrArrayDisk& x, const DistrArray& y);
double dot(const DistrArray& x, const DistrArrayDisk& y);

} // namespace molpro::linalg::array

#endif // LINEARALGEBRA_SRC_MOLPRO_LINALG_ARRAY_DISTRARRAYDISK_H
