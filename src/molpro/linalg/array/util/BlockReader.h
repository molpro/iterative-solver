#ifndef LINEARALGEBRA_SRC_MOLPRO_LINALG_ARRAY_BLOCKREADER_H
#define LINEARALGEBRA_SRC_MOLPRO_LINALG_ARRAY_BLOCKREADER_H
#include <vector>

#include <molpro/linalg/array/Span.h>
#include <molpro/linalg/array/util/Distribution.h>
#include <molpro/linalg/array/util/Task.h>

namespace molpro::linalg::array::util {
/*!
 * @brief Reads a section of array in regular sized blocks using a separate thread.
 *
 * Responsibilities
 * - split section of array into blocks
 * - implement get/put in terms of block index and using a separate thread
 */
template <class Array>
class BlockReader {
public:
  /*!
   * @param a array to read which implements get and put methods
   * @param start start index of the section
   * @param end past-the-end index of the section
   * @param block_size nominal size of the block (the remainder may be spread over blocks, so actual size could be +1)
   */
  BlockReader(Array& a, size_t start, size_t end, size_t block_size)
      : m_array(a), m_block_size(block_size), m_start(start), m_end(end) {
    if (start > end)
      throw std::runtime_error("invalid array range");
    if (m_block_size < 1)
      throw std::runtime_error("block size must be >= 1");
    auto section_length = end - start;
    auto nblocks = section_length / m_block_size;
    m_distribution = make_distribution_spread_remainder<size_t>(section_length, nblocks);
  }

  //! Get a block
  Task<void> get(size_t block_index, std::vector<double>& buffer) {
    check_block_index(block_index);
    auto [beg, end] = m_distribution.range(block_index);
    auto size = end - beg;
    buffer.resize(size);
    auto getter = [&, beg = beg, end = end]() {
      auto data = Span<double>(&buffer[0], size);
      m_array.get(m_start + beg, m_start + end, data);
    };
    return Task<void>::create(getter);
  }

  //! Get a block into the array
  Task<void> put(size_t block_index, const std::vector<double>& buffer) {
    check_block_index(block_index);
    auto [beg, end] = m_distribution.range(block_index);
    auto size = end - beg;
    if (size > buffer.size())
      throw std::runtime_error("buffer is too small");
    auto putter = [&, beg = beg, end = end]() {
      auto data = Span<double>(const_cast<double*>(&buffer[0]), size);
      m_array.put(m_start + beg, m_start + end, data);
    };
    return Task<void>::create(putter);
  }

  const Distribution<size_t>& distribution() const { return m_distribution; }

  size_t n_blocks() const { return m_distribution.size(); }

private:
  void check_block_index(size_t block_index) {
    if (block_index >= n_blocks())
      throw std::runtime_error("block index >= number of blocks");
  }

  size_t m_start = 0;
  size_t m_end = 0;
  size_t m_block_size = 1;
  Distribution<size_t> m_distribution; //!< distribution of blocks
  Array& m_array;
};

} // namespace molpro::linalg::array::util

#endif // LINEARALGEBRA_SRC_MOLPRO_LINALG_ARRAY_BLOCKREADER_H
