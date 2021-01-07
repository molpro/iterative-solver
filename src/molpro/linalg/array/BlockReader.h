#ifndef LINEARALGEBRA_SRC_MOLPRO_LINALG_ARRAY_BLOCKREADER_H
#define LINEARALGEBRA_SRC_MOLPRO_LINALG_ARRAY_BLOCKREADER_H
#include <vector>

#include <molpro/linalg/array/Span.h>
#include <molpro/linalg/array/util.h>

namespace molpro::linalg::array {

/*!
 * @brief Reads a container in blocks
 */
template <class Array>
class BlockReader {
public:
  util::Task<void> get(size_t block_index, std::vector<double>& buffer) {
    size_t beg, end;
    std::tie(beg, end) = range(block_index);
    auto size = end - beg;
    buffer.resize(size);
    auto getter = [&]() { m_array.get(beg, end, Span<double>(buffer.data(), size)); };
    return util::Task<void>::create(getter);
  }

  util::Task<void> put(size_t block_index, const std::vector<double>& buffer) {
    size_t beg, end;
    std::tie(beg, end) = range(block_index);
    auto size = end - beg;
    if (size > buffer.size())
      throw std::runtime_error("buffer is too small");
    auto getter = [&]() { m_array.get(beg, end, Span<double>(const_cast<double*>(buffer.data()), size)); };
    return util::Task<void>::create(getter);
  }

  std::pair<size_t, size_t> range(size_t block_index) const {
    if (block_index >= m_nblocks)
      throw std::runtime_error("block_index out of range");
    auto beg = m_start + block_index * m_block_size;
    auto end = beg + m_block_size;
    end = end > m_end ? m_end : end;
    return {beg, end};
  }

  size_t n_blocks() const { return m_nblocks; }
  size_t max_block_size() const { return m_block_size; }

private:
  size_t m_start;
  size_t m_end;
  size_t m_nblocks;
  size_t m_block_size;
  Array& m_array;
};

} // namespace molpro::linalg::array

#endif // LINEARALGEBRA_SRC_MOLPRO_LINALG_ARRAY_BLOCKREADER_H
