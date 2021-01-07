#ifndef LINEARALGEBRA_SRC_MOLPRO_LINALG_ARRAY_BUFFEREDREADER_H
#define LINEARALGEBRA_SRC_MOLPRO_LINALG_ARRAY_BUFFEREDREADER_H
#include <molpro/linalg/array/BlockReader.h>

#include <list>

namespace molpro::linalg::array {
/*!
 * @brief Reads array one block at a time using two buffers. When one block is read, the next one starts loading in a
 * separate thread.
 * @tparam Array
 */
template <class Array>
class BufferedReader {
public:
  /*!
   * @brief Wait for buffered block to finish reading and return it. Starts reading next_index if it is a valid block
   * index.
   *
   * First call will initiate buffering of the block. The returned vector in that case is undefined.
   *
   * Example
   * -------
   *
   * @param next_index index of the next block to start reading
   * @return
   */
  std::vector<double>& read(int next_index) {
    bool index_is_in_valid_range = next_index >= 0 && next_index < m_block_reader.n_blocks();
    if (index_is_in_valid_range) {
      if (!m_reader) {
        m_reader = std::make_unique<molpro::linalg::array::util::Task<void>>(
            m_block_reader.get(next_index, m_buffers.front()));
      } else {
        m_reader->wait();
        m_buffers.splice(m_buffers.end(), m_buffers, m_buffers.begin());
        m_reader =
            std::make_unique<molpro::linalg::array::util::Task<void>>(m_block_reader.get(next_index, m_buffers.back()));
      }
    } else {
      if (m_reader) {
        m_reader->wait();
        reset();
      }
    }
    return m_buffers.front();
  }

  void reset() { m_reader.reset(); }

private:
  BlockReader<Array> m_block_reader;
  std::list<std::vector<double>> m_buffers;
  std::unique_ptr<util::Task<void>> m_reader;
};

//! Apply a unary operator to the full buffered range one block at a time
template <class A, class Func>
double buffered_unary_operation(const BlockReader<A>& x, Func&& f) {
  auto reader = BufferedReader<A>(x);
  reader.read(0);
  for (size_t i = 0; i < x.n_blocks(); ++i) {
    auto& buffer = reader.read(i + 1);
    f(buffer);
  }
}

//! Apply a binary operator to the full buffered range one block at a time
template <class A, class B, class Func>
double buffered_binary_operation(const BlockReader<A>& x, const BlockReader<B>& y, Func&& f) {
  // Assume they are compatible
  auto n = x.n_chunks();
  auto buffer_x = std::vector<double>(n);
  auto buffer_y = std::vector<double>(n);
}
} // namespace molpro::linalg::array
#endif // LINEARALGEBRA_SRC_MOLPRO_LINALG_ARRAY_BUFFEREDREADER_H
