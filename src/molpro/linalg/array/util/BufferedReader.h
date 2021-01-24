#ifndef LINEARALGEBRA_SRC_MOLPRO_LINALG_ARRAY_BUFFEREDREADER_H
#define LINEARALGEBRA_SRC_MOLPRO_LINALG_ARRAY_BUFFEREDREADER_H
#include <molpro/linalg/array/util/BlockReader.h>

#include <list>
#include <memory>

namespace molpro::linalg::array::util {
/*!
 * @brief Reads array one block at a time using two buffers. When one block is read, the next one starts loading in a
 * separate thread.
 * @tparam Array
 */
template <class Array>
class BufferedReader {
public:
  explicit BufferedReader(BlockReader<Array> block_reader)
      : m_block_reader(std::move(block_reader)), m_reader(std::future<void>()) {
    for (auto i : {0, 1})
      m_buffers.emplace_back();
  }

  /*!
   * @brief Wait for buffered block to finish reading and return it. Starts reading next_index if it is a valid block
   * index.
   *
   * First call will initiate buffering of the block. The returned vector in that case is undefined.
   *
   * Example
   * -------
   * @code{cpp}
   * auto a = SomeArray{...};
   * auto buffered_reader = BufferedReader{BlockReader{a, ...}};
   * // First call to read to initiate IO, returns an empty buffer
   * buffered_reader.read(0);
   * // Second call to initiate next IO and return the buffer from previous call
   * auto& first_block = buffered_reader.read(1);
   * // Passing an invalid block index returns current buffer and does not initate IO
   * auto& second_block = buffered_reader.read(-1);
   * auto& second_block_again = buffered_reader.read(-1);
   * auto& second_block_again_and_read_third_block = buffered_reader.read(2);
   * @endcode
   *
   * The main use case is iterating over all blocks
   * @code{cpp}
   * auto a = SomeArray{...};
   * auto block_reader = BlockReader{a, ...};
   * auto buffered_reader = BufferedReader{block_reader};
   * auto block_reader.read(0);
   * for (auto i = 0; i < block_reader.n_blocks(); ++i){
   *    auto& block = block_reader.read(i+1);
   *    // operate on the block while the next one is loading
   *    // last iteration returns the block without IO
   * }
   * @endcode
   *
   * @param next_index index of the next block to start reading
   * @return
   */
  Block& read(int next_index) {
    bool index_is_in_valid_range = next_index >= 0 && next_index < m_block_reader.n_blocks();
    if (index_is_in_valid_range) {
      if (!m_reader) {
        m_reader = m_block_reader.get(next_index, m_buffers.back());
      } else {
        m_reader.wait();
        m_buffers.splice(m_buffers.end(), m_buffers, m_buffers.begin());
        m_reader = m_block_reader.get(next_index, m_buffers.back());
      }
    } else {
      if (m_reader) {
        m_reader.wait();
        m_buffers.splice(m_buffers.end(), m_buffers, m_buffers.begin());
      }
    }
    return m_buffers.front();
  }

  const BlockReader<Array>& block_reader() const { return m_block_reader; }
  BlockReader<Array>& block_reader() { return m_block_reader; }

private:
  BlockReader<Array> m_block_reader;
  std::list<Block> m_buffers;
  util::Task<void> m_reader;
};

/*!
 * @brief Apply a unary operation to the full buffered range one block at a time using separate thread for I/O
 * @param x block reader
 * @param f unary operation
 * @param put_x whether to call x.put() after the operation.
 */
template <class A, class Func>
void buffered_unary_operation(BlockReader<A>& x, Func&& f, bool put_x = true) {
  auto reader = BufferedReader<A>(x);
  reader.read(0);
  for (size_t i = 0; i < x.n_blocks(); ++i) {
    auto& block = reader.read(i + 1);
    f(block);
    if (put_x)
      x.put(i, block.buffer);
  }
}

//! Apply a unary operator without modifying the array
template <class A, class Func>
void buffered_unary_operation(const BlockReader<A>& x, Func&& f) {
  buffered_unary_operation(const_cast<BlockReader<A>&>(x), std::forward<Func&&>(f), false);
}

/*!
 * @brief Apply a binary operation to the full buffered range one block at a time using separate thread for I/O
 * @param x block reader providing access to array x
 * @param y block reader providing access to array y
 * @param f binary operation
 * @param put_x whether to call put on array x
 * @param put_y whether to call put on array y
 */
template <class A, class B, class Func>
void buffered_binary_operation(BlockReader<A>& x, BlockReader<B>& y, Func&& f, bool put_x = true, bool put_y = true) {
  if (!x.distribution().compatible(y.distribution()))
    throw std::runtime_error("attempting to operate on two arrays with different blocking structure");
  auto reader_x = BufferedReader<A>(x);
  auto reader_y = BufferedReader<B>(y);
  reader_x.read(0);
  reader_y.read(0);
  for (size_t i = 0; i < x.n_blocks(); ++i) {
    auto& block_x = reader_x.read(i + 1);
    auto& block_y = reader_y.read(i + 1);
    f(block_x, block_y);
    auto t_put_x = Task<void>(std::future<void>());
    auto t_put_y = Task<void>(std::future<void>());
    if (put_x)
      t_put_x = x.put(i, block_x.buffer);
    if (put_y)
      t_put_y = y.put(i, block_y.buffer);
  }
}

//! Apply a binary operation that modifies y
template <class A, class B, class Func>
void buffered_binary_operation(const BlockReader<A>& x, BlockReader<B>& y, Func&& f) {
  buffered_binary_operation(const_cast<BlockReader<A>&>(x), y, std::forward<Func&&>(f), false, true);
}

//! Apply a binary operation that modifies x
template <class A, class B, class Func>
void buffered_binary_operation(BlockReader<A>& x, const BlockReader<B>& y, Func&& f) {
  buffered_binary_operation(x, const_cast<BlockReader<A>&>(y), std::forward<Func&&>(f), true, false);
}

//! Apply a binary operation that does not modify the arrays
template <class A, class B, class Func>
void buffered_binary_operation(const BlockReader<A>& x, const BlockReader<B>& y, Func&& f) {
  buffered_binary_operation(const_cast<BlockReader<A>&>(x), const_cast<BlockReader<A>&>(y), std::forward<Func&&>(f),
                            false, false);
}
} // namespace molpro::linalg::array::util
#endif // LINEARALGEBRA_SRC_MOLPRO_LINALG_ARRAY_BUFFEREDREADER_H
