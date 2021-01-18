#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <mutex>

#include <molpro/linalg/array/util/BufferedReader.h>

using molpro::linalg::array::Span;
using molpro::linalg::array::util::BlockReader;
using molpro::linalg::array::util::buffered_unary_operation;
using molpro::linalg::array::util::BufferedReader;
using molpro::linalg::array::util::Task;
using ::testing::DoubleEq;
using ::testing::Each;

namespace {

struct DummyGetPutArray {
  void get(size_t beg, size_t end, Span<double>& buffer) { get_called_with_arguments.emplace_back(beg, end); }
  void put(size_t beg, size_t end, const Span<double>& buffer) { throw std::runtime_error("put should not be called"); }
  std::vector<std::pair<size_t, size_t>> get_called_with_arguments = {};
};

struct Array {
  Array(size_t size) : m_buffer(size, 0){};

  void get(size_t beg, size_t end, Span<double>& buffer) {
    m_mutex.lock();
    for (size_t i = 0; i < end - beg; ++i)
      buffer[i] = m_buffer[beg + i];
    m_mutex.unlock();
  }

  void put(size_t beg, size_t end, const Span<double>& buffer) {
    m_mutex.lock();
    for (size_t i = 0; i < end - beg; ++i)
      m_buffer[beg + i] = buffer[i];
    m_mutex.unlock();
  }

  std::vector<double> m_buffer;
  std::mutex m_mutex;
};

struct ArrayNoPut {
  ArrayNoPut(size_t size) : m_array(size){};

  void get(size_t beg, size_t end, Span<double>& buffer) { m_array.get(beg, end, buffer); }

  void put(size_t beg, size_t end, const Span<double>& buffer) { throw std::runtime_error("Put must not be called"); }

  Array m_array;
};
} // namespace

class BufferedReaderF : public ::testing::Test {
public:
  DummyGetPutArray a{};
  size_t start{0};
  size_t end{17};
  size_t block_size{3};
  BlockReader<DummyGetPutArray> block_reader{a, start, end, block_size};
};

TEST_F(BufferedReaderF, constructor) { auto buf_reader = BufferedReader{block_reader}; }

TEST_F(BufferedReaderF, read_first_call) {
  auto buf_reader = BufferedReader{block_reader};
  buf_reader.read(0);
  buf_reader.read(-1);
  ASSERT_EQ(a.get_called_with_arguments.size(), 1);
}

TEST_F(BufferedReaderF, read_loop) {
  auto buf_reader = BufferedReader{block_reader};
  auto& distribution = buf_reader.block_reader().distribution();
  auto ref_get_arguments = std::vector<std::pair<size_t, size_t>>{};
  buf_reader.read(0);
  for (int i = 0; i < distribution.size(); ++i) {
    auto& block = buf_reader.read(i + 1);
    auto [beg, end] = distribution.range(i);
    ASSERT_EQ(block.size(), end - beg) << " iter = " << i;
    ref_get_arguments.emplace_back(beg, end);
  }
  ASSERT_EQ(a.get_called_with_arguments, ref_get_arguments);
}

TEST(buffered_unary_operation, fill) {
  const double alpha = 3.0;
  const size_t size = 20;
  const size_t block_size = 3;
  auto array = Array(size);
  auto block_reader = BlockReader(array, 0, size, block_size);
  auto fill = [alpha](std::vector<double>& x) {
    for (auto& el : x)
      el = alpha;
  };
  buffered_unary_operation(block_reader, fill, true);
  ASSERT_THAT(array.m_buffer, Each(DoubleEq(alpha)));
}

TEST(buffered_unary_operation, equal) {
  const size_t size = 20;
  const size_t block_size = 3;
  auto array = ArrayNoPut(size);
  auto block_reader = BlockReader(array, 0, size, block_size);
  bool equal_to_one = true;
  auto eq = [&equal_to_one](std::vector<double>& x) {
    for (auto& el : x)
      equal_to_one &= el == 1.;
  };
  buffered_unary_operation(block_reader, eq, false);
  ASSERT_FALSE(equal_to_one);
}
