#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <molpro/linalg/array/util/BufferedReader.h>

using molpro::linalg::array::Span;
using molpro::linalg::array::util::BlockReader;
using molpro::linalg::array::util::BufferedReader;

namespace {

struct DummyGetPutArray {
  void get(size_t beg, size_t end, Span<double>& buffer) { get_called_with_arguments.emplace_back(beg, end); }
  void put(size_t beg, size_t end, const Span<double>& buffer) { throw std::runtime_error("put should not be called"); }
  std::vector<std::pair<size_t, size_t>> get_called_with_arguments = {};
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
