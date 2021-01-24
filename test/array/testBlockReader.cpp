#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <molpro/linalg/array/util/BlockReader.h>

using molpro::linalg::array::Span;
using molpro::linalg::array::util::Block;
using molpro::linalg::array::util::BlockReader;

namespace {
struct DummyArray {
  void get(size_t beg, size_t end, Span<double>& buffer) { get_called_with_arguments = std::make_pair(beg, end); }
  void put(size_t beg, size_t end, const Span<double>& buffer) { put_called_with_arguments = std::make_pair(beg, end); }
  std::optional<std::pair<size_t, size_t>> get_called_with_arguments;
  std::optional<std::pair<size_t, size_t>> put_called_with_arguments;
};
} // namespace

struct BlockReaderF : ::testing::Test {
  DummyArray a;
  size_t start = 3;
  size_t end = 37;
  size_t block_size = 5;
};

TEST_F(BlockReaderF, constructor) {
  auto b = BlockReader<DummyArray>(a, start, end, block_size);
  ASSERT_EQ(b.n_blocks(), (end - start) / block_size);
  ASSERT_EQ(b.n_blocks(), b.distribution().size());
}

TEST_F(BlockReaderF, get) {
  auto b = BlockReader<DummyArray>(a, start, end, block_size);
  auto buffer = std::vector<double>();
  for (size_t i = 0; i < b.n_blocks(); ++i) {
    b.get(i, buffer);
    ASSERT_TRUE(a.get_called_with_arguments);
    auto [beg, end] = b.distribution().range(i);
    ASSERT_EQ(a.get_called_with_arguments->first, start + beg) << " iter = " << std::to_string(i);
    ASSERT_EQ(a.get_called_with_arguments->second, start + end) << " iter = " << std::to_string(i);
    ASSERT_EQ(buffer.size(), end - beg);
  }
}

TEST_F(BlockReaderF, get_block) {
  auto b = BlockReader<DummyArray>(a, start, end, block_size);
  auto block = Block{};
  for (size_t i = 0; i < b.n_blocks(); ++i) {
    b.get(i, block);
    ASSERT_TRUE(a.get_called_with_arguments);
    auto [beg, end] = b.distribution().range(i);
    ASSERT_EQ(a.get_called_with_arguments->first, start + beg) << " iter = " << std::to_string(i);
    ASSERT_EQ(a.get_called_with_arguments->second, start + end) << " iter = " << std::to_string(i);
    ASSERT_EQ(block.start, start + beg);
    ASSERT_EQ(block.end, start + end);
    ASSERT_EQ(block.buffer.size(), end - beg);
  }
}

TEST_F(BlockReaderF, put) {
  auto b = BlockReader<DummyArray>(a, start, end, block_size);
  auto buffer = std::vector<double>();
  for (size_t i = 0; i < b.n_blocks(); ++i) {
    auto [beg, end] = b.distribution().range(i);
    buffer.assign(end - beg, 11);
    b.put(i, buffer);
    ASSERT_TRUE(a.put_called_with_arguments);
    ASSERT_EQ(a.put_called_with_arguments->first, start + beg) << " iter = " << std::to_string(i);
    ASSERT_EQ(a.put_called_with_arguments->second, start + end) << " iter = " << std::to_string(i);
    ASSERT_EQ(buffer.size(), end - beg);
  }
}
