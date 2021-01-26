#include <unistd.h>

#include "ArrayFile.h"
#include "util/BufferedReader.h"
#include "util/iterable_lingalg.h"
#include "util/select_max_dot.h"
#include "util/temp_file.h"

namespace molpro::linalg::array {
ArrayFile::ArrayFile(std::string_view directory, size_t dimension, size_t block_size)
    : m_dim(dimension), m_dir(std::filesystem::absolute(std::filesystem::path(directory))), m_file(make_file()),
      m_block_reader(std::make_unique<util::BlockReader<ArrayFile>>(*this, 0, dimension, block_size)) {}

ArrayFile::ArrayFile(size_t dimension, size_t block_size)
    : ArrayFile(std::filesystem::current_path().string(), dimension, block_size) {}

ArrayFile::~ArrayFile() {
  // FIXME issue #67 explicitly remove the file, so that temporary is deleted on Windows
}

ArrayFile::ArrayFile(ArrayFile&& other) = default;
ArrayFile& ArrayFile::operator=(ArrayFile&& other) = default;

size_t ArrayFile::size() const { return m_dim; }

std::fstream ArrayFile::make_file() {
  std::fstream file;
  std::string file_name = util::temp_file_name(m_dir.string() + "/", "");
  file.open(file_name.c_str(), std::ios::out | std::ios::binary);
  file.close();
  file.open(file_name.c_str(), std::ios::out | std::ios::in | std::ios::binary);
  unlink(file_name.c_str());
  return file;
}

void ArrayFile::get(index_type lo, index_type hi, Span<value_type> buf) const {
  if (lo >= hi)
    return;
  index_type length = hi - lo;
  int current = m_file.tellg();
  // FIXME why is this necessary?
  if (current < length)
    return;
  if (lo < 0 || hi > m_dim) {
    throw std::runtime_error("get() range is outside of array bounds");
  }
  m_file.seekg(lo * sizeof(value_type));
  m_file.read((char*)&buf[0], length * sizeof(value_type));
}

std::vector<ArrayFile::value_type> ArrayFile::get(index_type lo, index_type hi) const {
  if (lo >= hi)
    return {};
  auto buf = std::vector<value_type>(hi - lo);
  get(lo, hi, Span<value_type>(&buf[0], buf.size()));
  return buf;
}

void ArrayFile::put(ArrayFile::index_type lo, ArrayFile::index_type hi, const Span<value_type>& data) {
  if (lo >= hi)
    return;
  if (lo < 0 || hi > m_dim) {
    throw std::runtime_error("put() range is outside of array bounds");
  }
  auto length = hi - lo;
  m_file.seekp(lo * sizeof(value_type));
  m_file.write((const char*)&data[0], length * sizeof(value_type));
}

void ArrayFile::fill(ArrayFile::value_type value) {
  m_file.seekp(0);
  for (size_t i = 0; i < size(); ++i) {
    m_file.write((const char*)&value, sizeof(value_type));
  }
}

bool ArrayFile::compatible(const ArrayFile& other) {
  return size() == other.size() && m_block_reader->distribution().compatible(other.m_block_reader->distribution());
}

void ArrayFile::scal(ArrayFile::value_type value) {
  auto f_scal = [value](auto& x) { util::scal(value, x.buffer); };
  util::buffered_unary_operation(*m_block_reader, f_scal, true);
}

void ArrayFile::axpy(ArrayFile::value_type value, const ArrayFile& x) {
  auto f_axpy = [value](const util::Block& xx, util::Block& yy) { util::axpy(value, xx.buffer, yy.buffer); };
  util::buffered_binary_operation(*x.m_block_reader, *m_block_reader, f_axpy, false, true);
}

void ArrayFile::axpy(ArrayFile::value_type value, const Span<double>& x) {
  auto f_axpy = [value, &x](util::Block& y) {
    auto x_block = Span<double>(const_cast<double*>(x.data()) + y.start, y.buffer.size());
    util::axpy(value, x_block, y.buffer);
  };
  util::buffered_unary_operation(*m_block_reader, f_axpy, true);
}

void ArrayFile::axpy(ArrayFile::value_type value, const std::vector<double>& x) {
  axpy(value, util::vector_to_span(x));
}

void ArrayFile::axpy(ArrayFile::value_type value, const std::map<size_t, double>& x) {
  auto f_axpy = [value, &x](util::Block& y) {
    for (const auto [i, v] : x)
      if (i >= y.start && i < y.end)
        y.buffer[i - y.start] += value * v;
  };
  util::buffered_unary_operation(*m_block_reader, f_axpy, true);
}

double ArrayFile::dot(const ArrayFile& x) const {
  double tot = 0.;
  auto f_dot = [&tot](const util::Block& x, const util::Block& y) { tot += util::dot(x.buffer, y.buffer); };
  util::buffered_binary_operation(*x.m_block_reader, *m_block_reader, f_dot, false, false);
  return tot;
}

double ArrayFile::dot(const Span<double>& x) const {
  double tot = 0.;
  auto f_dot = [&tot, &x](const util::Block& y) {
    auto x_block = Span<double>(const_cast<double*>(x.data()) + y.start, y.buffer.size());
    tot += util::dot(x_block, y.buffer);
  };
  util::buffered_unary_operation(*m_block_reader, f_dot, false);
  return tot;
}

double ArrayFile::dot(const std::vector<double>& x) const { return dot(util::vector_to_span(x)); }

double ArrayFile::dot(const std::map<size_t, double>& x) const {
  double tot = 0.;
  auto f_dot = [&tot, &x](const util::Block& y) {
    for (const auto [i, v] : x)
      if (i >= y.start && i < y.end)
        tot += y.buffer[i - y.start] * v;
  };
  util::buffered_unary_operation(*m_block_reader, f_dot, false);
  return tot;
}

namespace {
using select_pair = std::pair<ArrayFile::value_type, size_t>; // value and index
using selection_priority_queue = std::priority_queue<select_pair, std::vector<select_pair>, std::greater<>>;
using selection_map = std::map<size_t, double>;

void extend_selection(selection_priority_queue& selection, const selection_map& block_selection, size_t n,
                      size_t offset) {
  for (auto [i, v] : block_selection)
    selection.emplace(v, offset + i);
  while (selection.size() > n)
    selection.pop();
}
auto pq_to_map(selection_priority_queue& selection) {
  auto selection_map = std::map<size_t, ArrayFile::value_type>{};
  while (!selection.empty()) {
    auto [v, i] = selection.top();
    selection_map.emplace(i, v);
    selection.pop();
  }
  return selection_map;
}
} // namespace

std::map<size_t, ArrayFile::value_type> ArrayFile::select_max_dot(size_t n, const ArrayFile& x) {
  auto selection = selection_priority_queue{};
  auto f_select_max_dot = [&selection, n](const util::Block& x, const util::Block& y) {
    const auto local_n = std::min(n, x.buffer.size());
    auto block_selection =
        util::select_max_dot<std::vector<double>, std::vector<double>, double, double>(local_n, x.buffer, y.buffer);
    extend_selection(selection, block_selection, n, x.start);
  };
  util::buffered_binary_operation(*x.m_block_reader, *m_block_reader, f_select_max_dot, false, false);
  return pq_to_map(selection);
}

std::map<size_t, ArrayFile::value_type> ArrayFile::select_max_dot(size_t n, const Span<double>& x) {
  auto selection = selection_priority_queue{};
  auto f_select_max_dot = [&x, &selection, n](const util::Block& y) {
    auto const block_size = y.buffer.size();
    auto const offset = y.start;
    auto const local_n = std::min(n, block_size);
    auto const x_block = Span<double>(const_cast<double*>(x.data() + offset), block_size);
    auto block_selection =
        util::select_max_dot<Span<double>, std::vector<double>, double, double>(local_n, x_block, y.buffer);
    extend_selection(selection, block_selection, n, offset);
  };
  util::buffered_unary_operation(*m_block_reader, f_select_max_dot, false);
  return pq_to_map(selection);
}

std::map<size_t, ArrayFile::value_type> ArrayFile::select_max_dot(size_t n, const std::vector<double>& x) {
  return select_max_dot(n, util::vector_to_span(x));
}

std::map<size_t, ArrayFile::value_type> ArrayFile::select_max_dot(size_t n, const std::map<size_t, double>& x) {
  auto selection = selection_priority_queue{};
  auto f_select_max_dot = [&x, &selection, n](const util::Block& y) {
    auto const block_size = y.buffer.size();
    auto local_map = selection_map{};
    for (auto it = x.lower_bound(y.start); it != x.upper_bound(y.end); ++it)
      local_map.emplace(it->first - y.start, it->second);
    auto const local_n = std::min(n, block_size);
    auto block_selection =
        util::select_max_dot_iter_sparse<std::vector<double>, std::map<size_t, double>, double, double>(
            local_n, y.buffer, local_map);
    extend_selection(selection, block_selection, n, y.start);
  };
  util::buffered_unary_operation(*m_block_reader, f_select_max_dot, false);
  return pq_to_map(selection);
}

} // namespace molpro::linalg::array
