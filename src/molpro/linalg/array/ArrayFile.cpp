#include <unistd.h>

#include "ArrayFile.h"
#include "util/temp_file.h"

namespace molpro::linalg::array {
ArrayFile::ArrayFile(size_t dimension, std::string_view directory)
    : m_dim(dimension), m_dir(std::filesystem::absolute(std::filesystem::path(directory))), m_file(make_file()) {}

ArrayFile::ArrayFile(size_t dimension)
    : m_dim(dimension), m_dir(std::filesystem::current_path()), m_file(make_file()) {}

ArrayFile::~ArrayFile() {
  // FIXME issue #67 explicitly remove the file, so that temporary is deleted on Windows
}

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

} // namespace molpro::linalg::array
