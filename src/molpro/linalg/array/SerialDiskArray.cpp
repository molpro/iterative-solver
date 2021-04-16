#include <unistd.h>
#include <vector>
#include <utility>

#include <molpro/linalg/array/util/temp_file.h>
#include "SerialDiskArray.h"


namespace molpro::linalg::array {

SerialDiskArray::SerialDiskArray() = default;

SerialDiskArray::SerialDiskArray(const SerialDiskArray& source) : m_dir(source.m_dir), m_file(make_file()) {
  if (!source.empty()){
    SerialDiskArray::copy(source);
  }
}

SerialDiskArray::SerialDiskArray(SerialDiskArray&& source) noexcept : m_dir(std::move(source.m_dir)),
                 m_file(std::move(source.m_file)), m_dimension(source.m_dimension) {}

SerialDiskArray::SerialDiskArray(const Span<value_type>& source) {}

SerialDiskArray::SerialDiskArray(const std::vector<value_type>& source) {}

SerialDiskArray& SerialDiskArray::operator=(const SerialDiskArray& source) {
  if (!source.empty()){
    SerialDiskArray::copy(source);
  }
}
SerialDiskArray& SerialDiskArray::operator=(const SerialDiskArray& source) {

  SerialDiskArray& SerialDiskArray::operator=(const Span<value_type>& source) {}

  SerialDiskArray& SerialDiskArray::operator=(const std::vector<value_type>& source) {}

SerialDiskArray::~SerialDiskArray() = default;

std::fstream SerialDiskArray::make_file() {
  std::fstream file;
  std::string file_name =
      util::temp_file_name(m_dir.string() + "/", "");
  file.open(file_name.c_str(), std::ios::out | std::ios::binary);
  file.close();
  file.open(file_name.c_str(), std::ios::out | std::ios::in | std::ios::binary);
  unlink(file_name.c_str());
  return file;
}

void swap(SerialDiskArray& x, SerialDiskArray& y) noexcept {
  using std::swap;
  swap(x.m_dimension, y.m_dimension);
  //swap(x.m_allocated, y.m_allocated);
  //swap(x.m_view_buffer, y.m_view_buffer);
  //swap(x.m_owned_buffer, y.m_owned_buffer);
  swap(x.m_file, y.m_file);
  swap(x.m_dir, y.m_dir);
}

void SerialDiskArray::scal(value_type alpha) {}

bool SerialDiskArray::empty() const { return false; }

SerialDiskArray::value_type SerialDiskArray::at(SerialDiskArray::index_type ind) const { return 0; }

void SerialDiskArray::set(SerialDiskArray::index_type ind, SerialDiskArray::value_type val) {}

void SerialDiskArray::get(SerialDiskArray::index_type lo, SerialDiskArray::index_type hi,
                          SerialDiskArray::value_type* buf) const {}

std::vector<SerialDiskArray::value_type> SerialDiskArray::get(SerialDiskArray::index_type lo,
                                                              SerialDiskArray::index_type hi) const {
  std::vector<value_type> data;
  return data;
}

void SerialDiskArray::put(SerialDiskArray::index_type lo, SerialDiskArray::index_type hi,
                          const SerialDiskArray::value_type* data) {}

void SerialDiskArray::acc(SerialDiskArray::index_type lo, SerialDiskArray::index_type hi,
                          const SerialDiskArray::value_type* data) {}

std::vector<SerialDiskArray::value_type>
SerialDiskArray::gather(const std::vector<SerialDiskArray::index_type>& indices) const {
  std::vector<value_type> data;
  return data;
}

void SerialDiskArray::scatter(const std::vector<SerialDiskArray::index_type>& indices,
                              const std::vector<SerialDiskArray::value_type>& data) {}

void SerialDiskArray::scatter_acc(std::vector<SerialDiskArray::index_type>& indices,
                                  const std::vector<SerialDiskArray::value_type>& data) {}

} // namespace molpro::linalg::array

