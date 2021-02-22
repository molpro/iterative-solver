#include <molpro/linalg/array/DistrArrayFile.h>
namespace molpro::linalg::array {
namespace {
int mpi_size(MPI_Comm comm) {
  int rank;
  MPI_Comm_size(comm, &rank);
  return rank;
}
} // namespace
} // namespace molpro::linalg::array

#include <unistd.h>

#include <molpro/linalg/array/DistrArrayFile.h>
#include <molpro/linalg/array/util/Distribution.h>
#include <molpro/linalg/array/util/temp_file.h>
#include <utility>

namespace molpro::linalg::array {
template <typename T>
DistrArrayFile<T>::DistrArrayFile() = default;

template <typename T>
DistrArrayFile<T>::DistrArrayFile(DistrArrayFile&& source) noexcept
    : DistrArrayDisk<T>(std::move(source)), m_dir(std::move(source.m_dir)), m_file(std::move(source.m_file)) {}

template <typename T>
DistrArrayFile<T>::DistrArrayFile(size_t dimension, MPI_Comm comm, const std::string& directory)
    : DistrArrayFile(std::make_unique<Distribution>(
                         util::make_distribution_spread_remainder<index_type>(dimension, mpi_size(comm))),
                     comm, directory) {}

template <typename T>
DistrArrayFile<T>::DistrArrayFile(std::unique_ptr<Distribution> distribution, MPI_Comm comm, const std::string& directory)
    : DistrArrayDisk<T>(std::move(distribution), comm), m_dir(fs::absolute(fs::path(directory))), m_file(make_file()) {
  if (this->m_distribution->border().first != 0)
    error("Distribution of array must start from 0");
}

template <typename T>
DistrArrayFile<T>::DistrArrayFile(const DistrArrayFile& source)
    : DistrArrayDisk<T>(source), m_dir(source.m_dir), m_file(make_file()) {
  if (!source.empty()) {
    DistrArrayFile<T>::copy(source);
  }
}

template <typename T>
DistrArrayFile<T>::DistrArrayFile(const DistrArray<T>& source)
    : DistrArrayFile(std::make_unique<Distribution>(source.distribution()), source.communicator()) {
  if (!source.empty()) {
    DistrArrayFile<T>::copy(source);
  }
}

template <typename T>
DistrArrayFile<T>& DistrArrayFile<T>::operator=(DistrArrayFile&& source) noexcept {
  DistrArrayFile t{std::move(source)};
  swap(*this, t);
  return *this;
}

template <typename T>
DistrArrayFile<T> DistrArrayFile<T>::CreateTempCopy(const DistrArray<T>& source, const std::string& directory) {
  DistrArrayFile t(std::make_unique<Distribution>(source.distribution()), source.communicator(), directory);
  t.copy(source);
  return t;
}

template <typename T>
void swap(DistrArrayFile<T>& x, DistrArrayFile<T>& y) noexcept {
  using std::swap;
  swap(x.m_dimension, y.m_dimension);
  swap(x.m_communicator, y.m_communicator);
  swap(x.m_allocated, y.m_allocated);
  swap(x.m_view_buffer, y.m_view_buffer);
  swap(x.m_owned_buffer, y.m_owned_buffer);
  swap(x.m_distribution, y.m_distribution);
  swap(x.m_file, y.m_file);
  swap(x.m_dir, y.m_dir);
}

template <typename T>
DistrArrayFile<T>::~DistrArrayFile() = default;

template <typename T>
bool DistrArrayFile<T>::compatible(const DistrArrayFile& source) const {
  auto res = compatible(source);
  if (this->m_distribution && source.m_distribution)
    res &= this->m_distribution->compatible(*source.m_distribution);
  else
    res &= !this->m_distribution && !source.m_distribution;
  return res;
}

template <typename T>
std::fstream DistrArrayFile<T>::make_file() {
  std::fstream file;
  std::string file_name = util::temp_file_name(m_dir.string() + "/", "");
  file.open(file_name.c_str(), std::ios::out | std::ios::binary);
  file.close();
  file.open(file_name.c_str(), std::ios::out | std::ios::in | std::ios::binary);
  unlink(file_name.c_str());
  return file;
}

template <typename T>
void DistrArrayFile<T>::open_access() {}
template <typename T>
void DistrArrayFile<T>::close_access() {}

template <typename T>
bool DistrArrayFile<T>::empty() const { return !m_file.is_open(); }

template <typename T>
void DistrArrayFile<T>::erase() {}

template <typename T>
typename DistrArrayFile<T>::value_type DistrArrayFile<T>::at(index_type ind) const {
  value_type val;
  get(ind, ind + 1, &val);
  return val;
}

template <typename T>
void DistrArrayFile<T>::set(index_type ind, value_type val) { put(ind, ind + 1, &val); }

template <typename T>
void DistrArrayFile<T>::get(index_type lo, index_type hi, value_type* buf) const {
  if (lo >= hi)
    return;
  index_type length = hi - lo;
  int current = m_file.tellg();
  if (current < length)
    return;
  int rank;
  MPI_Comm_rank(this->m_communicator, &rank);
  index_type lo_loc, hi_loc;
  std::tie(lo_loc, hi_loc) = this->m_distribution->range(rank);
  if (lo < lo_loc || hi > hi_loc) {
    error("Only local array indices can be accessed via DistrArrayFile.get() function");
  }
  index_type offset = lo - lo_loc;
  m_file.seekg(offset * sizeof(value_type));
  m_file.read((char*)buf, length * sizeof(value_type));
}

template <typename T>
std::vector<typename DistrArrayFile<T>::value_type> DistrArrayFile<T>::get(index_type lo,
                                                            index_type hi) const {
  if (lo >= hi)
    return {};
  auto buf = std::vector<typename DistrArrayFile<T>::value_type>(hi - lo);
  get(lo, hi, &buf[0]);
  return buf;
}

template <typename T>
void DistrArrayFile<T>::put(index_type lo, index_type hi, const value_type* data) {
  if (lo >= hi)
    return;
  int rank;
  MPI_Comm_rank(this->m_communicator, &rank);
  index_type lo_loc, hi_loc;
  std::tie(lo_loc, hi_loc) = this->m_distribution->range(rank);
  if (lo < lo_loc || hi > hi_loc) {
    error("Only values at local array indices can be written via DistrArrayFile.put() function");
  }
  index_type offset = lo - lo_loc;
  index_type length = hi - lo;
  m_file.seekp(offset * sizeof(value_type));
  m_file.write((const char*)data, length * sizeof(value_type));
}

template <typename T>
void DistrArrayFile<T>::acc(index_type lo, index_type hi, const value_type* data) {
  if (lo >= hi)
    return;
  auto disk_copy = get(lo, hi);
  std::transform(disk_copy.begin(), disk_copy.end(), data, disk_copy.begin(), [](auto& l, auto& r) { return l + r; });
  put(lo, hi, &disk_copy[0]);
}

template <typename T>
std::vector<typename DistrArrayFile<T>::value_type> DistrArrayFile<T>::gather(const std::vector<index_type>& indices) const {
  std::vector<value_type> data;
  data.reserve(indices.size());
  int rank;
  MPI_Comm_rank(this->m_communicator, &rank);
  auto minmax = std::minmax_element(indices.begin(), indices.end());
  index_type lo_loc, hi_loc;
  std::tie(lo_loc, hi_loc) = this->m_distribution->range(rank);
  if (*minmax.first < lo_loc || *minmax.second > hi_loc) {
    error("Only local array indices can be accessed via DistrArrayFile.gather() function");
  }
  for (auto i : indices) {
    data.push_back(at(i));
  }
  return data;
}

template <typename T>
void DistrArrayFile<T>::scatter(const std::vector<index_type>& indices, const std::vector<value_type>& data) {
  if (indices.size() != data.size()) {
    error("Length of the indices and data vectors should be the same: scatter()");
  }
  int rank;
  MPI_Comm_rank(this->m_communicator, &rank);
  auto minmax = std::minmax_element(indices.begin(), indices.end());
  index_type lo_loc, hi_loc;
  std::tie(lo_loc, hi_loc) = this->m_distribution->range(rank);
  if (*minmax.first < lo_loc || *minmax.second > hi_loc) {
    error("Only local array indices can be accessed via DistrArrayFile.gather() function");
  }
  for (auto i : indices) {
    set(i, data[i - *minmax.first]);
  }
}

template <typename T>
void DistrArrayFile<T>::scatter_acc(std::vector<index_type>& indices, const std::vector<value_type>& data) {
  auto disk_copy = gather(indices);
  std::transform(data.begin(), data.end(), disk_copy.begin(), disk_copy.begin(),
                 [](auto& l, auto& r) { return l + r; });
  scatter(indices, disk_copy);
}

template <typename T>
std::vector<typename DistrArrayFile<T>::value_type> DistrArrayFile<T>::vec() const {
  int rank;
  MPI_Comm_rank(this->m_communicator, &rank);
  index_type lo_loc, hi_loc;
  std::tie(lo_loc, hi_loc) = this->m_distribution->range(rank);
  return get(lo_loc, hi_loc);
}

} // namespace molpro::linalg::array

template class molpro::linalg::array::DistrArrayFile<double>;
template void molpro::linalg::array::swap(molpro::linalg::array::DistrArrayFile<double> &a1, molpro::linalg::array::DistrArrayFile<double> &a2);
