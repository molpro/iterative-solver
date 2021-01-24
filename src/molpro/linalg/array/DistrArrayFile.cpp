#include <functional>
#include <unistd.h>

#include "ArrayFile.h"
#include "DistrArrayFile.h"
#include "util/Distribution.h"
#include "util/temp_file.h"
#include "utility"

namespace molpro::linalg::array {
namespace {
int mpi_size(MPI_Comm comm) {
  int rank;
  MPI_Comm_size(comm, &rank);
  return rank;
}
} // namespace

DistrArrayFile::DistrArrayFile(DistrArrayFile&& source) noexcept
    : DistrArrayDisk(std::move(source)), m_local_array(std::move(source.m_local_array)) {}

DistrArrayFile::DistrArrayFile(size_t dimension, MPI_Comm comm, const std::string& directory)
    : DistrArrayFile(std::make_unique<Distribution>(
                         util::make_distribution_spread_remainder<index_type>(dimension, mpi_size(comm))),
                     comm, directory) {}

DistrArrayFile::DistrArrayFile(std::unique_ptr<Distribution> distribution, MPI_Comm comm, const std::string& directory)
    : DistrArrayDisk(std::move(distribution), comm), m_local_array(std::invoke([&]() {
        int rank;
        MPI_Comm_rank(comm, &rank);
        auto [beg, end] = m_distribution->range(rank);
        return std::make_unique<ArrayFile>(directory, end - beg);
      })) {
  if (m_distribution->border().first != 0)
    DistrArray::error("Distribution of array must start from 0");
}

DistrArrayFile::DistrArrayFile(const DistrArrayFile& source)
    : DistrArrayFile(std::make_unique<Distribution>(source.distribution()), source.communicator(),
                     source.m_local_array->directory()) {
  DistrArrayFile::copy(source);
}

DistrArrayFile::DistrArrayFile(const DistrArray& source)
    : DistrArrayFile(std::make_unique<Distribution>(source.distribution()), source.communicator()) {
  DistrArrayFile::copy(source);
}

DistrArrayFile& DistrArrayFile::operator=(DistrArrayFile&& source) noexcept {
  DistrArrayFile t{std::move(source)};
  swap(*this, t);
  return *this;
}

void swap(DistrArrayFile& x, DistrArrayFile& y) noexcept {
  using std::swap;
  swap(x.m_dimension, y.m_dimension);
  swap(x.m_communicator, y.m_communicator);
  swap(x.m_allocated, y.m_allocated);
  swap(x.m_distribution, y.m_distribution);
  swap(x.m_local_array, y.m_local_array);
}

DistrArrayFile::~DistrArrayFile() = default;

bool DistrArrayFile::compatible(const DistrArrayFile& source) const {
  auto res = DistrArray::compatible(source);
  if (m_distribution && source.m_distribution)
    res &= m_distribution->compatible(*source.m_distribution);
  else
    res &= !m_distribution && !source.m_distribution;
  return res;
}

DistrArray::value_type DistrArrayFile::at(DistrArray::index_type ind) const {
  value_type val;
  get(ind, ind + 1, &val);
  return val;
}

void DistrArrayFile::set(DistrArray::index_type ind, DistrArray::value_type val) { put(ind, ind + 1, &val); }

void DistrArrayFile::get(index_type lo, index_type hi, value_type* buf) const {
  int rank;
  MPI_Comm_rank(m_communicator, &rank);
  auto [lo_loc, hi_loc] = m_distribution->range(rank);
  if (lo < lo_loc || hi > hi_loc) {
    error("Only local array indices can be accessed via DistrArrayFile.get() function");
  }
  auto start = lo - lo_loc;
  auto length = hi - lo;
  m_local_array->get(start, start + length, Span<value_type>(buf, length));
}

std::vector<DistrArrayFile::value_type> DistrArrayFile::get(DistrArray::index_type lo,
                                                            DistrArray::index_type hi) const {
  if (lo >= hi)
    return {};
  auto buf = std::vector<DistrArray::value_type>(hi - lo);
  get(lo, hi, &buf[0]);
  return buf;
}

void DistrArrayFile::put(index_type lo, index_type hi, const value_type* data) {
  if (lo >= hi)
    return;
  int rank;
  MPI_Comm_rank(m_communicator, &rank);
  auto [lo_loc, hi_loc] = m_distribution->range(rank);
  if (lo < lo_loc || hi > hi_loc) {
    error("Only values at local array indices can be written via DistrArrayFile.put() function");
  }
  auto start = lo - lo_loc;
  auto length = hi - lo;
  m_local_array->put(start, start + length, Span<value_type>(const_cast<value_type*>(data), length));
}

void DistrArrayFile::acc(DistrArray::index_type lo, DistrArray::index_type hi, const DistrArray::value_type* data) {
  if (lo >= hi)
    return;
  auto disk_copy = get(lo, hi);
  std::transform(disk_copy.begin(), disk_copy.end(), data, disk_copy.begin(), [](auto& l, auto& r) { return l + r; });
  put(lo, hi, &disk_copy[0]);
}

std::vector<DistrArrayFile::value_type> DistrArrayFile::gather(const std::vector<index_type>& indices) const {
  std::vector<value_type> data;
  data.reserve(indices.size());
  int rank;
  MPI_Comm_rank(m_communicator, &rank);
  auto minmax = std::minmax_element(indices.begin(), indices.end());
  DistrArray::index_type lo_loc, hi_loc;
  std::tie(lo_loc, hi_loc) = m_distribution->range(rank);
  if (*minmax.first < lo_loc || *minmax.second > hi_loc) {
    error("Only local array indices can be accessed via DistrArrayFile.gather() function");
  }
  for (auto i : indices) {
    data.push_back(at(i));
  }
  return data;
}

void DistrArrayFile::scatter(const std::vector<index_type>& indices, const std::vector<value_type>& data) {
  if (indices.size() != data.size()) {
    error("Length of the indices and data vectors should be the same: DistrArray::scatter()");
  }
  int rank;
  MPI_Comm_rank(m_communicator, &rank);
  auto minmax = std::minmax_element(indices.begin(), indices.end());
  DistrArray::index_type lo_loc, hi_loc;
  std::tie(lo_loc, hi_loc) = m_distribution->range(rank);
  if (*minmax.first < lo_loc || *minmax.second > hi_loc) {
    error("Only local array indices can be accessed via DistrArrayFile.gather() function");
  }
  for (auto i : indices) {
    set(i, data[i - *minmax.first]);
  }
}

void DistrArrayFile::scatter_acc(std::vector<index_type>& indices, const std::vector<value_type>& data) {
  auto disk_copy = gather(indices);
  std::transform(data.begin(), data.end(), disk_copy.begin(), disk_copy.begin(),
                 [](auto& l, auto& r) { return l + r; });
  scatter(indices, disk_copy);
}

std::vector<DistrArrayFile::value_type> DistrArrayFile::vec() const {
  int rank;
  MPI_Comm_rank(m_communicator, &rank);
  DistrArray::index_type lo_loc, hi_loc;
  std::tie(lo_loc, hi_loc) = m_distribution->range(rank);
  return get(lo_loc, hi_loc);
}

} // namespace molpro::linalg::array
