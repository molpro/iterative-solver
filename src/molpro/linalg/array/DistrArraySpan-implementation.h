#ifndef LINEARALGEBRA_SRC_MOLPRO_LINALG_ARRAY_DISTRARRAYSPAN_IMPLEMENTATION_H_
#define LINEARALGEBRA_SRC_MOLPRO_LINALG_ARRAY_DISTRARRAYSPAN_IMPLEMENTATION_H_
#include "DistrArraySpan.h"
#include "util/Distribution.h"
#include <memory>
#include <string>

namespace molpro::linalg::array {
template <typename T>
DistrArraySpan<T>::DistrArraySpan() = default;

template <typename T>
DistrArraySpan<T>::DistrArraySpan(size_t dimension, MPI_Comm commun)
    : DistrArraySpan(std::make_unique<Distribution>(
    util::make_distribution_spread_remainder<index_type>(dimension, mpi_size(commun))),
                     commun) {}

template <typename T>
DistrArraySpan<T>::DistrArraySpan(std::unique_ptr<Distribution> distribution, MPI_Comm commun)
    : DistrArray<T>(distribution->border().second, commun), m_distribution(std::move(distribution)) {
  if (m_distribution->border().first != 0)
    error("Distribution of array must start from 0");
}

template <typename T>
DistrArraySpan<T>::DistrArraySpan(const DistrArraySpan& source)
    : DistrArray<T>(source.size(), source.communicator()),
      m_distribution(source.m_distribution ? std::make_unique<Distribution>(*source.m_distribution) : nullptr) {
  if (!source.empty()) {
    DistrArraySpan<T>::allocate_buffer(source.m_span);
  }
}

template <typename T>
DistrArraySpan<T>::DistrArraySpan(const DistrArray<T>& source)
    : DistrArray<T>(source), m_distribution(std::make_unique<Distribution>(source.distribution())) {
  if (!source.empty()) {
    DistrArraySpan<T>::allocate_buffer(Span<value_type>(&(*source.local_buffer())[0], source.size()));
  }
}

template <typename T>
DistrArraySpan<T>::DistrArraySpan(DistrArraySpan&& source) noexcept
    : DistrArray<T>(source.m_dimension, source.m_communicator), m_span(std::move(source.m_span)),
      m_allocated(source.m_allocated), m_distribution(std::move(source.m_distribution)) {
  source.m_allocated = false;
}

template <typename T>
DistrArraySpan<T>::~DistrArraySpan() {}

template <typename T>
DistrArraySpan<T>& DistrArraySpan<T>::operator=(const DistrArraySpan<T>& source) {
  if (this == &source)
    return *this;
  if (source.empty() || empty() || !DistrArray<T>::compatible(source)) {
    free_buffer();
    DistrArraySpan<T> t{source};
    swap(*this, t);
  } else {
    allocate_buffer(source.m_span);
  }
  return *this;
}

template <typename T>
DistrArraySpan<T>& DistrArraySpan<T>::operator=(DistrArraySpan<T>&& source) noexcept {
  DistrArraySpan<T> t{std::move(source)};
  swap(*this, t);
  return *this;
}

template <typename T>
void swap(DistrArraySpan<T>& a1, DistrArraySpan<T>& a2) noexcept {
  using std::swap;
  swap(a1.m_dimension, a2.m_dimension);
  swap(a1.m_communicator, a2.m_communicator);
  swap(a1.m_distribution, a2.m_distribution);
  swap(a1.m_allocated, a2.m_allocated);
  swap(a1.m_span, a2.m_span);
}

template <typename T>
void DistrArraySpan<T>::free_buffer() {
  if (m_allocated) {
    // TODO: what to do with m_span?
    m_allocated = false;
  }
}

template <typename T>
void DistrArraySpan<T>::allocate_buffer() {
} // TODO: Doesn't make sense to have it... Or should we allocate empty space?

template <typename T>
void DistrArraySpan<T>::allocate_buffer(Span<value_type> buffer) {
  // if (!empty())  // TODO: OK to be "re-writable"?
  //  return;
  if (!m_distribution)
    error("Cannot allocate an array without distribution");
  // m_buffer = std::make_unique<LocalBufferSpan>(buffer);
  m_span = buffer;
  int rank;
  MPI_Comm_rank(this->m_communicator, &rank);
  index_type lo, hi;
  std::tie(lo, hi) = m_distribution->range(rank);
  size_t n = hi - lo;
  if (m_span.size() < n)
    error("Specified external buffer is too small");
  m_allocated = true;
}

template <typename T>
DistrArraySpan<T>::LocalBufferSpan::LocalBufferSpan(DistrArraySpan<T>& source) {
  if (!source.m_allocated)
    source.error("attempting to access local buffer of empty array");
  int rank;
  MPI_Comm_rank(source.communicator(), &rank);
  index_type hi;
  std::tie(this->m_start, hi) = source.distribution().range(rank);
  this->m_buffer = source.m_span.data();
  this->m_size = source.m_span.size();
  if (this->m_size != (hi - this->m_start))
    source.error("size of mapped buffer different from the local distribution size");
}

template <typename T>
DistrArraySpan<T>::LocalBufferSpan::LocalBufferSpan(const DistrArraySpan<T>& source) {
  if (!source.m_allocated)
    source.error("attempting to access local buffer of empty array");
  int rank;
  MPI_Comm_rank(source.communicator(), &rank);
  index_type hi;
  std::tie(this->m_start, hi) = source.distribution().range(rank);
  this->m_buffer = const_cast<value_type*>(source.m_span.data());
  this->m_size = source.m_span.size();
  if (this->m_size != (hi - this->m_start))
    source.error("size of mapped buffer different from the local distribution size");
}

template <typename T>
bool DistrArraySpan<T>::empty() const {
  return !m_allocated;
}

template <typename T>
const typename DistrArraySpan<T>::Distribution& DistrArraySpan<T>::distribution() const {
  if (!m_distribution)
    error("allocate buffer before asking for distribution");
  return *m_distribution;
}

template <typename T>
std::unique_ptr<typename DistrArray<T>::LocalBuffer> DistrArraySpan<T>::local_buffer() {
  return std::make_unique<LocalBufferSpan>(*this);
}

template <typename T>
std::unique_ptr<const typename DistrArray<T>::LocalBuffer> DistrArraySpan<T>::local_buffer() const {
  return std::make_unique<const LocalBufferSpan>(*this);
}

template <typename T>
typename DistrArraySpan<T>::value_type DistrArraySpan<T>::at(typename DistrArraySpan<T>::index_type ind) const {
  value_type val;
  get(ind, ind + 1, &val);
  return val;
}

template <typename T>
void DistrArraySpan<T>::set(DistrArraySpan<T>::index_type ind, DistrArraySpan<T>::value_type val) {
  put(ind, ind + 1, &val);
}

template <typename T>
void DistrArraySpan<T>::get(DistrArraySpan<T>::index_type lo, DistrArraySpan<T>::index_type hi,
                            DistrArraySpan<T>::value_type* buf) const {
  if (lo >= hi)
    return;
  int rank;
  MPI_Comm_rank(this->m_communicator, &rank);
  DistrArraySpan<T>::index_type lo_loc, hi_loc;
  std::tie(lo_loc, hi_loc) = m_distribution->range(rank);
  if (lo < lo_loc || hi > hi_loc) {
    error("Only local array indices can be accessed via DistrArraySpan.get() function");
  }
  DistrArraySpan<T>::index_type offset = lo - lo_loc;
  DistrArraySpan<T>::index_type length = hi - lo;
  for (int i = offset; i < offset + length; i++) {
    buf[i - offset] = m_span[i];
  }
}

template <typename T>
std::vector<typename DistrArraySpan<T>::value_type> DistrArraySpan<T>::get(index_type lo, index_type hi) const {
  if (lo >= hi)
    return {};
  auto buf = std::vector<value_type>(hi - lo);
  get(lo, hi, &buf[0]);
  return buf;
}

template <typename T>
void DistrArraySpan<T>::put(index_type lo, index_type hi, const value_type* data) {
  if (lo >= hi)
    return;
  int rank;
  MPI_Comm_rank(this->m_communicator, &rank);
  index_type lo_loc, hi_loc;
  std::tie(lo_loc, hi_loc) = m_distribution->range(rank);
  if (lo < lo_loc || hi > hi_loc) {
    error("Only values at local array indices can be written via DistrArraySpan.put() function");
  }
  index_type offset = lo - lo_loc;
  index_type length = hi - lo;
  for (index_type i = offset; i < offset + length; i++) {
    m_span[i] = data[i - offset];
  }
}

template <typename T>
void DistrArraySpan<T>::acc(index_type lo, index_type hi, const value_type* data) {
  if (lo >= hi)
    return;
  auto disk_copy = get(lo, hi);
  std::transform(disk_copy.begin(), disk_copy.end(), data, disk_copy.begin(), [](auto& l, auto& r) { return l + r; });
  put(lo, hi, &disk_copy[0]);
}

template <typename T>
std::vector<typename DistrArraySpan<T>::value_type>
DistrArraySpan<T>::gather(const std::vector<index_type>& indices) const {
  std::vector<value_type> data;
  data.reserve(indices.size());
  int rank;
  MPI_Comm_rank(this->m_communicator, &rank);
  auto minmax = std::minmax_element(indices.begin(), indices.end());
  index_type lo_loc, hi_loc;
  std::tie(lo_loc, hi_loc) = m_distribution->range(rank);
  if (*minmax.first < lo_loc || *minmax.second > hi_loc) {
    error("Only local array indices can be accessed via DistrArraySpan.gather() function");
  }
  for (auto i : indices) {
    data.push_back(at(i));
  }
  return data;
}

template <typename T>
void DistrArraySpan<T>::scatter(const std::vector<index_type>& indices, const std::vector<value_type>& data) {
  if (indices.size() != data.size()) {
    error("Length of the indices and data vectors should be the same: DistrArray::scatter()");
  }
  int rank;
  MPI_Comm_rank(this->m_communicator, &rank);
  auto minmax = std::minmax_element(indices.begin(), indices.end());
  index_type lo_loc, hi_loc;
  std::tie(lo_loc, hi_loc) = m_distribution->range(rank);
  if (*minmax.first < lo_loc || *minmax.second > hi_loc) {
    error("Only local array indices can be accessed via DistrArraySpan.gather() function");
  }
  for (auto i : indices) {
    set(i, data[i - *minmax.first]);
  }
}

template <typename T>
void DistrArraySpan<T>::scatter_acc(std::vector<index_type>& indices, const std::vector<value_type>& data) {
  auto disk_copy = gather(indices);
  std::transform(data.begin(), data.end(), disk_copy.begin(), disk_copy.begin(),
                 [](auto& l, auto& r) { return l + r; });
  scatter(indices, disk_copy);
}

template <typename T>
std::vector<typename DistrArraySpan<T>::value_type> DistrArraySpan<T>::vec() const {
  int rank;
  MPI_Comm_rank(this->m_communicator, &rank);
  index_type lo_loc, hi_loc;
  std::tie(lo_loc, hi_loc) = m_distribution->range(rank);
  return get(lo_loc, hi_loc);
}

} // namespace molpro::linalg::array

#endif // LINEARALGEBRA_SRC_MOLPRO_LINALG_ARRAY_DISTRARRAYSPAN_IMPLEMENTATION_H_
