#ifndef LINEARALGEBRA_SRC_MOLPRO_LINALG_ARRAY_DISTRARRAYMPI3_IMPLEMENTATION_H_
#define LINEARALGEBRA_SRC_MOLPRO_LINALG_ARRAY_DISTRARRAYMPI3_IMPLEMENTATION_H_
#include "DistrArrayMPI3.h"
#include "util/Distribution.h"
#include <algorithm>
#include <string>
#include <tuple>
namespace molpro::linalg::array {

template <typename T>
DistrArrayMPI3<T>::DistrArrayMPI3() = default;

template <typename T>
DistrArrayMPI3<T>::DistrArrayMPI3(size_t dimension, MPI_Comm commun)
    : DistrArrayMPI3(std::make_unique<Distribution>(
                         util::make_distribution_spread_remainder<index_type>(dimension, comm_size(commun))),
                     commun) {}

template <typename T>
DistrArrayMPI3<T>::DistrArrayMPI3(std::unique_ptr<Distribution> distribution, MPI_Comm commun)
    : DistrArray<T>(distribution->border().second, commun), m_distribution(std::move(distribution)) {
  if (m_distribution->border().first != 0)
    DistrArray<T>::error("Distribution of array must start from 0");
}

template <typename T>
DistrArrayMPI3<T>::~DistrArrayMPI3() {
  if (m_allocated)
    DistrArrayMPI3<T>::free_buffer();
}

template <typename T>
void DistrArrayMPI3<T>::allocate_buffer() {
  if (!empty())
    return;
  if (!m_distribution)
    error("Cannot allocate an array without distribution");
  int rank;
  MPI_Comm_rank(this->m_communicator, &rank);
  index_type lo, hi;
  std::tie(lo, hi) = m_distribution->range(rank);
  MPI_Aint n = hi - lo;
  double* base = nullptr;
  int size_of_type = sizeof(value_type);
  n *= size_of_type;
  MPI_Win_allocate(n, size_of_type, MPI_INFO_NULL, this->m_communicator, &base, &m_win);
  MPI_Win_lock_all(0, m_win);
  m_allocated = true;
}

template <typename T>
void DistrArrayMPI3<T>::allocate_buffer(Span<value_type> buffer) {
  if (!empty())
    return;
  if (!m_distribution)
    error("Cannot allocate an array without distribution");
  int rank;
  MPI_Comm_rank(this->m_communicator, &rank);
  index_type lo, hi;
  std::tie(lo, hi) = m_distribution->range(rank);
  MPI_Aint n = hi - lo;
  if (buffer.size() < n)
    error("Specified external buffer is too small");
  int size_of_type = sizeof(value_type);
  n *= size_of_type;
  MPI_Win_create(&buffer[0], n, size_of_type, MPI_INFO_NULL, this->m_communicator, &m_win);
  MPI_Win_lock_all(0, m_win);
  m_allocated = true;
}

template <typename T>
bool DistrArrayMPI3<T>::empty() const { return !m_allocated; }

template <typename T>
DistrArrayMPI3<T>::DistrArrayMPI3(const DistrArrayMPI3& source)
    : DistrArray<T>(source.size(), source.communicator()),
      m_distribution(source.m_distribution ? std::make_unique<Distribution>(*source.m_distribution) : nullptr) {
  if (!source.empty()) {
    DistrArrayMPI3<T>::allocate_buffer();
    DistrArray<T>::copy(source);
  }
}

template <typename T>
DistrArrayMPI3<T>::DistrArrayMPI3(const DistrArray<T>& source)
    : DistrArray<T>(source), m_distribution(std::make_unique<Distribution>(source.distribution())) {
  if (!source.empty()) {
    DistrArrayMPI3<T>::allocate_buffer();
    DistrArray<T>::copy(source);
  }
}

template <typename T>
DistrArrayMPI3<T>::DistrArrayMPI3(DistrArrayMPI3<T>&& source) noexcept
    : DistrArray<T>(source.m_dimension, source.m_communicator), m_win(source.m_win), m_allocated(source.m_allocated),
      m_distribution(std::move(source.m_distribution)) {
  source.m_allocated = false;
}

template <typename T>
DistrArrayMPI3<T>& DistrArrayMPI3<T>::operator=(const DistrArrayMPI3<T>& source) {
  if (this == &source)
    return *this;
  if (source.empty() || empty() || !this->compatible(source)) {
    free_buffer();
    DistrArrayMPI3<T> t{source};
    std::swap(*this, t);
  } else {
    allocate_buffer();
    this->copy(source);
  }
  return *this;
}

template <typename T>
DistrArrayMPI3<T>& DistrArrayMPI3<T>::operator=(DistrArrayMPI3<T>&& source) noexcept {
  DistrArrayMPI3<T> t{std::move(source)};
  using std::swap;
  swap(*this, t);
  return *this;
}

template <typename T>
void swap(DistrArrayMPI3<T>& a1, DistrArrayMPI3<T>& a2) noexcept {
  using std::swap;
  swap(a1.m_dimension, a2.m_dimension);
  swap(a1.m_communicator, a2.m_communicator);
  swap(a1.m_distribution, a2.m_distribution);
  swap(a1.m_allocated, a2.m_allocated);
  swap(a1.m_win, a2.m_win);
}

template <typename T>
void DistrArrayMPI3<T>::free_buffer() {
  if (m_allocated) {
    MPI_Win_unlock_all(m_win);
    MPI_Win_free(&m_win);
    m_allocated = false;
  }
}

template <typename T>
void DistrArrayMPI3<T>::sync() const {
  if (!empty()) {
    MPI_Win_flush_all(m_win);
    MPI_Win_sync(m_win);
  }
  MPI_Barrier(this->m_communicator);
}

template <typename T>
std::unique_ptr<const typename DistrArray<T>::LocalBuffer> DistrArrayMPI3<T>::local_buffer() const {
  return std::make_unique<const LocalBufferMPI3>(*const_cast<DistrArrayMPI3<T>*>(this));
}

template <typename T>
std::unique_ptr<typename DistrArrayMPI3<T>::LocalBuffer> DistrArrayMPI3<T>::local_buffer() {
  return std::make_unique<LocalBufferMPI3>(*this);
}

template <typename T>
typename DistrArray<T>::value_type DistrArrayMPI3<T>::at(index_type ind) const {
  value_type val;
  get(ind, ind + 1, &val);
  return val;
}
template <typename T>
void DistrArrayMPI3<T>::set(index_type ind, value_type val) { put(ind, ind + 1, &val); }

template <typename T>
void DistrArrayMPI3<T>::_get_put(index_type lo, index_type hi, const value_type* buf, RMAType option) {
  if (lo >= hi)
    return;
  auto name = std::string{"DistrArrayMPI3::_get_put"};
  if (hi > this->m_dimension)
    error(name + " out of bounds");
  if (empty())
    error(name + " called on an empty array");
  index_type p_lo, p_hi;
  std::tie(p_lo, p_hi) = m_distribution->cover(lo, hi);
  auto* curr_buf = const_cast<value_type*>(buf);
  auto requests = std::vector<MPI_Request>(p_hi - p_lo + 1);
  for (size_t i = p_lo; i < p_hi + 1; ++i) {
    index_type bound_lo, bound_hi;
    std::tie(bound_lo, bound_hi) = m_distribution->range(i);
    auto local_lo = std::max(lo, bound_lo);
    auto local_hi = std::min(hi, bound_hi);
    MPI_Aint offset = (local_lo - bound_lo);
    int count = (int(local_hi - local_lo));
    if (option == RMAType::get)
      MPI_Rget(curr_buf, count, MPI_DOUBLE, i, offset, count, MPI_DOUBLE, m_win, &requests[i - p_lo]);
    else if (option == RMAType::put)
      MPI_Rput(curr_buf, count, MPI_DOUBLE, i, offset, count, MPI_DOUBLE, m_win, &requests[i - p_lo]);
    else if (option == RMAType::acc)
      MPI_Raccumulate(curr_buf, count, MPI_DOUBLE, i, offset, count, MPI_DOUBLE, MPI_SUM, m_win, &requests[i - p_lo]);
    curr_buf += count;
  }
  MPI_Waitall(requests.size(), requests.data(), MPI_STATUSES_IGNORE);
}

template <typename T>
void DistrArrayMPI3<T>::get(index_type lo, index_type hi, value_type* buf) const {
  const_cast<DistrArrayMPI3<T>*>(this)->_get_put(lo, hi, buf, RMAType::get);
}

template <typename T>
std::vector<typename DistrArrayMPI3<T>::value_type> DistrArrayMPI3<T>::get(index_type lo, index_type hi) const {
  if (lo >= hi)
    return {};
  auto val = std::vector<value_type>(hi - lo);
  get(lo, hi, val.data());
  return val;
}

template <typename T>
void DistrArrayMPI3<T>::put(index_type lo, index_type hi, const value_type* data) { _get_put(lo, hi, data, RMAType::put); }

template <typename T>
std::vector<typename DistrArrayMPI3<T>::value_type> DistrArrayMPI3<T>::gather(const std::vector<index_type>& indices) const {
  auto data = std::vector<value_type>(indices.size());
  const_cast<DistrArrayMPI3*>(this)->_gather_scatter(indices, data, RMAType::gather);
  return data;
}

template <typename T>
void DistrArrayMPI3<T>::scatter(const std::vector<index_type>& indices, const std::vector<value_type>& data) {
  _gather_scatter(indices, const_cast<std::vector<value_type>&>(data), RMAType::scatter);
}

template <typename T>
void DistrArrayMPI3<T>::scatter_acc(std::vector<index_type>& indices, const std::vector<value_type>& data) {
  _gather_scatter(indices, const_cast<std::vector<value_type>&>(data), RMAType::scatter_acc);
}

template <typename T>
void DistrArrayMPI3<T>::_gather_scatter(const std::vector<index_type>& indices, std::vector<value_type>& data,
                                     RMAType option) {
  if (indices.empty())
    return;
  auto name = std::string{"DistrArrayMPI3::_gather_scatter"};
  if (*std::max_element(indices.begin(), indices.end()) >= this->m_dimension)
    error(name + " out of bounds");
  if (indices.size() > data.size())
    error(name + " data buffer is too small");
  if (empty())
    error(name + " called on an empty array");
  auto requests = std::vector<MPI_Request>(indices.size());
  for (size_t i = 0; i < indices.size(); ++i) {
    int p;
    index_type lo;
    p = m_distribution->cover(indices[i]);
    std::tie(lo, std::ignore) = m_distribution->range(p);
    MPI_Aint offset = indices[i] - lo;
    if (option == RMAType::gather)
      MPI_Rget(&data[i], 1, MPI_DOUBLE, p, offset, 1, MPI_DOUBLE, m_win, &requests[i]);
    else if (option == RMAType::scatter)
      MPI_Rput(&data[i], 1, MPI_DOUBLE, p, offset, 1, MPI_DOUBLE, m_win, &requests[i]);
    else if (option == RMAType::scatter_acc)
      MPI_Raccumulate(&data[i], 1, MPI_DOUBLE, p, offset, 1, MPI_DOUBLE, MPI_SUM, m_win, &requests[i]);
  }
  MPI_Waitall(requests.size(), requests.data(), MPI_STATUSES_IGNORE);
}
template <typename T>
std::vector<typename DistrArrayMPI3<T>::value_type> DistrArrayMPI3<T>::vec() const { return get(0, this->m_dimension); }

template <typename T>
void DistrArrayMPI3<T>::acc(index_type lo, index_type hi, const value_type* data) { _get_put(lo, hi, data, RMAType::acc); }

template <typename T>
void DistrArrayMPI3<T>::error(const std::string& message) const { MPI_Abort(this->m_communicator, 1); }

template <typename T>
const typename DistrArray<T>::Distribution& DistrArrayMPI3<T>::distribution() const {
  if (!m_distribution)
    error("allocate buffer before asking for distribution");
  return *m_distribution;
}

template <typename T>
DistrArrayMPI3<T>::LocalBufferMPI3::LocalBufferMPI3(DistrArrayMPI3<T>& source) {
  if (!source.m_allocated)
    source.error("attempting to access local buffer of empty array");
  int rank;
  MPI_Comm_rank(source.communicator(), &rank);
  index_type hi;
  std::tie(this->m_start, hi) = source.distribution().range(rank);
  this->m_size = hi - this->m_start;
  int flag;
  MPI_Win_get_attr(source.m_win, MPI_WIN_BASE, &this->m_buffer, &flag);
}

} // namespace molpro::linalg::array

#endif // LINEARALGEBRA_SRC_MOLPRO_LINALG_ARRAY_DISTRARRAYMPI3_IMPLEMENTATION_H_
