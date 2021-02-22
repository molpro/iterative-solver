#ifndef LINEARALGEBRA_SRC_MOLPRO_LINALG_ARRAY_DISTRARRAYDISK_IMPLEMENTATION_H_
#define LINEARALGEBRA_SRC_MOLPRO_LINALG_ARRAY_DISTRARRAYDISK_IMPLEMENTATION_H_

#include "DistrArrayDisk.h"
#include "util.h"
#include "util/Distribution.h"
namespace molpro::linalg::array {
template <typename T>
DistrArrayDisk<T>::DistrArrayDisk(std::unique_ptr<Distribution> distr, MPI_Comm commun)
    : DistrArray<T>(distr->border().second, commun), m_distribution(std::move(distr)) {}

template <typename T>
DistrArrayDisk<T>::DistrArrayDisk() = default;

template <typename T>
DistrArrayDisk<T>::DistrArrayDisk(const DistrArrayDisk& source)
    : DistrArray<T>(source),
      m_distribution(source.m_distribution ? std::make_unique<Distribution>(*source.m_distribution) : nullptr) {
  if (source.m_allocated) {
    DistrArrayDisk<T>::allocate_buffer();
    DistrArray<T>::copy(source);
  }
}

template <typename T>
DistrArrayDisk<T>::DistrArrayDisk(DistrArrayDisk&& source) noexcept
    : DistrArray<T>(source), m_distribution(std::move(source.m_distribution)) {
  using std::swap;
  if (source.m_allocated) {
    swap(m_allocated, source.m_allocated);
    swap(m_view_buffer, source.m_view_buffer);
    swap(m_owned_buffer, source.m_owned_buffer);
  }
}

template <typename T>
DistrArrayDisk<T>::~DistrArrayDisk() = default;

template <typename T>
DistrArrayDisk<T>::LocalBufferDisk::LocalBufferDisk(DistrArrayDisk& source) : m_source{source} {
  int rank = mpi_rank(source.communicator());
  index_type hi;
  std::tie(this->m_start, hi) = source.distribution().range(rank);
  this->m_size = hi - this->m_start;
  if (!source.m_allocated) {
    m_snapshot_buffer.resize(this->m_size);
    this->m_buffer = &m_snapshot_buffer[0];
    m_dump = true;
    source.get(this->start(), this->start() + this->size(), this->m_buffer);
  } else {
    this->m_buffer = source.m_view_buffer.data();
    m_dump = false;
  }
}

template <typename T>
bool DistrArrayDisk<T>::LocalBufferDisk::is_snapshot() { return !m_snapshot_buffer.empty(); }

template <typename T>
DistrArrayDisk<T>::LocalBufferDisk::~LocalBufferDisk() {
  if (m_dump)
    m_source.put(this->start(), this->start() + this->size(), this->m_buffer);
}

template <typename T>
void DistrArrayDisk<T>::allocate_buffer() {
  if (m_allocated)
    return;
  auto rank = mpi_rank(this->communicator());
  index_type lo, hi;
  std::tie(lo, hi) = distribution().range(rank);
  size_t sz = hi - lo;
  if (m_owned_buffer.size() < sz)
    m_owned_buffer.resize(sz);
  m_view_buffer = Span<value_type>(&m_owned_buffer[0], m_owned_buffer.size());
  m_allocated = true;
}

template <typename T>
void DistrArrayDisk<T>::allocate_buffer(Span<value_type> buffer) {
  auto rank = mpi_rank(this->communicator());
  index_type lo, hi;
  std::tie(lo, hi) = distribution().range(rank);
  size_t sz = hi - lo;
  if (buffer.size() < sz)
    error("provided buffer is too small");
  if (m_allocated) {
    std::copy(begin(m_view_buffer), end(m_view_buffer), begin(buffer));
    free_buffer();
  }
  swap(m_view_buffer, buffer);
  m_allocated = true;
  if (!m_owned_buffer.empty()) {
    m_owned_buffer.clear();
    m_owned_buffer.shrink_to_fit();
  }
}

template <typename T>
void DistrArrayDisk<T>::free_buffer() {
  m_view_buffer = Span<value_type>{};
  m_owned_buffer.clear();
  m_owned_buffer.shrink_to_fit();
  m_allocated = false;
}

template <typename T>
void DistrArrayDisk<T>::flush() {
  if (!m_allocated)
    return;
  auto rank = mpi_rank(this->communicator());
  index_type lo, hi;
  std::tie(lo, hi) = distribution().range(rank);
  this->put(lo, hi, m_view_buffer.data());
}

template <typename T>
const typename DistrArrayDisk<T>::Distribution& DistrArrayDisk<T>::distribution() const {
  if (!m_distribution)
    error("allocate buffer before asking for distribution");
  return *m_distribution;
}

template <typename T>
bool DistrArrayDisk<T>::empty() const { return !m_allocated; }

template <typename T>
std::unique_ptr<typename DistrArrayDisk<T>::LocalBuffer> DistrArrayDisk<T>::local_buffer() {
  return std::make_unique<LocalBufferDisk>(*this);
}

template <typename T>
std::unique_ptr<const typename DistrArrayDisk<T>::LocalBuffer> DistrArrayDisk<T>::local_buffer() const {
  auto l = std::make_unique<LocalBufferDisk>(*const_cast<DistrArrayDisk*>(this));
  l->dump() = false;
  return l;
}

template <typename T>
std::unique_ptr<Task<void>> DistrArrayDisk<T>::tput(index_type lo, index_type hi, const value_type* data) {
return std::make_unique<Task<void>>(
Task<void>::create(std::launch::async, [lo, hi, data, this]() { this->put(lo, hi, data); }));
}

template <typename T>
std::unique_ptr<Task<void>> DistrArrayDisk<T>::tget(index_type lo, index_type hi, value_type* buf) {
return std::make_unique<Task<void>>(
Task<void>::create(std::launch::async, [lo, hi, buf, this]() { this->get(lo, hi, buf); }));
}

template <typename T>
std::unique_ptr<Task<std::vector<typename DistrArrayDisk<T>::value_type>>> DistrArrayDisk<T>::tget(index_type lo, index_type hi) {
return std::make_unique<Task<std::vector<value_type>>>(
Task<std::vector<value_type>>::create(std::launch::async, [lo, hi, this]() { return this->get(lo, hi); }));
}

template <typename T>
std::unique_ptr<Task<typename DistrArrayDisk<T>::value_type>> DistrArrayDisk<T>::tat(index_type ind) const {
return std::make_unique<Task<value_type>>(Task<value_type>::create(
    std::launch::async, [ ind, this ]() -> auto { return this->at(ind); }));
}

template <typename T>
std::unique_ptr<Task<void>> DistrArrayDisk<T>::tset(index_type ind, value_type val) {
return std::make_unique<Task<void>>(Task<void>::create(
    std::launch::async, [ ind, val, this ]() -> auto { return this->set(ind, val); }));
}

template <typename T>
std::unique_ptr<Task<void>> DistrArrayDisk<T>::tacc(index_type lo, index_type hi,
const value_type* data) {
return std::make_unique<Task<void>>(Task<void>::create(
    std::launch::async, [ lo, hi, data, this ]() -> auto { return this->acc(lo, hi, data); }));
}

template <typename T>
std::unique_ptr<Task<std::vector<typename DistrArrayDisk<T>::value_type>>>
DistrArrayDisk<T>::tgather(const std::vector<index_type>& indices) const {
  return std::make_unique<Task<std::vector<value_type>>>(Task<std::vector<value_type>>::create(
      std::launch::async, [&indices, this ]() -> auto { return this->gather(indices); }));
}

template <typename T>
std::unique_ptr<Task<void>> DistrArrayDisk<T>::tscatter(const std::vector<index_type>& indices,
                                                     const std::vector<value_type>& data) {
  return std::make_unique<Task<void>>(Task<void>::create(
      std::launch::async, [&indices, &data, this ]() -> auto { return this->scatter(indices, data); }));
}

template <typename T>
std::unique_ptr<Task<void>> DistrArrayDisk<T>::tscatter_acc(std::vector<index_type>& indices,
const std::vector<value_type>& data) {
return std::make_unique<Task<void>>(Task<void>::create(
    std::launch::async, [&indices, &data, this ]() -> auto { return this->scatter_acc(indices, data); }));
}

template <typename T>
std::unique_ptr<Task<std::vector<typename DistrArrayDisk<T>::value_type>>> DistrArrayDisk<T>::tvec() const {
  return std::make_unique<Task<std::vector<value_type>>>(Task<std::vector<value_type>>::create(
      std::launch::async, [this]() -> auto { return this->vec(); }));
}

template <typename T>
std::unique_ptr<Task<std::unique_ptr<typename DistrArray<T>::LocalBuffer>>> DistrArrayDisk<T>::tlocal_buffer() {
  return std::make_unique<Task<std::unique_ptr<typename DistrArray<T>::LocalBuffer>>>(
      Task<std::unique_ptr<typename DistrArray<T>::LocalBuffer>>::create(
          std::launch::async, [this]() -> auto { return this->local_buffer(); }));
}

template <typename T>
std::unique_ptr<Task<std::unique_ptr<const typename DistrArray<T>::LocalBuffer>>> DistrArrayDisk<T>::tlocal_buffer() const {
  return std::make_unique<Task<std::unique_ptr<const typename DistrArray<T>::LocalBuffer>>>(
      Task<std::unique_ptr<const typename DistrArray<T>::LocalBuffer>>::create(
          std::launch::async, [this]() -> auto { return this->local_buffer(); }));
}

} // namespace molpro::linalg::array
#endif // LINEARALGEBRA_SRC_MOLPRO_LINALG_ARRAY_DISTRARRAYDISK_IMPLEMENTATION_H_
