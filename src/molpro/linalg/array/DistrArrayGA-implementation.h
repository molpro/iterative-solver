#ifndef LINEARALGEBRA_SRC_MOLPRO_LINALG_ARRAY_DISTRARRAYGA_IMPLEMENTATION_H_
#define LINEARALGEBRA_SRC_MOLPRO_LINALG_ARRAY_DISTRARRAYGA_IMPLEMENTATION_H_

#include "DistrArrayGA.h"
#include "util/Distribution.h"

#include <algorithm>
#include <ga-mpi.h>
#include <ga.h>
#include <string>

namespace molpro::linalg::array {
template <typename T>
DistrArrayGA<T>::DistrArrayGA() = default;

template <typename T>
DistrArrayGA<T>::DistrArrayGA(size_t dimension, MPI_Comm comm)
    : DistrArray<T>(dimension, comm), m_comm_rank(get_communicator_rank(comm)),
      m_comm_size(get_communicator_size(comm)) {}

template <typename T>
std::map<MPI_Comm, int> DistrArrayGA<T>::_ga_pgroups{};

template <typename T>
DistrArrayGA<T>::DistrArrayGA(const DistrArrayGA &source)
    : DistrArray<T>(source.m_dimension, source.m_communicator), m_comm_rank(source.m_comm_rank),
      m_comm_size(source.m_comm_size), m_ga_chunk(source.m_ga_chunk),
      m_distribution(source.m_distribution ? std::make_unique<Distribution>(*source.m_distribution) : nullptr) {
  if (!source.empty()) {
    DistrArrayGA::allocate_buffer();
    DistrArrayGA::copy(source);
  }
}

template <typename T>
DistrArrayGA<T>::DistrArrayGA(DistrArrayGA &&source) noexcept
    : DistrArray<T>(source.m_dimension, source.m_communicator), m_comm_rank(source.m_comm_rank),
      m_comm_size(source.m_comm_size), m_ga_handle(source.m_ga_handle), m_ga_pgroup(source.m_ga_pgroup),
      m_ga_chunk(source.m_ga_chunk), m_ga_allocated(source.m_ga_allocated),
      m_distribution(std::move(source.m_distribution)) {
  source.m_ga_allocated = false;
}

template <typename T>
DistrArrayGA<T> &DistrArrayGA<T>::operator=(const DistrArrayGA<T> &source) {
  if (this == &source)
    return *this;
  if (source.empty() || empty() || !this->compatible(source)) {
    free_buffer();
    DistrArrayGA<T> t{source};
    swap(*this, t);
  } else {
    allocate_buffer();
    this->copy(source);
  }
  return *this;
}

template <typename T>
DistrArrayGA<T> &DistrArrayGA<T>::operator=(DistrArrayGA<T> &&source) noexcept {
  DistrArrayGA<T> t{std::move(source)};
  swap(*this, t);
  return *this;
}

template <typename T>
void swap(DistrArrayGA<T> &a1, DistrArrayGA<T> &a2) noexcept {
  using std::swap;
  swap(a1.m_dimension, a2.m_dimension);
  swap(a1.m_communicator, a2.m_communicator);
  swap(a1.m_distribution, a2.m_distribution);
  swap(a1.m_comm_rank, a2.m_comm_rank);
  swap(a1.m_comm_size, a2.m_comm_size);
  swap(a1.m_ga_handle, a2.m_ga_handle);
  swap(a1.m_ga_chunk, a2.m_ga_chunk);
  swap(a1.m_ga_pgroup, a2.m_ga_pgroup);
  swap(a1.m_ga_allocated, a2.m_ga_allocated);
}

template <typename T>
DistrArrayGA<T>::~DistrArrayGA() {
  if (!DistrArrayGA<T>::empty())
    GA_Destroy(m_ga_handle);
}

template <typename T>
bool DistrArrayGA<T>::empty() const {
  return !m_ga_allocated;
}

template <typename T>
void DistrArrayGA<T>::error(const std::string &message) const {
  GA_Error(const_cast<char *>(message.c_str()), 1);
}

template <typename T>
void DistrArrayGA<T>::allocate_buffer() {
  if (!empty())
    return;
  auto name = std::string{"DistrArrayGA::allocate_buffer"};
  if (_ga_pgroups.find(this->m_communicator) == _ga_pgroups.end()) {
    //  global processor ranks
    int loc_size, glob_size, glob_rank;
    MPI_Comm_size(this->m_communicator, &loc_size);
    MPI_Comm_size(GA_MPI_Comm(), &glob_size);
    MPI_Comm_rank(GA_MPI_Comm(), &glob_rank);
    auto glob_ranks = std::vector<int>(loc_size);
    MPI_Allgather(&glob_rank, 1, MPI_INT, glob_ranks.data(), 1, MPI_INT, this->m_communicator);
    // create new GA processor group
    m_ga_pgroup = GA_Pgroup_create(glob_ranks.data(), loc_size);
    _ga_pgroups[this->m_communicator] = m_ga_pgroup;
  } else
    m_ga_pgroup = _ga_pgroups[this->m_communicator];
  m_ga_handle = NGA_Create_handle();
  NGA_Set_pgroup(m_ga_handle, m_ga_pgroup);
  auto dims = (int)this->m_dimension;
  NGA_Set_data(m_ga_handle, 1, &dims, C_DBL);
  NGA_Set_array_name(m_ga_handle, (char *)"Array");
  NGA_Set_chunk(m_ga_handle, &m_ga_chunk);
  auto succ = GA_Allocate(m_ga_handle);
  if (!succ)
    error("Failed to allocate");
  m_ga_allocated = true;
  m_distribution = std::make_unique<Distribution>(make_distribution());
}

template <typename T>
void DistrArrayGA<T>::free_buffer() {
  if (!DistrArrayGA<T>::empty()) {
    GA_Destroy(m_ga_handle);
    m_ga_allocated = false;
  }
}

template <typename T>
void DistrArrayGA<T>::sync() const {
  auto name = std::string{"DistrArrayGA::sync"};
  GA_Pgroup_sync(m_ga_pgroup);
}

template <typename T>
DistrArrayGA<T>::LocalBufferGA::LocalBufferGA(DistrArrayGA &source) : DistrArray<T>::LocalBuffer{} {
  if (source.empty())
    return;
  int lo, hi, ld{0};
  NGA_Distribution(source.m_ga_handle, source.m_comm_rank, &lo, &hi);
  NGA_Access(source.m_ga_handle, &lo, &hi, &this->m_buffer, &ld);
  if (this->m_buffer == nullptr)
    source.error("Array::LocalBuffer::LocalBuffer() Failed to get local buffer");
  this->m_start = lo;
  this->m_size = hi - lo + 1;
}

template <typename T>
std::unique_ptr<const typename DistrArray<T>::LocalBuffer> DistrArrayGA<T>::local_buffer() const {
  return std::make_unique<const DistrArrayGA::LocalBufferGA>(*const_cast<DistrArrayGA *>(this));
}

template <typename T>
std::unique_ptr<typename DistrArray<T>::LocalBuffer> DistrArrayGA<T>::local_buffer() {
  return std::make_unique<DistrArrayGA<T>::LocalBufferGA>(*this);
}

template <typename T>
void DistrArrayGA<T>::check_ga_ind_overlow(index_type ind) const {
  if (ind > std::numeric_limits<int>::max())
    error("DistrArrayGA::check_ga_ind_overlow specified range will overflow. GA only uses int type indices.");
}

template <typename T>
typename DistrArrayGA<T>::value_type DistrArrayGA<T>::at(index_type ind) const {
  auto name = std::string{"DistrArrayGA::at"};
  if (ind >= this->m_dimension)
    error(name + " out of bounds");
  if (empty())
    error(name + " called on empty array");
  check_ga_ind_overlow(ind);
  double buffer;
  int lo = ind, high = ind, ld = 1;
  NGA_Get(m_ga_handle, &lo, &high, &buffer, &ld);
  return buffer;
}

template <typename T>
void DistrArrayGA<T>::set(index_type ind, value_type val) {
  put(ind, ind + 1, &val);
}

template <typename T>
void DistrArrayGA<T>::get(index_type lo, index_type hi, value_type *buf) const {
  if (empty() || lo >= hi)
    return;
  auto name = std::string{"DistrArrayGA::get"};
  check_ga_ind_overlow(lo);
  check_ga_ind_overlow(hi);
  int ld, ilo = lo, ihi = int(hi) - 1;
  NGA_Get(m_ga_handle, &ilo, &ihi, buf, &ld);
}

template <typename T>
std::vector<typename DistrArrayGA<T>::value_type> DistrArrayGA<T>::get(index_type lo, index_type hi) const {
  if (lo >= hi)
    return {};
  auto buf = std::vector<value_type>(hi - lo);
  get(lo, hi, buf.data());
  return buf;
}

template <typename T>
void DistrArrayGA<T>::put(index_type lo, index_type hi, const value_type *data) {
  if (lo >= hi)
    return;
  auto name = std::string{"DistrArrayGA::put"};
  if (empty())
    error(name + " attempting to put data into an empty array");
  check_ga_ind_overlow(lo);
  check_ga_ind_overlow(hi);
  int ld, ilo = lo, ihi = int(hi) - 1;
  NGA_Put(m_ga_handle, &ilo, &ihi, const_cast<value_type *>(data), &ld);
}

template <typename T>
std::vector<typename DistrArrayGA<T>::value_type>
DistrArrayGA<T>::gather(const std::vector<index_type> &indices) const {
  auto name = std::string{"DistrArrayGA::gather"};
  if (empty())
    return {};
  for (auto el : indices)
    check_ga_ind_overlow(el);
  int n = indices.size();
  auto data = std::vector<double>(n);
  auto iind = std::vector<int>(indices.begin(), indices.end());
  int **subsarray = new int *[n];
  for (int i = 0; i < n; ++i)
    subsarray[i] = &(iind.at(i));
  NGA_Gather(m_ga_handle, data.data(), subsarray, n);
  delete[] subsarray;
  return data;
}

template <typename T>
void DistrArrayGA<T>::scatter(const std::vector<index_type> &indices, const std::vector<value_type> &data) {
  auto name = std::string{"DistrArrayGA::scatter"};
  if (empty())
    error(name + " attempting to scatter into an empty array");
  for (auto el : indices)
    check_ga_ind_overlow(el);
  int n = indices.size();
  auto iind = std::vector<int>(indices.begin(), indices.end());
  int **subsarray = new int *[n];
  for (int i = 0; i < n; ++i)
    subsarray[i] = &(iind.at(i));
  NGA_Scatter(m_ga_handle, const_cast<double *>(data.data()), subsarray, n);
  delete[] subsarray;
}

template <typename T>
void DistrArrayGA<T>::scatter_acc(std::vector<index_type> &indices, const std::vector<value_type> &data) {
  auto name = std::string{"DistrArrayGA::scatter_acc"};
  if (empty())
    error(name + " attempting to scatter_acc into an empty array");
  for (auto el : indices)
    check_ga_ind_overlow(el);
  int n = indices.size();
  auto iind = std::vector<int>(indices.begin(), indices.end());
  int **subsarray = new int *[n];
  for (int i = 0; i < n; ++i) {
    subsarray[i] = &(iind.at(i));
  }
  value_type alpha = 1;
  NGA_Scatter_acc(m_ga_handle, const_cast<double *>(data.data()), subsarray, n, &alpha);
  delete[] subsarray;
}

template <typename T>
std::vector<typename DistrArrayGA<T>::value_type> DistrArrayGA<T>::vec() const {
  if (empty())
    return {};
  check_ga_ind_overlow(this->m_dimension);
  auto name = std::string{"DistrArrayGA::vec"};
  std::vector<double> vec(this->m_dimension);
  double *buffer = vec.data();
  int lo = 0, hi = int(this->m_dimension) - 1, ld;
  NGA_Get(m_ga_handle, &lo, &hi, buffer, &ld);
  return vec;
}

template <typename T>
void DistrArrayGA<T>::acc(index_type lo, index_type hi, const value_type *data) {
  auto name = std::string{"DistrArrayGA::acc"};
  if (empty())
    error(name + " attempting to accumulate an empty array");
  check_ga_ind_overlow(lo);
  check_ga_ind_overlow(hi);
  double scaling_constant = 1;
  int ld, ilo = lo, ihi = hi;
  NGA_Acc(m_ga_handle, &ilo, &ihi, const_cast<double *>(data), &ld, &scaling_constant);
}

template <typename T>
const typename DistrArrayGA<T>::Distribution &DistrArrayGA<T>::distribution() const {
  if (!m_distribution)
    error("allocate buffer before asking for distribution");
  return *m_distribution;
}

template <typename T>
typename DistrArrayGA<T>::Distribution DistrArrayGA<T>::make_distribution() const {
  if (!m_ga_allocated)
    return {};
  auto chunk_borders = std::vector<index_type>{0};
  int lo, hi;
  for (size_t rank = 0; rank < get_communicator_size(this->communicator()); ++rank) {
    NGA_Distribution(m_ga_handle, rank, &lo, &hi);
    chunk_borders.push_back(hi);
  }
  return {chunk_borders};
}

} // namespace molpro::linalg::array

#endif // LINEARALGEBRA_SRC_MOLPRO_LINALG_ARRAY_DISTRARRAYGA_IMPLEMENTATION_H_
