#ifndef LINEARALGEBRA_SRC_MOLPRO_LINALG_ARRAY_DISTRARRAY_IMPLEMENTATION_H_
#define LINEARALGEBRA_SRC_MOLPRO_LINALG_ARRAY_DISTRARRAY_IMPLEMENTATION_H_
#include "DistrArray.h"
#include "util/select_max_dot.h"
#include <algorithm>
#include <functional>
#include <iostream>
#include <numeric>

namespace molpro::linalg::array {

template <typename T>
DistrArray<T>::DistrArray(size_t dimension, MPI_Comm commun) : m_dimension(dimension), m_communicator(commun) {}

template <typename T>
void DistrArray<T>::sync() const {
  MPI_Barrier(m_communicator);
}

template <typename T>
void DistrArray<T>::error(const std::string& message) const {
  std::cerr << message << std::endl;
  MPI_Abort(m_communicator, 1);
}

template <typename T>
bool DistrArray<T>::compatible(const DistrArray& other) const {
  bool result = (m_dimension == other.m_dimension);
  if (m_communicator == other.m_communicator)
    result &= true;
  else if (m_communicator == MPI_COMM_NULL || other.m_communicator == MPI_COMM_NULL)
    result &= false;
  else {
    int comp;
    MPI_Comm_compare(m_communicator, other.m_communicator, &comp);
    result &= (comp == MPI_IDENT || comp == MPI_CONGRUENT);
  }
  return result;
}

template <typename T>
bool DistrArray<T>::empty() const {
  return true;
}

template <typename T>
void DistrArray<T>::zero() {
  fill(0);
}

template <typename T>
void DistrArray<T>::fill(DistrArray<T>::value_type val) {
  if (empty())
    error("DistrArray<T>::fill cannot fill empty array");
  auto lb = local_buffer();
  for (auto& el : *lb)
    el = val;
}

template <typename T>
void DistrArray<T>::axpy(value_type a, const DistrArray<T>& y) {
  auto name = std::string{"Array::axpy"};
  if (!compatible(y))
    error(name + " incompatible arrays");
  if (empty() || y.empty())
    error(name + " cannot use empty arrays");
  if (a == 0)
    return;
  auto loc_x = local_buffer();
  auto loc_y = y.local_buffer();
  if (!loc_x->compatible(*loc_y))
    error(name + " incompatible local buffers");
  if (a == 1)
    for (size_t i = 0; i < loc_x->size(); ++i)
      (*loc_x)[i] += (*loc_y)[i];
  else if (a == -1)
    for (size_t i = 0; i < loc_x->size(); ++i)
      (*loc_x)[i] -= (*loc_y)[i];
  else
    for (size_t i = 0; i < loc_x->size(); ++i)
      (*loc_x)[i] += a * (*loc_y)[i];
}

template <typename T>
void DistrArray<T>::scal(DistrArray<T>::value_type a) {
  auto x = local_buffer();
  for (auto& el : *x)
    el *= a;
}

template <typename T>
void DistrArray<T>::add(const DistrArray& y) {
  return axpy(1, y);
}

template <typename T>
void DistrArray<T>::add(DistrArray<T>::value_type a) {
  auto x = local_buffer();
  for (auto& el : *x)
    el += a;
}

template <typename T>
void DistrArray<T>::sub(const DistrArray& y) {
  return axpy(-1, y);
}

template <typename T>
void DistrArray<T>::sub(DistrArray<T>::value_type a) {
  return add(-a);
}

template <typename T>
void DistrArray<T>::recip() {
  auto x = local_buffer();
  for (auto& el : *x)
    el = 1. / el;
}

template <typename T>
void DistrArray<T>::times(const DistrArray& y) {
  auto name = std::string{"Array::times"};
  if (!compatible(y))
    error(name + " incompatible arrays");
  if (empty() || y.empty())
    error(name + " cannot use empty arrays");
  auto loc_x = local_buffer();
  auto loc_y = y.local_buffer();
  if (!loc_x->compatible(*loc_y))
    error(name + " incompatible local buffers");
  for (size_t i = 0; i < loc_x->size(); ++i)
    (*loc_x)[i] *= (*loc_y)[i];
}

template <typename T>
void DistrArray<T>::times(const DistrArray& y, const DistrArray& z) {
  auto name = std::string{"Array::times"};
  if (!compatible(y))
    error(name + " array y is incompatible");
  if (!compatible(z))
    error(name + " array z is incompatible");
  if (empty() || y.empty() || z.empty())
    error(name + " cannot use empty arrays");
  auto loc_x = local_buffer();
  auto loc_y = y.local_buffer();
  auto loc_z = z.local_buffer();
  if (!loc_x->compatible(*loc_y) || !loc_x->compatible(*loc_z))
    error(name + " incompatible local buffers");
  for (size_t i = 0; i < loc_x->size(); ++i)
    (*loc_x)[i] = (*loc_y)[i] * (*loc_z)[i];
}

template <typename T>
typename DistrArray<T>::value_type DistrArray<T>::dot(const DistrArray& y) const {
  auto name = std::string{"Array::dot"};
  if (!compatible(y))
    error(name + " array x is incompatible");
  if (empty() || y.empty())
    error(name + " calling dot on empty arrays");
  auto loc_x = local_buffer();
  auto loc_y = y.local_buffer();
  if (!loc_x->compatible(*loc_y))
    error(name + " incompatible local buffers");
  auto a = std::inner_product(begin(*loc_x), end(*loc_x), begin(*loc_y), (value_type)0);
  MPI_Allreduce(MPI_IN_PLACE, &a, 1, MPI_DOUBLE, MPI_SUM, communicator());
  return a;
}

template <typename T>
void DistrArray<T>::_divide(const DistrArray& y, const DistrArray& z, DistrArray<T>::value_type shift, bool append,
                            bool negative) {
  auto name = std::string{"Array::divide"};
  if (!compatible(y))
    error(name + " array y is incompatible");
  if (!compatible(z))
    error(name + " array z is incompatible");
  if (empty() || y.empty() || z.empty())
    error(name + " calling divide with an empty array");
  auto loc_x = local_buffer();
  auto loc_y = y.local_buffer();
  auto loc_z = z.local_buffer();
  if (!loc_x->compatible(*loc_y) || !loc_x->compatible(*loc_z))
    error(name + " incompatible local buffers");
  if (append) {
    if (negative)
      for (size_t i = 0; i < loc_x->size(); ++i)
        (*loc_x)[i] -= (*loc_y)[i] / ((*loc_z)[i] + shift);
    else
      for (size_t i = 0; i < loc_x->size(); ++i)
        (*loc_x)[i] += (*loc_y)[i] / ((*loc_z)[i] + shift);
  } else {
    if (negative)
      for (size_t i = 0; i < loc_x->size(); ++i)
        (*loc_x)[i] = -(*loc_y)[i] / ((*loc_z)[i] + shift);
    else
      for (size_t i = 0; i < loc_x->size(); ++i)
        (*loc_x)[i] = (*loc_y)[i] / ((*loc_z)[i] + shift);
  }
}

template <typename T>
std::map<size_t, typename DistrArray<T>::value_type> DistrArray<T>::select_max_dot(size_t n,
                                                                                   const DistrArray& y) const {
  if (!compatible(y))
    error("DistrArray<T>::select_max_dot: incompatible arrays");
  if (empty() || y.empty())
    error("DistrArray<T>::select_max_dot: arrays are empty");
  if (n > size() || n > y.size())
    error("DistrArray<T>::select_max_dot: n is too large");
  auto xbuf = local_buffer();
  auto ybuf = y.local_buffer();
  auto local_selection =
      util::select_max_dot<LocalBuffer, LocalBuffer, value_type, value_type>(std::min(n, xbuf->size()), *xbuf, *ybuf);
  auto shifted_local_selection = decltype(local_selection)();
  for (auto& el : local_selection)
    shifted_local_selection.emplace(xbuf->start() + el.first, el.second);
  return util::select_max_dot_broadcast(n, shifted_local_selection, communicator());
}

template <typename T>
std::map<size_t, typename DistrArray<T>::value_type>
DistrArray<T>::select_max_dot(size_t n, const DistrArray<T>::SparseArray& y) const {
  auto name = std::string("DistrArray<T>::select_max_dot:");
  if (empty())
    error(name + " array is empty");
  if (size() < y.rbegin()->first + 1)
    error(name + " sparse array x is too large");
  if (empty() || y.empty())
    error(name + " arrays are empty");
  if (n > size() || n > y.size())
    error(" n is too large");
  auto xbuf = local_buffer();
  auto local_selection = util::select_max_dot_iter_sparse<LocalBuffer, SparseArray, value_type, value_type>(
      std::min(n, xbuf->size()), *xbuf, y);
  auto shifted_local_selection = decltype(local_selection)();
  for (auto& el : local_selection)
    shifted_local_selection.emplace(xbuf->start() + el.first, el.second);
  return util::select_max_dot_broadcast(n, shifted_local_selection, communicator());
}

namespace util {
template <class Compare, class T>
std::list<std::pair<typename DistrArray<T>::index_type, typename DistrArray<T>::value_type>>
extrema(const DistrArray<T>& x, int n) {
  if (x.empty())
    return {};
  auto buffer = x.local_buffer();
  auto length = buffer->size();
  auto nmin = length > n ? n : length;
  auto loc_extrema = std::list<std::pair<typename DistrArray<T>::index_type, T>>();
  for (size_t i = 0; i < nmin; ++i)
    loc_extrema.emplace_back(buffer->start() + i, (*buffer)[i]);
  auto compare = Compare();
  auto compare_pair = [&compare](const auto& p1, const auto& p2) { return compare(p1.second, p2.second); };
  for (size_t i = nmin; i < length; ++i) {
    loc_extrema.emplace_back(buffer->start() + i, (*buffer)[i]);
    loc_extrema.sort(compare_pair);
    loc_extrema.pop_back();
  }
  auto indices_loc = std::vector<typename DistrArray<T>::index_type>(n, x.size() + 1);
  auto indices_glob = std::vector<typename DistrArray<T>::index_type>(n);
  auto values_loc = std::vector<T>(n);
  auto values_glob = std::vector<T>(n);
  size_t ind = 0;
  for (auto& it : loc_extrema) {
    indices_loc[ind] = it.first;
    values_loc[ind] = it.second;
    ++ind;
  }
  MPI_Request requests[3];
  int comm_rank, comm_size;
  MPI_Comm_rank(x.communicator(), &comm_rank);
  MPI_Comm_size(x.communicator(), &comm_size);
  // root collects values, does the final sort and sends the result back
  if (sizeof(T) != sizeof(double))
    throw std::logic_error("Incomplete implementation");
  if (comm_rank == 0) {
    auto ntot = n * comm_size;
    indices_loc.resize(ntot);
    values_loc.resize(ntot);
    auto ndummy = std::vector<int>(comm_size);
    auto d = int(n - nmin);
    MPI_Igather(&d, 1, MPI_INT, ndummy.data(), 1, MPI_INT, 0, x.communicator(), &requests[0]);
    MPI_Igather(MPI_IN_PLACE, n, MPI_UNSIGNED_LONG, indices_loc.data(), n, MPI_UNSIGNED_LONG, 0, x.communicator(),
                &requests[1]);
    MPI_Igather(MPI_IN_PLACE, n, MPI_DOUBLE, values_loc.data(), n, MPI_UNSIGNED_LONG, 0, x.communicator(),
                &requests[2]);
    MPI_Waitall(3, requests, MPI_STATUSES_IGNORE);
    auto tot_dummy = std::accumulate(begin(ndummy), end(ndummy), 0);
    if (tot_dummy != 0) {
      size_t shift = 0;
      for (size_t i = 0, ind = 0; i < comm_size; ++i) {
        for (size_t j = 0; j < n - ndummy[i]; ++j, ++ind) {
          indices_loc[ind] = indices_loc[ind + shift];
          values_loc[ind] = values_loc[ind + shift];
        }
        shift += ndummy[i];
      }
      indices_loc.resize(ntot - tot_dummy);
      values_loc.resize(ntot - tot_dummy);
    }
    std::vector<unsigned int> sort_permutation(indices_loc.size());
    std::iota(begin(sort_permutation), end(sort_permutation), (unsigned int)0);
    std::sort(begin(sort_permutation), end(sort_permutation), [&values_loc, &compare](const auto& i1, const auto& i2) {
      return compare(values_loc[i1], values_loc[i2]);
    });
    for (size_t i = 0; i < n; ++i) {
      auto j = sort_permutation[i];
      indices_glob[i] = indices_loc[j];
      values_glob[i] = values_loc[j];
    }
  } else {
    auto d = int(n - nmin);
    MPI_Igather(&d, 1, MPI_INT, nullptr, 1, MPI_INT, 0, x.communicator(), &requests[0]);
    MPI_Igather(indices_loc.data(), n, MPI_UNSIGNED_LONG, nullptr, n, MPI_UNSIGNED_LONG, 0, x.communicator(),
                &requests[1]);
    MPI_Igather(values_loc.data(), n, MPI_DOUBLE, nullptr, n, MPI_DOUBLE, 0, x.communicator(), &requests[2]);
    MPI_Waitall(3, requests, MPI_STATUSES_IGNORE);
  }
  MPI_Ibcast(indices_glob.data(), n, MPI_UNSIGNED_LONG, 0, x.communicator(), &requests[0]);
  MPI_Ibcast(values_glob.data(), n, MPI_DOUBLE, 0, x.communicator(), &requests[1]);
  MPI_Waitall(2, requests, MPI_STATUSES_IGNORE);
  auto map_extrema = std::list<std::pair<typename DistrArray<T>::index_type, T>>();
  for (size_t i = 0; i < n; ++i)
    map_extrema.emplace_back(indices_glob[i], values_glob[i]);
  return map_extrema;
}
} // namespace util

template <typename T>
std::list<std::pair<typename DistrArray<T>::index_type, typename DistrArray<T>::value_type>>
DistrArray<T>::min_n(int n) const {
  if (empty())
    return {};
  return util::extrema<std::less<DistrArray<T>::value_type>>(*this, n);
}

template <typename T>
std::list<std::pair<typename DistrArray<T>::index_type, typename DistrArray<T>::value_type>>
DistrArray<T>::max_n(int n) const {
  if (empty())
    return {};
  return util::extrema<std::greater<DistrArray<T>::value_type>>(*this, n);
}

template <typename T>
std::list<std::pair<typename DistrArray<T>::index_type, typename DistrArray<T>::value_type>>
DistrArray<T>::min_abs_n(int n) const {
  if (empty())
    return {};
  return util::extrema<util::CompareAbs<DistrArray<T>::value_type, std::less<>>>(*this, n);
}

template <typename T>
std::list<std::pair<typename DistrArray<T>::index_type, typename DistrArray<T>::value_type>>
DistrArray<T>::max_abs_n(int n) const {
  if (empty())
    return {};
  return util::extrema<util::CompareAbs<DistrArray<T>::value_type, std::greater<>>>(*this, n);
}

template <typename T>
std::vector<typename DistrArray<T>::index_type> DistrArray<T>::min_loc_n(int n) const {
  if (empty())
    return {};
  auto min_list = min_abs_n(n);
  auto min_vec = std::vector<index_type>(n);
  std::transform(begin(min_list), end(min_list), begin(min_vec), [](const auto& p) { return p.first; });
  return min_vec;
}

template <typename T>
void DistrArray<T>::copy(const DistrArray<T>& y) {
  auto name = std::string{"Array::copy"};
  if (!compatible(y))
    error(name + " incompatible arrays");
  if (empty() != y.empty())
    error(name + " one of the arrays is empty");
  auto loc_x = local_buffer();
  auto loc_y = y.local_buffer();
  if (!loc_x->compatible(*loc_y))
    error(name + " incompatible local buffers");
  for (size_t i = 0; i < loc_x->size(); ++i)
    (*loc_x)[i] = (*loc_y)[i];
}

template <typename T>
void DistrArray<T>::copy_patch(const DistrArray<T>& y, DistrArray<T>::index_type start, DistrArray<T>::index_type end) {
  auto name = std::string{"Array::copy_patch"};
  if (!compatible(y))
    error(name + " incompatible arrays");
  if (empty() != y.empty())
    error(name + " one of the arrays is empty");
  auto loc_x = local_buffer();
  auto loc_y = y.local_buffer();
  if (!loc_x->compatible(*loc_y))
    error(name + " incompatible local buffers");
  if (start > end)
    return;
  auto s = start <= loc_x->start() ? 0 : start - loc_x->start();
  auto e = end - start + 1 >= loc_x->size() ? loc_x->size() : end - start + 1;
  for (auto i = s; i < e; ++i)
    (*loc_x)[i] = (*loc_y)[i];
}

template <typename T>
typename DistrArray<T>::value_type DistrArray<T>::dot(const SparseArray& y) const {
  auto name = std::string{"Array::dot SparseArray "};
  if (y.empty())
    return 0;
  if (empty())
    error(name + " calling dot on empty arrays");
  if (size() < y.rbegin()->first + 1)
    error(name + " sparse array x is incompatible");
  auto loc_x = local_buffer();
  value_type res = 0;
  if (loc_x->size() > 0) {
    index_type i;
    value_type v;
    for (auto it = y.lower_bound(loc_x->start()); it != y.upper_bound(loc_x->start() + loc_x->size() - 1); ++it) {
      std::tie(i, v) = *it;
      res += (*loc_x)[i - loc_x->start()] * v;
    }
  }
  if (sizeof(res) != sizeof(double))
    throw std::logic_error("unimplemented non-double collective");
  MPI_Allreduce(MPI_IN_PLACE, &res, 1, MPI_DOUBLE, MPI_SUM, communicator());
  return res;
}

template <typename T>
void DistrArray<T>::axpy(value_type a, const SparseArray& y) {
  auto name = std::string{"Array::axpy SparseArray"};
  if (a == 0 || y.empty())
    return;
  if (empty())
    error(name + " calling dot on empty arrays");
  if (size() < y.rbegin()->first + 1)
    error(name + " sparse array x is incompatible");
  auto loc_x = local_buffer();
  if (loc_x->size() > 0) {
    index_type i;
    value_type v;
    if (a == 1)
      for (auto it = y.lower_bound(loc_x->start()); it != y.upper_bound(loc_x->start() + loc_x->size() - 1); ++it) {
        std::tie(i, v) = *it;
        (*loc_x)[i - loc_x->start()] += v;
      }
    else if (a == -1)
      for (auto it = y.lower_bound(loc_x->start()); it != y.upper_bound(loc_x->start() + loc_x->size() - 1); ++it) {
        std::tie(i, v) = *it;
        (*loc_x)[i - loc_x->start()] -= v;
      }
    else
      for (auto it = y.lower_bound(loc_x->start()); it != y.upper_bound(loc_x->start() + loc_x->size() - 1); ++it) {
        std::tie(i, v) = *it;
        (*loc_x)[i - loc_x->start()] += a * v;
      }
  }
}
} // namespace molpro::linalg::array

#endif // LINEARALGEBRA_SRC_MOLPRO_LINALG_ARRAY_DISTRARRAY_IMPLEMENTATION_H_
