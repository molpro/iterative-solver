#ifndef LINEARALGEBRA_SRC_MOLPRO_LINALG_ARRAY_UTIL_ITERABLE_LINGALG_H
#define LINEARALGEBRA_SRC_MOLPRO_LINALG_ARRAY_UTIL_ITERABLE_LINGALG_H
/*!
 * @file Implementation of linear algebra operations for iterable containers with explicit instantiation for
 * containers that are used in our implemented arrays.
 */
#include <algorithm>
#include <functional>
#include <map>
#include <numeric>
#include <vector>

#include <molpro/linalg/array/Span.h>

namespace molpro::linalg::array::util {

/*!
 * @brief Scale each element in an iterable container by a constant
 * @param alpha constant to scale by
 * @param x iterable container
 */
template <typename T, class A>
void scal(T alpha, A &x) {
  for (auto &el : x)
    el *= alpha;
}

//! x[i] += alpha
template <typename T, class A>
void add(T alpha, A &x) {
  for (auto &el : x)
    el += alpha;
}

//! x[i] *= y[i]
template <class A, class B>
void times(A &x, const B &y) {
  using std::begin;
  using std::end;
  std::transform(begin(x), end(x), begin(y), begin(x), std::multiplies<>());
}

//! x[i] = y[i] * z[i]
template <class A, class B, class C>
void times(A &x, const B &y, const C &z) {
  using std::begin;
  using std::end;
  std::transform(begin(y), end(y), begin(z), begin(x), std::multiplies<>());
}

/*!
 * @brief Replace each element in an iterable container with a constant
 * @param alpha constant
 * @param x iterable container
 */
template <typename T, class A>
void fill(T alpha, A &x) {
  std::fill(std::begin(x), std::end(x), alpha);
}

/*!
 * @brief apply y += a * x
 * @warning no size checking is performed
 * @param alpha constant to scale x by
 * @param x iterable container
 * @param y iterable container
 */
template <typename T, class AL, class AR>
void axpy(T alpha, const AR &x, AL &y) {
  std::transform(std::begin(y), std::end(y), std::begin(x), std::begin(y),
                 [alpha](auto ely, auto elx) { return ely + alpha * elx; });
}

/*!
 * @brief apply y += a * x, where x is a sparse container
 * @param alpha constant to scale x by
 * @param x sparse container
 * @param y iterable container
 */
template <typename T, class AL, class AR>
void axpy_sparse(T alpha, const AR &x, AL &y) {
  for (const auto [i, v] : x)
    if (i < y.size())
      y[i] += alpha * v;
}

/*!
 * @brief calculate scalar product of two arrays
 * @warning no size checking is performed
 * @param alpha constant to scale x by
 * @param x iterable container
 * @param y iterable container
 */
template <typename T, class AL, class AR>
T dot(const AL &x, const AR &y) {
  return std::inner_product(std::begin(x), std::end(x), std::begin(y), (T)0);
}

/*!
 * @brief calculate scalar product of iterable and sparse array
 * @warning no size checking is performed
 * @param alpha constant to scale x by
 * @param x iterable container
 * @param y sparse container
 */
template <typename T, class AL, class AR>
T dot_sparse(const AL &x, const AR &y) {
  auto tot = T{0};
  for (const auto [i, v] : y)
    if (i < x.size())
      tot += x[i] * v;
  return tot;
}

extern template void scal(double alpha, Span<double> &x);
extern template void add(double alpha, Span<double> &x);
extern template void times(Span<double> &x, const Span<double> &y);
extern template void times(Span<double> &x, const Span<double> &y, const Span<double> &z);
extern template void fill(double alpha, Span<double> &x);
extern template void axpy(double alpha, const Span<double> &x, Span<double> &y);
extern template void axpy_sparse(double alpha, const std::map<size_t, double> &x, Span<double> &y);
extern template double dot(const Span<double> &x, const Span<double> &y);
extern template double dot_sparse(const Span<double> &x, const std::map<size_t, double> &y);

inline void scal(double alpha, std::vector<double> &x) {
  auto s = vector_to_span(x);
  scal(alpha, s);
}

inline void add(double alpha, std::vector<double> &x) {
  auto s = vector_to_span(x);
  add(alpha, s);
}

inline void times(std::vector<double> &x, const std::vector<double> &y) {
  auto s = vector_to_span(x);
  times(s, vector_to_span(y));
}

inline void times(std::vector<double> &x, const std::vector<double> &y, const std::vector<double> &z) {
  auto s = vector_to_span(x);
  times(s, vector_to_span(y), vector_to_span(z));
}

inline void fill(double alpha, std::vector<double> &x) {
  auto s = vector_to_span(x);
  fill(alpha, s);
}

inline void axpy(double alpha, const std::vector<double> &x, std::vector<double> &y) {
  auto sy = vector_to_span(y);
  axpy(alpha, vector_to_span(x), sy);
}

inline void axpy(double alpha, const Span<double> &x, std::vector<double> &y) {
  auto sy = vector_to_span(y);
  axpy(alpha, x, sy);
}

inline void axpy(double alpha, const std::vector<double> &x, Span<double> &y) { axpy(alpha, vector_to_span(x), y); }

inline void axpy_sparse(double alpha, const std::map<size_t, double> &x, std::vector<double> &y) {
  auto sy = vector_to_span(y);
  axpy_sparse(alpha, x, sy);
}

inline double dot(const std::vector<double> &x, const std::vector<double> &y) {
  return dot<double>(vector_to_span(x), vector_to_span(y));
}

inline double dot(const Span<double> &x, const std::vector<double> &y) { return dot<double>(x, vector_to_span(y)); }

inline double dot(const std::vector<double> &x, const Span<double> &y) { return dot<double>(vector_to_span(x), y); }

inline double dot_sparse(const std::vector<double> &x, const std::map<size_t, double> &y) {
  return dot_sparse<double>(vector_to_span(x), y);
}

inline double dot_sparse(const std::map<size_t, double> &x, const std::vector<double> &y) {
  return dot_sparse<double>(vector_to_span(y), x);
}

} // namespace molpro::linalg::array::util
#endif // LINEARALGEBRA_SRC_MOLPRO_LINALG_ARRAY_UTIL_ITERABLE_LINGALG_H
