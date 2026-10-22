#ifndef LINEARALGEBRA_SRC_MOLPRO_LINALG_ITSOLV_SUBSPACE_UTIL_H
#define LINEARALGEBRA_SRC_MOLPRO_LINALG_ITSOLV_SUBSPACE_UTIL_H
#include <Eigen/Dense>
#include <limits>
#include <molpro/linalg/itsolv/ArrayHandlers.h>
#include <molpro/linalg/itsolv/subspace/Matrix.h>
#include <molpro/linalg/itsolv/wrap.h>
#include <molpro/linalg/scalar_traits.h>

#include <cmath>
#include <vector>

namespace molpro::linalg::itsolv::subspace::util {
namespace detail {

//! Checks that type T1 is same as one of T2, Ts ...
template <typename T1, typename T2, typename... Ts>
struct is_one_of {
  static constexpr bool value = std::is_same<T1, T2>::value || is_one_of<T1, Ts...>::value;
};

template <typename T1, typename T2>
struct is_one_of<T1, T2> {
  static constexpr bool value = std::is_same<T1, T2>::value;
};

template <class R, class Q, class Z, class W, bool = is_one_of<Z, R, Q>::value&& is_one_of<W, R, Q>::value,
          bool = std::is_same<R, Z>::value, bool = std::is_same<W, Q>::value>
struct Overlap {};

template <class R, class Q, class Z, class W>
struct Overlap<R, Q, Z, W, true, true, true> {
  using value_type = typename array::ArrayHandler<Z, W>::value_type;
  static Matrix<value_type> _(const CVecRef<R>& left, const CVecRef<Q>& right, array::ArrayHandler<Z, W>& handler) {
    return handler.gemm_inner(left, right);
  }
};

template <class R, class Q, class Z, class W>
struct Overlap<R, Q, Z, W, true, false, false> {
  using value_type = typename array::ArrayHandler<Z, W>::value_type;
  static Matrix<value_type> _(const CVecRef<R>& left, const CVecRef<Q>& right, array::ArrayHandler<Z, W>& handler) {
    // gemm_inner(right, left)(i, j) = <right_i|left_j>, and <left_j|right_i> is its conjugate
    auto mat = handler.gemm_inner(right, left);
    auto m = Matrix<value_type>({left.size(), right.size()});
    conjugate_transpose_copy(m, mat);
    return m;
  }
};

template <class R, class Q, class Z, class W>
constexpr bool Z_and_W_are_one_of_R_and_Q = detail::is_one_of<Z, R, Q>::value&& detail::is_one_of<W, R, Q>::value;
} // namespace detail

//! Calculates overlap matrix between left and right vectors
template <class R, class Q, class Z, class W>
auto overlap(const CVecRef<R>& left, const CVecRef<Q>& right, array::ArrayHandler<Z, W>& handler)
    -> std::enable_if_t<detail::Z_and_W_are_one_of_R_and_Q<R, Q, Z, W>,
                        Matrix<typename array::ArrayHandler<Z, W>::value_type>> {
  return detail::Overlap<R, Q, Z, W>::_(left, right, handler);
}

//! Calculates overlap matrix for a parameter set. Matrix is hermitian by construction.
template <class R>
Matrix<typename array::ArrayHandler<R, R>::value_type> overlap(const CVecRef<R>& params,
                                                               array::ArrayHandler<R, R>& handler) {
  auto m = Matrix<typename array::ArrayHandler<R, R>::value_type>({params.size(), params.size()});
  for (size_t i = 0; i < m.rows(); ++i)
    for (size_t j = 0; j <= i; ++j) {
      m(i, j) = handler.dot(params[i], params[j]);
      m(j, i) = conjugate(m(i, j)); // the matrix is hermitian, which for a real element type is symmetric
    }
  return m;
}

/*!
 * @brief The Euclidean norms of the solutions x_i = sum_j c_ij b_j, from their subspace coefficients c (one row per
 * solution) and the overlap matrix S_jk = <b_j|b_k> of the subspace vectors, as sqrt((c S c^dagger)_ii), calculated
 * together with one matrix product
 */
template <typename T>
std::vector<real_type_t<T>> solution_norms(const Matrix<T>& solutions, const Matrix<T>& overlap) {
  using row_major_type = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
  const auto nsol = Eigen::Index(solutions.rows()), dim = Eigen::Index(solutions.cols());
  auto norms = std::vector<real_type_t<T>>(nsol);
  if (nsol == 0 || dim == 0)
    return norms;
  const auto c = Eigen::Map<const row_major_type>(solutions.data().data(), nsol, dim);
  const auto s = Eigen::Map<const row_major_type>(overlap.data().data(), dim, dim);
  const row_major_type cs = c.conjugate() * s;
  using std::abs;
  using std::sqrt;
  for (Eigen::Index i = 0; i < nsol; ++i)
    norms[i] = sqrt(abs(cs.row(i).dot(c.row(i).conjugate())));
  return norms;
}

/*!
 * @brief The subspace coefficients of the solutions scaled to unit Euclidean length, so that their sizes can be
 * compared with thresholds whatever the normalisation of the solutions (e.g. in the metric of a generalised
 * eigenproblem)
 * @param solutions coefficients of the solutions, one row per solution
 * @param overlap overlap matrix of the subspace vectors
 */
template <typename T>
Matrix<T> unit_length_solutions(const Matrix<T>& solutions, const Matrix<T>& overlap) {
  auto scaled = solutions;
  const auto norms = solution_norms(solutions, overlap);
  for (size_t i = 0; i < scaled.rows(); ++i)
    if (norms[i] > 0)
      for (size_t j = 0; j < scaled.cols(); ++j)
        scaled(i, j) /= T(norms[i]);
  return scaled;
}

//! Replaces a matrix by its hermitian part, which for a real element type is its symmetric part
template <typename T>
void matrix_symmetrize(Matrix<T>& mat) {
  assert(mat.rows() == mat.cols() && "must be a square matrix");
  for (size_t i = 0; i < mat.rows(); ++i)
    for (size_t j = 0; j < i; ++j) {
      mat(i, j) = T(0.5) * (mat(i, j) + conjugate(mat(j, i)));
      mat(j, i) = conjugate(mat(i, j));
    }
}

//! Return maximum element in a matrix along specified rows and columns
template <typename T>
typename Matrix<T>::coord_type max_element_index(const std::list<size_t>& rows, const std::list<size_t>& cols,
                                                 const Matrix<T>& mat) {
  auto max_el = std::numeric_limits<T>::lowest();
  auto ind = typename Matrix<T>::coord_type{0, 0};
  for (auto i : rows) {
    for (auto j : cols) {
      if (mat(i, j) > max_el) {
        max_el = mat(i, j);
        ind = {i, j};
      }
    }
  }
  return ind;
}

/*!
 * @brief Returns order of rows in a matrix slice that brings it closest to identity.
 *
 * @code{.cpp}
 * auto order = eye_order(mat);
 * for( size_t i = 0; i < n_rows; ++i){
 *   mat_new.row(i) = mat.row(order[i]);
 * }
 *
 * @endcode
 * @tparam Slice Slice type
 * @param mat  square matrix
 */
template <typename Slice>
std::vector<size_t> eye_order(const Slice& mat) {
  auto dim = mat.dimensions();
  auto rows = std::list<size_t>{};
  auto cols = std::list<size_t>{};
  for (size_t i = 0; i < dim.first; ++i) {
    rows.emplace_back(i);
    cols.emplace_back(i);
  }
  auto order = std::vector<size_t>(dim.first);
  size_t i, j;
  while (!rows.empty() && !cols.empty()) {
    std::tie(i, j) = max_element_index(rows, cols, mat);
    order.at(j) = i;
    auto it_row = std::find(begin(rows), end(rows), i);
    auto it_col = std::find(begin(cols), end(cols), j);
    rows.erase(it_row);
    cols.erase(it_col);
  }
  return order;
}

} // namespace molpro::linalg::itsolv::subspace::util

#endif // LINEARALGEBRA_SRC_MOLPRO_LINALG_ITSOLV_SUBSPACE_UTIL_H
