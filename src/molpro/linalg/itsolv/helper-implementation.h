#ifndef LINEARALGEBRA_SRC_MOLPRO_LINALG_ITERATIVESOLVER_HELPER_IMPLEMENTATION_H_
#define LINEARALGEBRA_SRC_MOLPRO_LINALG_ITERATIVESOLVER_HELPER_IMPLEMENTATION_H_
#include <Eigen/Dense>

#include <molpro/Profiler.h>
#include <molpro/lapacke.h>
#include <molpro/linalg/itsolv/helper-dispatch.h>
#include <molpro/linalg/itsolv/helper.h>

#include "Logger.h"
#include "subspace/Matrix.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <complex>
#include <cstddef>
#include <iomanip>
#include <list>
#include <numeric>
#include <span>
#include <type_traits>

namespace molpro::linalg::itsolv {

template <typename value_type>
std::list<SVD<value_type>> svd_eigen_jacobi(size_t nrows, size_t ncols, const array::Span<value_type>& m,
                                            real_type_t<value_type> threshold) {
  auto mat = Eigen::Map<const Eigen::Matrix<value_type, Eigen::Dynamic, Eigen::Dynamic>>(m.data(), nrows, ncols);
#if EIGEN_VERSION_AT_LEAST(3, 4, 90)
  // Cast to unsigned int to avoid -Wdeprecated-enum-enum-conversion: the two
  // Eigen flags belong to different enum types but are meant to be ORed here.
  auto svd = Eigen::JacobiSVD<Eigen::Matrix<value_type, Eigen::Dynamic, Eigen::Dynamic>>(
      mat, static_cast<unsigned int>(Eigen::ComputeThinV) |
               static_cast<unsigned int>(Eigen::NoQRPreconditioner));
#else
  auto svd = Eigen::JacobiSVD<Eigen::Matrix<value_type, Eigen::Dynamic, Eigen::Dynamic>, Eigen::NoQRPreconditioner>(
      mat, Eigen::ComputeThinV);
#endif
  auto svd_system = std::list<SVD<value_type>>{};
  auto sv = svd.singularValues();
  for (int i = int(ncols) - 1; i >= 0; --i) {
    if (std::abs(sv(i)) < threshold) { // TODO: This seems to discard values ABOVE the threshold, not below it. it's
      auto t = SVD<value_type>{}; // also not scaling this threshold relative to the max singular value - find out why
      t.value = sv(i);
      t.v.reserve(ncols);
      for (size_t j = 0; j < ncols; ++j) {
        t.v.emplace_back(svd.matrixV()(j, i));
      }
      svd_system.emplace_back(std::move(t));
    }
  }
  return svd_system;
}

template <typename value_type>
std::list<SVD<value_type>> svd_eigen_bdcsvd(size_t nrows, size_t ncols, const array::Span<value_type>& m,
                                            real_type_t<value_type> threshold) {
  auto mat = Eigen::Map<const Eigen::Matrix<value_type, Eigen::Dynamic, Eigen::Dynamic>>(m.data(), nrows, ncols);
  auto svd = Eigen::BDCSVD<Eigen::Matrix<value_type, Eigen::Dynamic, Eigen::Dynamic>>(mat, Eigen::ComputeThinV);
  auto svd_system = std::list<SVD<value_type>>{};
  auto sv = svd.singularValues();
  for (int i = int(ncols) - 1; i >= 0; --i) {
    if (std::abs(sv(i)) < threshold) {
      auto t = SVD<value_type>{};
      t.value = sv(i);
      t.v.reserve(ncols);
      for (size_t j = 0; j < ncols; ++j) {
        t.v.emplace_back(svd.matrixV()(j, i));
      }
      svd_system.emplace_back(std::move(t));
    }
  }
  return svd_system;
}

#ifdef HAVE_LAPACKE
/*!
 * @brief Singular value decomposition through LAPACK's divide-and-conquer driver (?gesdd).
 *
 * Only instantiable for the scalar types LAPACK provides kernels for; @see has_lapack_kernel. The
 * precision-generic path is svd_eigen_jacobi(), which svd_system() uses by default.
 */
template <typename value_type>
std::list<SVD<value_type>> svd_lapacke_gesdd(size_t nrows, size_t ncols, const array::Span<value_type>& mat,
                                             real_type_t<value_type> threshold) {
  static_assert(has_lapack_kernel_v<value_type>, "LAPACK has no ?gesdd kernel for this scalar type");
  const size_t sdim = std::min(nrows, ncols);
  // ?gesdd destroys its input, so hand it a copy
  std::vector<value_type> a(mat.begin(), mat.end());
  std::vector<real_type_t<value_type>> sv(sdim);
  std::vector<value_type> u(nrows * nrows), v(ncols * ncols);
  const auto info =
      lapack::gesdd(LAPACK_ROW_MAJOR, 'A', lapack_int(nrows), lapack_int(ncols), a.data(), lapack_int(ncols), sv.data(),
                    u.data(), lapack_int(nrows), v.data(), lapack_int(ncols));
  if (info != 0)
    throw std::runtime_error("?gesdd (singular value decomposition) failed with info = " + std::to_string(info));
  auto svd_system = std::list<SVD<value_type>>{};
  for (int i = int(ncols) - 1; i >= 0; --i) {
    if (std::abs(sv[i]) < threshold) {
      auto t = SVD<value_type>{};
      t.value = sv[i];
      t.v.reserve(ncols);
      for (size_t j = 0; j < ncols; ++j) {
        t.v.emplace_back(v[i * ncols + j]);
      }
      svd_system.emplace_back(std::move(t));
    }
  }
  return svd_system;
}

//! @copydoc svd_lapacke_gesdd ; this one uses the QR iteration driver (?gesvd)
template <typename value_type>
std::list<SVD<value_type>> svd_lapacke_gesvd(size_t nrows, size_t ncols, const array::Span<value_type>& mat,
                                             real_type_t<value_type> threshold) {
  static_assert(has_lapack_kernel_v<value_type>, "LAPACK has no ?gesvd kernel for this scalar type");
  const size_t sdim = std::min(nrows, ncols);
  // ?gesvd destroys its input, so hand it a copy
  std::vector<value_type> a(mat.begin(), mat.end());
  std::vector<real_type_t<value_type>> sv(sdim);
  std::vector<value_type> u(nrows * nrows), v(ncols * ncols);
  std::vector<real_type_t<value_type>> superb(sdim > 0 ? sdim - 1 : 0);
  const auto info = lapack::gesvd(LAPACK_ROW_MAJOR, 'N', 'A', lapack_int(nrows), lapack_int(ncols), a.data(),
                                  lapack_int(ncols), sv.data(), u.data(), lapack_int(nrows), v.data(),
                                  lapack_int(ncols), superb.data());
  if (info != 0)
    throw std::runtime_error("?gesvd (singular value decomposition) failed with info = " + std::to_string(info));
  auto svd_system = std::list<SVD<value_type>>{};
  for (int i = int(ncols) - 1; i >= 0; --i) {
    if (std::abs(sv[i]) < threshold) {
      auto t = SVD<value_type>{};
      t.value = sv[i];
      t.v.reserve(ncols);
      for (size_t j = 0; j < ncols; ++j) {
        t.v.emplace_back(v[i * ncols + j]);
      }
      svd_system.emplace_back(std::move(t));
    }
  }
  return svd_system;
}

#endif

/*!
 * @brief Eigen-decomposition of a real symmetric matrix in double precision.
 *
 * Retained for backwards compatibility; a thin forwarder to the precision-generic
 * eigensolver_hermitian(), which dispatches to LAPACK or to Eigen depending on the scalar type and
 * on what this build found. @see eigensolver_hermitian for the layout contract and return value.
 */
inline int eigensolver_lapacke_dsyev(std::span<const double> matrix, std::span<double> eigenvectors,
                                     std::span<double> eigenvalues, const size_t dimension) {
  return eigensolver_hermitian<double>(matrix, eigenvectors, eigenvalues, dimension);
}

//! @copydoc eigensolver_lapacke_dsyev
inline std::list<SVD<double>> eigensolver_lapacke_dsyev(size_t dimension, std::span<const double> matrix) {
  return eigensolver_hermitian<double>(dimension, matrix);
}

/**
 * Get the rank of some matrix, given a threshold.
 * @param[in] eigenvalues the matrix, as a vector.
 * @param[in] threshold the threshold. Note that this is the normalised threshold, a value between 0 and 1, relative to
 * the largest element in the matrix.
 * \returns the rank. For an empty matrix, returns 0.
 */
template <typename value_type>
size_t get_rank(std::span<const value_type> eigenvalues, value_type threshold) {
  if (eigenvalues.size() == 0) {
    return 0;
  }
  value_type max = *max_element(eigenvalues.begin(), eigenvalues.end());
  value_type threshold_scaled = threshold * max;
  size_t count =
      std::count_if(eigenvalues.begin(), eigenvalues.end(), [&](auto const& val) { return val >= threshold_scaled; });
  return count;
}

/**
 * Get the rank of some matrix, given a threshold.
 * @param[in] svd_system a std::list containing SVD objects, such as those created by itsolv::svd_system
 * @param[in] threshold the threshold. Note that this is the normalised threshold, a value between 0 and 1, relative to
 * the largest element in the matrix.
 * \returns the rank. For an empty matrix, returns 0.
 */
template <typename value_type>
size_t get_rank(std::list<SVD<value_type>> svd_system, value_type threshold) {
  // compute max
  value_type max_value = 0;
  typename std::list<SVD<value_type>>::iterator it;
  for (it = svd_system.begin(); it != svd_system.end(); it++) {
    if (it->value > max_value) {
      max_value = it->value;
    }
  }
  // scale threshold
  value_type threshold_scaled = threshold * max_value;

  size_t rank = 0;
  // get rank
  for (it = svd_system.begin(); it != svd_system.end(); it++) {
    if (it->value > threshold_scaled) {
      rank += 1;
    }
  }
  return rank;
}

template <typename value_type, typename std::enable_if_t<!is_complex<value_type>{}, std::nullptr_t>>
std::list<SVD<value_type>> svd_system(size_t nrows, size_t ncols, const array::Span<value_type>& m,
                                      real_type_t<value_type> threshold, bool hermitian, bool reduce_to_rank) {
  std::list<SVD<value_type>> svds;
  assert(m.size() == nrows * ncols);
  if (m.empty())
    return {};
  if (hermitian) {
    assert(nrows == ncols);
    // dispatches to ?syev/?heev or to Eigen::SelfAdjointEigenSolver depending on the scalar type
    svds = eigensolver_hermitian<value_type>(nrows, std::span<const value_type>{m.data(), m.size()});
    for (auto s = svds.begin(); s != svds.end();)
      if (s->value > threshold)
        s = svds.erase(s);
      else
        ++s;
  } else {
    // The general (non-hermitian) decomposition goes through Eigen for every scalar type: the
    // subspace matrices are small, and ?gesdd/?gesvd (svd_lapacke_gesdd/svd_lapacke_gesvd above)
    // offer nothing here that would justify differing results between precisions.
    svds = svd_eigen_jacobi<value_type>(nrows, ncols, m, threshold);
    // return svd_eigen_bdcsvd<value_type>(nrows, ncols, m, threshold);
  }

  // reduce to rank
  if (reduce_to_rank) {
    int rank = get_rank(svds, threshold);
    for (int i = ncols; i > rank; i--) {
      svds.pop_back();
    }
  }
  return svds;
}

template <typename value_type, typename std::enable_if_t<is_complex<value_type>{}, int>>
std::list<SVD<value_type>> svd_system(size_t nrows, size_t ncols, const array::Span<value_type>& m,
                                      real_type_t<value_type> threshold, bool hermitian, bool reduce_to_rank) {
  assert(false); // Complex not implemented here
  return {};
}

template <typename value_type>
void printMatrix(const std::vector<value_type>& m, size_t rows, size_t cols, std::string title, std::ostream& s) {
  s << title << "\n"
    << Eigen::Map<const Eigen::Matrix<value_type, Eigen::Dynamic, Eigen::Dynamic>>(m.data(), rows, cols) << std::endl;
}

template <typename value_type, typename std::enable_if_t<is_complex<value_type>{}, int>>
void eigenproblem(std::vector<value_type>& eigenvectors, std::vector<value_type>& eigenvalues,
                  const std::vector<value_type>& matrix, const std::vector<value_type>& metric, size_t dimension,
                  bool hermitian, real_type_t<value_type> svdThreshold, int verbosity) {
  assert(false); // Complex not implemented here
}

template <typename value_type, typename std::enable_if_t<!is_complex<value_type>{}, std::nullptr_t>>
void eigenproblem(std::vector<value_type>& eigenvectors, std::vector<value_type>& eigenvalues,
                  const std::vector<value_type>& matrix, const std::vector<value_type>& metric, size_t dimension,
                  bool hermitian, real_type_t<value_type> svdThreshold, int verbosity,
                  std::vector<std::pair<std::size_t, value_type>>* imag_eval_parts) {
  // let ADL pick up the overloads of extended- and arbitrary-precision scalar types
  using std::abs;
  using std::sqrt;
  // Tolerances calibrated for double precision, rescaled to the precision actually in use
  const value_type zero_tol = precision_scaled<value_type>(1e-10);                   // a quantity that ought to vanish
  const value_type null_eigenvalue_tol = precision_scaled<value_type>(1e-12);        // an eigenvalue that vanishes
  const value_type null_metric_eigenvalue_tol = precision_scaled<value_type>(1e-14); // a singular metric direction
  using MatrixT = Eigen::Matrix<value_type, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;
  using ComplexMatrixT = Eigen::Matrix<std::complex<value_type>, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;
  using MatrixRowMajT = Eigen::Matrix<value_type, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
  using VectorT = Eigen::Vector<value_type, Eigen::Dynamic>;
  using ComplexVectorT = Eigen::Vector<std::complex<value_type>, Eigen::Dynamic>;

  auto prof = molpro::Profiler::single();
  prof->start("itsolv::eigenproblem");
  Eigen::Map<const MatrixRowMajT> HrowMajor(
      matrix.data(), dimension, dimension);
  MatrixT H(dimension, dimension);
  H = HrowMajor;
  Eigen::Map<const MatrixT> S(metric.data(), dimension, dimension);
  ComplexMatrixT subspaceEigenvectors;
  ComplexVectorT subspaceEigenvalues;

  // initialisation of variables
  VectorT metricEvals(dimension);
  MatrixT metricEvecs(dimension, dimension);
  int rank = 0;

  // Perform an eigenvalue decomposition of the metric
  // Note: Since the metric must necessarily be hermitian (and due to its real-valuedness in this
  // function therefore symmetric), we can use the hermitian eigensolver for this
  int success = eigensolver_hermitian<value_type>(std::span<const value_type>{metric.data(), metric.size()},
                                                  {metricEvecs.data(), dimension * dimension},
                                                  {metricEvals.data(), dimension}, dimension);
  if (success != 0) {
    throw std::runtime_error("Eigensolver did not converge");
  }
  rank = get_rank<value_type>(std::span<const value_type>{metricEvals.data(), dimension}, svdThreshold);

  if (verbosity > 1 && rank < S.cols())
    molpro::cout << "SVD rank " << rank << " in subspace of dimension " << S.cols() << std::endl;
  if (verbosity > 2 && rank < S.cols())
    molpro::cout << "singular values " << metricEvals.transpose() << std::endl;

  // Transform H into a symmetrically orthogonalized basis via (S^{-1/2})^\dagger H S^{-1/2}
  // taking into account the possibility of rank-deficiency of S (aka: zero SV)
  // Note that since S is hermitian and positive (semi-)definite, its SVD is equal to its
  // eigendecomposition
  auto svmh = metricEvals.tail(rank);
  for (auto k = 0; k < rank; k++) {
    assert(abs(svmh(k)) <= svdThreshold || svmh(k) >= 0); // metric is supposed to be positive (semi-)definite
    svmh(k) = svmh(k) > null_metric_eigenvalue_tol ? 1 / sqrt(svmh(k)) : 0;
  }
  auto Hbar =
      svmh.asDiagonal() * metricEvecs.rightCols(rank).adjoint() * H * metricEvecs.rightCols(rank) * svmh.asDiagonal();

  // Perform an eigendecomposition of the transformed matrix
  Eigen::EigenSolver<MatrixT> s(Hbar);
  subspaceEigenvalues = s.eigenvalues();
  if (s.eigenvalues().imag().norm() < zero_tol) {
    // real eigenvalues
    subspaceEigenvalues = subspaceEigenvalues.real();
    subspaceEigenvectors = s.eigenvectors();
    // complex eigenvectors need to be rotated
    // assume that they come in consecutive pairs
    for (int i = 0; i < subspaceEigenvectors.cols() - 1; i++) {
      if (subspaceEigenvectors.col(i).imag().norm() <= zero_tol) {
          continue;
      }

      const int j = i + 1;
      if (abs(subspaceEigenvalues(i) - subspaceEigenvalues(j)) >= zero_tol or
          subspaceEigenvectors.col(j).imag().norm() <= zero_tol) {
          continue;
      }

      // For a real-valued matrix, eigenvectors can always be chosen to be real. If we have a complex eigenvector,
      // it's complex conjugate must also be an eigenvector with the same eigenvalue. We can combine these two
      // vectors as either u + u^* = 2 Re(u) or i*(u - u^*) = -2 Im(u).
      // In other words, the real and imaginary part of u are the corresponding real-valued eigenvectors.
      subspaceEigenvectors.col(j) = subspaceEigenvectors.col(i).imag() / subspaceEigenvectors.col(i).imag().norm();
      subspaceEigenvectors.col(i) = subspaceEigenvectors.col(i).real() / subspaceEigenvectors.col(i).real().norm();
    }

    // Convert eigenvectors back into original basis (minus singular dimensions)
    subspaceEigenvectors = metricEvecs.rightCols(rank) * svmh.asDiagonal() * subspaceEigenvectors;
  } else {
    // complex eigenvalues
#ifdef __INTEL_COMPILER
    molpro::cout << "Hbar\n" << Hbar << std::endl;
    molpro::cout << "Eigenvalues\n" << s.eigenvalues() << std::endl;
    molpro::cout << "Eigenvectors\n" << s.eigenvectors() << std::endl;
    throw std::runtime_error("Intel compiler does not support working with complex eigen3 entities properly");
#endif

    // Convert eigenvectors back into original basis (minus singular dimensions)
    subspaceEigenvectors = metricEvecs.rightCols(rank) * svmh.asDiagonal() * s.eigenvectors();
  }

  // Determine order of eigenvalues such that they come in non-descending order of their real part
  // (and non-descending order of imaginary part, in case of equal real parts)
  Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic> perm(subspaceEigenvalues.size());
  perm.setIdentity();
  std::ranges::sort(
      perm.indices(),
      [](const std::complex<value_type>& lhs, const std::complex<value_type>& rhs) {
        if (lhs.real() != rhs.real()) {
          return lhs.real() < rhs.real();
        }

        if (abs(lhs.imag()) != abs(rhs.imag())) {
          // This fixes the order of distinct complex eigenvalue pairs that share the same real part
          return abs(lhs.imag()) < abs(rhs.imag());
        }

        // This fixes the order within a complex eigenvalue pair
        return lhs.imag() < rhs.imag();
      },
      [&subspaceEigenvalues](auto idx) { return subspaceEigenvalues[idx]; });

  // Apply determined order to eigenvalues and -vectors
  subspaceEigenvectors = subspaceEigenvectors * perm;
  subspaceEigenvalues = perm.transpose() * subspaceEigenvalues;


  // TODO: Need to address the case of near-zero eigenvalues (as below for non-hermitian case) and clean-up
  //  non-hermitian case

  if (!hermitian) {
    for (auto repeat = 0; repeat < 1; ++repeat)
      for (Eigen::Index k = 0; k < subspaceEigenvectors.cols(); k++) {
        if (abs(subspaceEigenvalues(k)) < null_eigenvalue_tol) {
          // special case of zero eigenvalue -- make some real non-zero vector definitely in the null space
          subspaceEigenvectors.col(k).real() += value_type(0.3256897) * subspaceEigenvectors.col(k).imag();
          subspaceEigenvectors.col(k).imag().setZero();
        }

        auto ovl = subspaceEigenvectors.col(k).dot(S * subspaceEigenvectors.col(k));
        // S is supposed to be positive (semi-)definite implying that ovl must be a non-negative real number
        assert(abs(ovl.imag()) < zero_tol);
        assert(ovl.real() > 0);
        subspaceEigenvectors.col(k) /= sqrt(ovl.real());
      }
  }

  // Fix indeterminate phase of eigenvectors by requiring the max component to be positive
  for (std::size_t i = 0; i < subspaceEigenvectors.cols(); ++i) {
    const auto &col = subspaceEigenvectors.col(i);
    auto it = std::ranges::max_element(col, std::less<>{}, [](auto val) { return abs(val); });
    auto idx = std::distance(col.begin(), it);
    if (subspaceEigenvectors.col(i)[idx].real() < 0) {
      subspaceEigenvectors.col(i) *= -1;
    }
  }

  if (imag_eval_parts) {
    // Complex eigenvalues are tolerable -> process them to be able to represent everything
    // by real-valued vectors
    imag_eval_parts->clear();

    for (Eigen::Index root = 0; root < Hbar.cols(); ++root) {
      if (subspaceEigenvalues(root).imag() == 0) {
        continue;
      }

      // Complex-valued eigenvalues must appear as complex conjugate pairs
      assert(root + 1 < subspaceEigenvalues.size());
      assert(abs(std::conj(subspaceEigenvalues(root)) - subspaceEigenvalues(root + 1)) < zero_tol);

      imag_eval_parts->emplace_back(root, subspaceEigenvalues(root).imag());
      imag_eval_parts->emplace_back(root + 1, -subspaceEigenvalues(root).imag());

      // Set the eigenvalue pair to their real-part only (imaginary part is tracked separately in imag_eval_parts)
      subspaceEigenvalues(root) = subspaceEigenvalues(root + 1) = subspaceEigenvalues(root).real();

      // Pretend the real and imaginary part were separate eigenvectors (this is required in order
      // to represent all data without the need for using complex numbers).
      // However, as the eigenvalues are not degenerate, the real and imaginary parts of the eigenvectors
      // are in fact NOT eigenvectors themselves.
      // If the true eigenvectors are required, they can easily be recovered from the real and imaginary
      // parts we store here.
      subspaceEigenvectors.col(root + 1) = subspaceEigenvectors.col(root).imag();
      subspaceEigenvectors.col(root) = subspaceEigenvectors.col(root).real();

      // Skip the second eigenvalue in the pair of complex conjugate eigenvalues
      ++root;
    }
  }

  if ((subspaceEigenvectors - subspaceEigenvectors.real()).norm() > zero_tol or
      (subspaceEigenvalues - subspaceEigenvalues.real()).norm() > zero_tol) {
    throw std::runtime_error("unexpected complex solution found");
  }

  eigenvectors.resize(dimension * Hbar.cols());
  eigenvalues.resize(Hbar.cols());

  Eigen::Map<MatrixT>(eigenvectors.data(), dimension, Hbar.cols()) =
      subspaceEigenvectors.real();
  Eigen::Map<VectorT> ev(eigenvalues.data(), Hbar.cols());
  ev = subspaceEigenvalues.real();

  prof->stop();
}

template <typename value_type, typename std::enable_if_t<is_complex<value_type>{}, int>>
void solve_LinearEquations(std::vector<value_type>& solution, std::vector<value_type>& eigenvalues,
                           const std::vector<value_type>& matrix, const std::vector<value_type>& metric,
                           const std::vector<value_type>& rhs, const size_t dimension, size_t nroot,
                           real_type_t<value_type> augmented_hessian, real_type_t<value_type> svdThreshold,
                           int verbosity) {
  assert(false); // Complex not implemented here
}

template <typename value_type, typename std::enable_if_t<!is_complex<value_type>{}, std::nullptr_t>>
void solve_LinearEquations(std::vector<value_type>& solution, std::vector<value_type>& eigenvalues,
                           const std::vector<value_type>& matrix, const std::vector<value_type>& metric,
                           const std::vector<value_type>& rhs, const size_t dimension, size_t nroot,
                           real_type_t<value_type> augmented_hessian, real_type_t<value_type> svdThreshold,
                           int verbosity) {
  const Eigen::Index nX = dimension;
  solution.resize(nX * nroot);
  //  std::cout << "augmented_hessian "<<augmented_hessian<<std::endl;
  if (augmented_hessian > 0) { // Augmented hessian
    Eigen::Matrix<value_type, Eigen::Dynamic, Eigen::Dynamic> subspaceMatrix;
    Eigen::Matrix<value_type, Eigen::Dynamic, Eigen::Dynamic> subspaceOverlap;
    subspaceMatrix.conservativeResize(nX + 1, nX + 1);
    subspaceOverlap.conservativeResize(nX + 1, nX + 1);
    // both arrive row-major, as subspace::Matrix stores them and as the straight-solve
    // branch below already reads them
    using row_major_type = Eigen::Matrix<value_type, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    subspaceMatrix.block(0, 0, nX, nX) = Eigen::Map<const row_major_type>(matrix.data(), nX, nX);
    subspaceOverlap.block(0, 0, nX, nX) = Eigen::Map<const row_major_type>(metric.data(), nX, nX);
    eigenvalues.resize(nroot);
    for (size_t root = 0; root < nroot; root++) {
      for (Eigen::Index i = 0; i < nX; i++) {
        subspaceMatrix(i, nX) = subspaceMatrix(nX, i) = -augmented_hessian * rhs[i * nroot + root];
        subspaceOverlap(i, nX) = subspaceOverlap(nX, i) = 0;
      }
      subspaceMatrix(nX, nX) = 0;
      subspaceOverlap(nX, nX) = 1;
      //      std::cout << "subspace augmented hessian subspaceMatrix\n"<<subspaceMatrix<<std::endl;
      //      std::cout << "subspace augmented hessian subspaceOverlap\n"<<subspaceOverlap<<std::endl;

      Eigen::GeneralizedEigenSolver<Eigen::Matrix<value_type, Eigen::Dynamic, Eigen::Dynamic>> s(subspaceMatrix,
                                                                                                 subspaceOverlap);
      auto eval = s.eigenvalues();
      auto evec = s.eigenvectors();
      Eigen::Index imax = 0;
      for (Eigen::Index i = 0; i < nX + 1; i++)
        if (eval(i).real() < eval(imax).real())
          imax = i;
      eigenvalues[root] = eval(imax).real();
      auto Solution = evec.col(imax).real().head(nX) / (augmented_hessian * evec.real()(nX, imax));
      for (auto k = 0; k < nX; k++)
        solution[k + nX * root] = Solution(k);
      //      std::cout << "subspace augmented hessian solution\n"<<Solution<<std::endl;
    }
  } else { // straight solution of linear equations
    Eigen::Map<const Eigen::Matrix<value_type, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> subspaceMatrixR(
        matrix.data(), nX, nX);
    Eigen::Map<const Eigen::Matrix<value_type, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> RHS_R(rhs.data(), nX,
                                                                                                       nroot);
    Eigen::Matrix<value_type, Eigen::Dynamic, Eigen::Dynamic> subspaceMatrix = subspaceMatrixR;
    Eigen::Matrix<value_type, Eigen::Dynamic, Eigen::Dynamic> RHS = RHS_R;
    Eigen::Matrix<value_type, Eigen::Dynamic, Eigen::Dynamic> Solution;
//    std::cout << "solve_LinearEquations RHS_R\n"<<RHS_R<<std::endl;
//    for (size_t i=0; i<RHS_R.cols()*RHS_R.rows(); ++i)
//      std::cout << " "<<RHS_R.data()[i];
//    std::cout << std::endl;
//    std::cout << "solve_LinearEquations RHS\n"<<RHS<<std::endl;
//    for (size_t i=0; i<RHS.cols()*RHS.rows(); ++i)
//      std::cout << " "<<RHS.data()[i];
//    std::cout << std::endl;
    Solution = subspaceMatrix.householderQr().solve(RHS);
    //    std::cout << "subspace linear equations solution\n"<<Solution<<std::endl;
    for (size_t root = 0; root < nroot; root++)
      for (auto k = 0; k < nX; k++)
        solution[k + nX * root] = Solution(k, root);
  }
}

template <typename value_type, typename std::enable_if_t<!is_complex<value_type>{}, std::nullptr_t>>
void solve_DIIS(std::vector<value_type>& solution, const std::vector<value_type>& matrix, const size_t dimension,
                real_type_t<value_type> svdThreshold, int verbosity) {
  // let ADL pick up the overloads of extended- and arbitrary-precision scalar types
  using std::abs;
  using std::isnan;
  using VectorT = Eigen::Vector<value_type, Eigen::Dynamic>;
  using MatrixT = Eigen::Matrix<value_type, Eigen::Dynamic, Eigen::Dynamic>;
  auto nAug = dimension + 1;
  //  auto nQ = dimension - 1;
  solution.resize(dimension);
  //  if (nQ > 0) {
  VectorT Rhs(nAug), Coeffs(nAug);
  MatrixT BAug(nAug, nAug);
  //    Eigen::Matrix<value_type, Eigen::Dynamic, 1> Rhs(nQ), Coeffs(nQ);
  //    Eigen::Matrix<value_type, Eigen::Dynamic, Eigen::Dynamic> B(nQ, nQ);
  //
  Eigen::Map<const Eigen::Matrix<value_type, Eigen::Dynamic, Eigen::Dynamic>> subspaceMatrix(matrix.data(), dimension,
                                                                                             dimension);
  BAug.block(0, 0, dimension, dimension) = subspaceMatrix;
  for (size_t i = 0; i < dimension; ++i) {
    BAug(dimension, i) = BAug(i, dimension) = -1;
    Rhs(i) = 0;
  }
  BAug(dimension, dimension) = 0;
  Rhs(dimension) = -1;
  //
  //        molpro::cout << "BAug:" << std::endl << BAug << std::endl;
  //        molpro::cout << "Rhs:" << std::endl << Rhs << std::endl;

  // invert the system, determine extrapolation coefficients.
  Eigen::JacobiSVD<MatrixT> svd(BAug, Eigen::ComputeThinU | Eigen::ComputeThinV);

  //  std::cout << "svd thresholds " << svdThreshold << "," << svd.singularValues().maxCoeff() << std::endl;
  //  std::cout << "singular values " << svd.singularValues().transpose() << std::endl;
  svd.setThreshold(svdThreshold * svd.singularValues().maxCoeff() * 0);
  //    molpro::cout << "svdThreshold "<<svdThreshold<<std::endl;
  //    molpro::cout << "U\n"<<svd.matrixU()<<std::endl;
  //    molpro::cout << "V\n"<<svd.matrixV()<<std::endl;
  //    molpro::cout << "singularValues\n"<<svd.singularValues()<<std::endl;
  Coeffs = svd.solve(Rhs).head(dimension);
  //  Coeffs = BAug.fullPivHouseholderQr().solve(Rhs);
  //  molpro::cout << "Coeffs "<<Coeffs.transpose()<<std::endl;
  if (verbosity > 1)
    molpro::cout << "Combination of iteration vectors: " << Coeffs.transpose() << std::endl;
  for (size_t k = 0; k < (size_t)Coeffs.rows(); k++) {
    if (isnan(abs(Coeffs(k)))) {
      molpro::cout << "B:" << std::endl << BAug << std::endl;
      molpro::cout << "Rhs:" << std::endl << Rhs << std::endl;
      molpro::cout << "Combination of iteration vectors: " << Coeffs.transpose() << std::endl;
      throw std::overflow_error("NaN detected in DIIS submatrix solution");
    }
    solution[k] = Coeffs(k);
  }
}
} // namespace molpro::linalg::itsolv


namespace molpro::linalg::itsolv::detail {

/*!
 * @brief Deduces a set of parameters that are redundant due to linear dependencies
 *
 * Only the last nR parameters are considered for removal. Linear dependencies are discovered by performing SVD of the
 * overlap matrix.
 *
 * @param overlap overlap matrix of the full subspace
 * @param oR offset to the start of parameter block
 * @param nR number of parameters to consider for removal
 * @param svd_thresh singular value threshold for choosing the null space
 * @param logger logger
 * @return indices of the last nR parameters that are considered redundant.
 */
template <typename value_type, typename value_type_abs>
auto redundant_parameters(const subspace::Matrix<value_type>& overlap, const size_t oR, const size_t nR,
                          const value_type_abs svd_thresh, Logger& logger) {
  auto prof = molpro::Profiler::single();
  prof->start("itsolv::svd_system");
  logger.trace("redundant_parameters()");
  auto redundant_params = std::vector<int>{};
  auto rspace_indices = std::vector<int>(nR);
  std::iota(std::begin(rspace_indices), std::end(rspace_indices), 0);
  auto svd = svd_system(overlap.rows(), overlap.cols(),
                        array::Span(const_cast<value_type*>(overlap.data().data()), overlap.size()), svd_thresh, true);
  prof->stop();
  prof->start("find redundant parameters");
  for (const auto& singular_system : svd) {
    if (!rspace_indices.empty()) {
      auto rspace_contribution = std::vector<value_type_abs>{};
      for (auto i : rspace_indices)
        rspace_contribution.push_back(std::abs(singular_system.v.at(oR + i)));
      auto it_min = std::max_element(std::begin(rspace_contribution), std::end(rspace_contribution));
      auto imin = std::distance(std::begin(rspace_contribution), it_min);
      redundant_params.push_back(rspace_indices[imin]);
      rspace_indices.erase(std::begin(rspace_indices) + imin);
      std::stringstream ss;
      ss << std::setprecision(3) << "redundant parameter found, i = " << redundant_params.back()
         << ", svd.value = " << singular_system.value
         << ", svd.v[i] = " << singular_system.v[oR + redundant_params.back()];
      logger.info(ss.str());
    }
  }
  prof->stop();
  return redundant_params;
}

}

#endif // LINEARALGEBRA_SRC_MOLPRO_LINALG_ITERATIVESOLVER_HELPER_IMPLEMENTATION_H_
