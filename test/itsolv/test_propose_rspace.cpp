#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <molpro/linalg/itsolv/propose_rspace.h>
#include <molpro/linalg/itsolv/wrap.h>

#include <cmath>
#include <complex>
#include <map>
#include <vector>

using molpro::linalg::itsolv::ArrayHandlers;
using molpro::linalg::itsolv::cwrap;
using molpro::linalg::itsolv::Logger;
using molpro::linalg::itsolv::wrap;
using molpro::linalg::itsolv::subspace::Dimensions;
using molpro::linalg::itsolv::subspace::Matrix;

namespace {
// New parameters must come out orthogonal to the span of the existing P+Q+D vectors even when those are not
// orthogonal to each other, as happens when the caller supplies its own initial vectors (issue #640). The existing
// vectors here are nearly parallel, so a single projection leaves overlaps of order epsilon times the condition number
// of their (equilibrated) overlap matrix, about 1e-11; projecting on each vector separately left about 1e-4.
template <typename T>
void check_orthogonal_to_nonorthogonal_space() {
  using R = std::vector<T>;
  using P = std::map<size_t, T>;
  const size_t n = 8, nQ = 3, nR = 2;
  T phase = 1;
  if constexpr (!std::is_same_v<T, double>)
    phase = std::polar(1.0, 0.3);
  // nearly parallel, very differently scaled existing vectors
  std::vector<R> q(nQ, R(n)), r(nR, R(n));
  for (size_t k = 0; k < n; ++k) {
    q[0][k] = T(1.0 + k);
    q[1][k] = 1e3 * (q[0][k] + T(1e-3 * std::cos(double(k))) * phase);
    q[2][k] = 1e-2 * (q[0][k] + T(1e-2 * std::sin(2.0 * k)) * phase);
    for (size_t j = 0; j < nR; ++j)
      r[j][k] = T(std::cos(0.7 * k + j)) + T(0.1 * j) * phase;
  }
  auto handlers = ArrayHandlers<R, R, P>();
  Matrix<T> overlap({nQ, nQ});
  for (size_t i = 0; i < nQ; ++i)
    for (size_t j = 0; j < nQ; ++j)
      overlap(i, j) = handlers.qq().dot(q[i], q[j]);
  const std::vector<P> p;
  const std::vector<R> d;
  Logger logger;
  // <r_j|q_i>, as append_overlap_with_r() provides in propose_rspace()
  const auto rx_overlap = handlers.rq().gemm_inner(cwrap(r), cwrap(q));
  auto wr = wrap(r);
  const auto null_params = molpro::linalg::itsolv::detail::modified_gram_schmidt(
      wr, overlap, rx_overlap, Dimensions(0, nQ, 0), cwrap(p), cwrap(q), cwrap(d), 1e-10, handlers, logger);
  EXPECT_TRUE(null_params.empty());
  for (size_t j = 0; j < nR; ++j) {
    for (size_t i = 0; i < nQ; ++i)
      EXPECT_LT(std::abs(handlers.qq().dot(q[i], r[j])) / std::sqrt(std::abs(handlers.qq().dot(q[i], q[i]))), 1e-10)
          << "q[" << i << "], r[" << j << "]";
    for (size_t k = 0; k < nR; ++k)
      EXPECT_NEAR(std::abs(handlers.rr().dot(r[j], r[k])), j == k ? 1 : 0, 1e-12) << "r[" << j << "], r[" << k << "]";
  }
}
} // namespace

TEST(propose_rspace, modified_gram_schmidt_nonorthogonal_space) { check_orthogonal_to_nonorthogonal_space<double>(); }

TEST(propose_rspace, modified_gram_schmidt_nonorthogonal_space_complex) {
  check_orthogonal_to_nonorthogonal_space<std::complex<double>>();
}
