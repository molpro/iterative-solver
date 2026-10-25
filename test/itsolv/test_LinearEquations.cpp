#include "test.h"

#include <Eigen/Eigenvalues>
#include <complex>
#include <deque>
#include <iomanip>
#include <map>
#include <numeric>
#include <sstream>
#include <string>

#include "vector_types.h"
#include <molpro/linalg/itsolv/CastOptions.h>
#include <molpro/linalg/itsolv/SolverFactory-implementation.h>
#include <molpro/linalg/itsolv/helper.h>

using molpro::linalg::itsolv::CastOptions;
using molpro::linalg::itsolv::LinearEquations;

class Problem_ : public molpro::linalg::itsolv::Problem<Rvector> {
public:
  Problem_(int n, int nroot = 1) {
    this->n = n;
    matrix.resize(n, n);
    rhs.resize(n, nroot);
    expected_solution.resize(n, nroot);
    for (int i = 0; i < n; ++i) {
      for (int j = 0; j < n; ++j)
        matrix(i, j) = i + j + 1;
      matrix(i, i) += 1;
    }
    for (int i = 0; i < nroot; ++i)
      for (int j = 0; j < n; ++j) {
        rhs(j, i) = (i + 1) * (n * (n + 1) / 2 + j * n + 1);
        expected_solution(j, i) = i + 1;
      }
  }
  int n;
  Eigen::MatrixXd matrix;
  Eigen::MatrixXd rhs;
  Eigen::MatrixXd expected_solution;
  void action(const molpro::linalg::itsolv::CVecRef<Rvector>& parameters, const molpro::linalg::itsolv::VecRef<Rvector>& action) const override {
    for (int v = 0; v <parameters.size(); ++v) {
      Eigen::Map<Eigen::VectorXd>(action[v].get().data(), n) =
          matrix * Eigen::Map<const Eigen::VectorXd>(parameters[v].get().data(), n);
//      std::cout << "parameters: " << Eigen::Map<const Eigen::VectorXd>(parameters[v].get().data(), n) << std::endl;
//      std::cout << "action: " << Eigen::Map<const Eigen::VectorXd>(action[v].get().data(), n) << std::endl;
    }
  }
  bool diagonals(Rvector& d) const override {
    for (int i = 0; i < n; ++i)
      d[i] = matrix(i, i);
    return true;
  }

  bool RHS(Rvector& r, unsigned int instance) const override {
    if (instance < rhs.cols()) {
      std::copy(rhs.col(instance).begin(), rhs.col(instance).end(), r.begin());
      return true;
    } else
      return false;
  }
};

#ifndef NOFORTRAN
extern "C" int test_linearequationsf(const double* matrix, const double* rhs, size_t n, size_t np, size_t nroot,
                                     int hermitian, double augmented_hessian);
#endif

TEST(LinearEquation, symmetric_system) {

  size_t n_min = 3, n_max = 33;
  double augmented_hessian = 0;
  for (int n = n_min; n <= n_max; n += 3) {
    for (int nroot = 1; nroot <= n && nroot < 14; nroot++) {
      std::cout << "n=" << n << ", nroot=" << nroot << std::endl;
      Problem_ problem(n, nroot);
//      std::cout << "rhs\n" << problem.rhs << std::endl;
//      std::cout << "expected_solution\n" << problem.expected_solution << std::endl;
//      std::cout << "recreated rhs\n" <<  problem.matrix*problem.expected_solution  << std::endl;
      // ASSERT_EQ(  problem.matrix*problem.expected_solution, problem.rhs);
      auto solver = molpro::linalg::itsolv::create_LinearEquations<Rvector, Qvector, Pvector>();
      for (int root = 0; false and root < nroot; ++root) {
        std::vector<double> r(problem.n);
        for (int i = 0; i < problem.n; ++i)
          r[i] = problem.rhs(root, i);
        solver->add_equations(r);
      }
      std::vector<std::vector<double>> parameters(nroot);
      for (auto& v : parameters)
        v.resize(n);
      std::vector<std::vector<double>> actions(nroot);
      for (auto& v : actions)
        v.resize(n);
      solver->set_convergence_threshold(1e-10);
      solver->solve(parameters, actions, problem, true);
      std::vector<int> roots(nroot);
      std::iota(roots.begin(), roots.end(), 0);
      solver->solution(roots, parameters, actions);
      for (int root = 0; root < nroot; ++root) {
//        std::cout << "Solution\n" << Eigen::Map<const Eigen::VectorXd>(parameters[root].data(), n) << std::endl;
        for (int i = 0; i < n; ++i)
          EXPECT_NEAR(parameters[root][i], problem.expected_solution(i, root), 1e-5);
      }
      //            auto options = set_options(solver, mat.rows(), nroot, 0, hermitian, augmented_hessian);
#ifndef NOFORTRAN
      EXPECT_TRUE(test_linearequationsf(problem.matrix.data(), problem.rhs.data(), n, 0, nroot, true, augmented_hessian) != 0);
#endif
    }
  }
}
// With the augmented Hessian the solution x of each equation satisfies (M - lambda) x = rhs, where (y, t) is the lowest
// eigenvector of [[M, -a rhs], [-a rhs^T, 0]] and x = y / (a t)
TEST(LinearEquation, augmented_hessian) {
  const int n = 10, nroot = 2;
  Problem_ problem(n, nroot);
  for (const double a : {0.01, 0.1, 1.0}) {
    auto solver = molpro::linalg::itsolv::create_LinearEquations<Rvector, Qvector, Pvector>(
        "Davidson", "augmented_hessian=" + std::to_string(a));
    solver->set_convergence_threshold(1e-10);
    std::vector<Rvector> parameters(nroot, Rvector(n)), actions(nroot, Rvector(n));
    ASSERT_TRUE(solver->solve(parameters, actions, problem, true)) << "a=" << a;
    std::vector<int> roots(nroot);
    std::iota(roots.begin(), roots.end(), 0);
    solver->solution(roots, parameters, actions);
    for (int root = 0; root < nroot; ++root) {
      Eigen::MatrixXd augmented = Eigen::MatrixXd::Zero(n + 1, n + 1);
      augmented.topLeftCorner(n, n) = problem.matrix;
      augmented.block(0, n, n, 1) = -a * problem.rhs.col(root);
      augmented.block(n, 0, 1, n) = -a * problem.rhs.col(root).transpose();
      Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(augmented);
      const Eigen::VectorXd lowest = es.eigenvectors().col(0);
      const Eigen::VectorXd expected = lowest.head(n) / (a * lowest(n));
      for (int i = 0; i < n; ++i)
        EXPECT_NEAR(parameters[root][i], expected(i), 1e-6 * expected.norm()) << "a=" << a << ", root=" << root;
    }
  }
}

// Convergence must not depend on the scale of the problem. The residual is measured relative to the right-hand side,
// but multiplying the matrix by a constant divides the preconditioned residual by it, and a short preconditioned
// residual must not be mistaken for a redundant one (issue #640)
TEST(LinearEquation, scale_invariance) {
  const int n = 300, nroot = 2;
  Problem_ problem(n, nroot);
  Eigen::MatrixXd matrix(n, n);
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j)
      matrix(i, j) = i + j;
    matrix(i, i) = 302 * i + 300;
    for (int k = 0; k < nroot; ++k)
      problem.rhs(i, k) = 1.0 / (i + k + 1);
  }
  std::vector<int> iterations;
  for (const double scale : {1e-6, 1.0, 1e6}) {
    problem.matrix = scale * matrix;
    auto solver = molpro::linalg::itsolv::create_LinearEquations<Rvector, Qvector, Pvector>("Davidson");
    solver->set_convergence_threshold(1e-13);
    std::vector<Rvector> parameters(nroot, Rvector(n)), actions(nroot, Rvector(n));
    EXPECT_TRUE(solver->solve(parameters, actions, problem, true)) << "scale=" << scale;
    iterations.push_back(solver->statistics().iterations);
    std::vector<int> roots(nroot);
    std::iota(roots.begin(), roots.end(), 0);
    solver->solution(roots, parameters, actions);
    for (int root = 0; root < nroot; ++root) {
      const Eigen::VectorXd x = Eigen::Map<const Eigen::VectorXd>(parameters[root].data(), n);
      EXPECT_LT((problem.matrix * x - problem.rhs.col(root)).norm(), 1e-12 * problem.rhs.col(root).norm())
          << "scale=" << scale << ", root=" << root;
    }
  }
  EXPECT_EQ(iterations[0], iterations[1]);
  EXPECT_EQ(iterations[2], iterations[1]);
}

// The value threshold applies to the change in chi_ii = <b_i|x_i> between iterations (issue #85). It adds to the
// residual criterion, so a tight value threshold with a loose residual threshold needs more iterations, and confirms
// that chi has settled.
TEST(LinearEquation, value_threshold) {
  const int n = 300, nroot = 2;
  Problem_ problem(n, nroot);
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j)
      problem.matrix(i, j) = i + j;
    problem.matrix(i, i) = 302 * i + 300;
    for (int k = 0; k < nroot; ++k)
      problem.rhs(i, k) = 1.0 / (i + k + 1);
  }
  std::vector<int> iterations;
  for (const double value_threshold : {std::numeric_limits<double>::max(), 1e-8, 1e-12}) {
    auto solver = molpro::linalg::itsolv::create_LinearEquations<Rvector, Qvector, Pvector>("Davidson");
    solver->set_convergence_threshold(1e-3);
    solver->set_convergence_threshold_value(value_threshold);
    std::vector<Rvector> parameters(nroot, Rvector(n)), actions(nroot, Rvector(n));
    EXPECT_TRUE(solver->solve(parameters, actions, problem, true)) << "value threshold " << value_threshold;
    iterations.push_back(solver->statistics().iterations);
    std::vector<int> roots(nroot);
    std::iota(roots.begin(), roots.end(), 0);
    solver->solution(roots, parameters, actions);
    for (int root = 0; root < nroot; ++root) {
      const Eigen::VectorXd x = Eigen::Map<const Eigen::VectorXd>(parameters[root].data(), n);
      const Eigen::VectorXd exact = problem.matrix.partialPivLu().solve(problem.rhs.col(root));
      const double chi_error = std::abs(problem.rhs.col(root).dot(x - exact));
      if (value_threshold < 1)
        EXPECT_LT(chi_error, value_threshold) << "value threshold " << value_threshold << ", root " << root;
    }
  }
  EXPECT_LT(iterations[0], iterations[1]);
  EXPECT_LE(iterations[1], iterations[2]);
}

//! A hermitian, diagonally dominant system in any value type
template <typename scalar>
class HermitianEquations : public molpro::linalg::itsolv::Problem<std::vector<scalar>> {
public:
  using R = std::vector<scalar>;
  static constexpr size_t n = 60;
  size_t nrhs;
  explicit HermitianEquations(size_t nrhs) : nrhs(nrhs) {}
  scalar element(size_t i, size_t j) const {
    if (i == j)
      return scalar(1. + i);
    const double re = 0.1 / (1. + std::abs(double(i) - double(j)));
    if constexpr (molpro::linalg::itsolv::is_complex<scalar>{})
      return scalar(re, i < j ? 0.05 * re : -0.05 * re);
    else
      return scalar(re);
  }

  bool diagonals(R& d) const override {
    for (size_t i = 0; i < n; ++i)
      d[i] = element(i, i);
    return true;
  }

  void action(const molpro::linalg::itsolv::CVecRef<R>& parameters,
              const molpro::linalg::itsolv::VecRef<R>& actions) const override {
    for (size_t k = 0; k < parameters.size(); ++k)
      for (size_t i = 0; i < n; ++i) {
        actions[k].get()[i] = 0;
        for (size_t j = 0; j < n; ++j)
          actions[k].get()[i] += element(i, j) * parameters[k].get()[j];
      }
  }

  bool RHS(R& r, unsigned int instance) const override {
    if (instance >= nrhs)
      return false;
    for (size_t i = 0; i < n; ++i)
      r[i] = scalar(1. / (1. + i + instance));
    return true;
  }
};

template <typename scalar>
class LinearEquations_reset_D : public ::testing::Test {};
using value_types = ::testing::Types<double, std::complex<double>, long double>;
TYPED_TEST_SUITE(LinearEquations_reset_D, value_types);

//! A D-space reset that falls due once all equations have converged leaves the converged subspace alone
TYPED_TEST(LinearEquations_reset_D, not_done_after_convergence) {
  using R = std::vector<TypeParam>;
  for (size_t nrhs = 1; nrhs <= 3; ++nrhs) {
    HermitianEquations<TypeParam> problem(nrhs);
    auto run = [&](const std::string& options) {
      auto solver =
          molpro::linalg::itsolv::create_LinearEquations<R, std::deque<TypeParam>, std::map<size_t, TypeParam>>(
              "Davidson", options);
      solver->set_convergence_threshold(1e-9);
      solver->set_verbosity(molpro::linalg::itsolv::Verbosity::None);
      auto params = std::vector<R>(nrhs, R(problem.n));
      for (size_t i = 0; i < nrhs; ++i)
        params[i][i] = 1; // select() of the diagonal guess is not implemented for complex values
      auto actions = params;
      EXPECT_TRUE(solver->solve(params, actions, problem));
      return solver;
    };
    const auto reference = run("max_size_qspace=3");
    ASSERT_GT(reference->dimensions().nD, 0) << "no D space to reset";
    // the reset falls due in the iteration that converges
    const auto iterations = reference->statistics().iterations;
    const auto reset = run("max_size_qspace=3,reset_D=" + std::to_string(iterations));
    EXPECT_EQ(reset->statistics().iterations, iterations) << "nrhs = " << nrhs;
    EXPECT_EQ(reset->statistics().r_creations, reference->statistics().r_creations) << "nrhs = " << nrhs;
    EXPECT_EQ(reset->dimensions().nD, reference->dimensions().nD) << "nrhs = " << nrhs;
  }
}

//! report() leaves the format of the stream as it was
TEST(LinearEquation, report_keeps_stream_format) {
  using R = std::vector<double>;
  HermitianEquations<double> problem(1);
  auto solver = molpro::linalg::itsolv::create_LinearEquations<R, std::deque<double>, std::map<size_t, double>>();
  solver->set_verbosity(molpro::linalg::itsolv::Verbosity::None);
  R parameters(problem.n), actions(problem.n);
  parameters[0] = 1;
  solver->solve(parameters, actions, problem);
  ASSERT_FALSE(solver->errors().empty());
  std::ostringstream stream;
  stream << std::fixed << std::setprecision(3);
  const auto flags = stream.flags();
  solver->report(stream);
  EXPECT_EQ(stream.flags(), flags);
  EXPECT_EQ(stream.precision(), 3);
}
