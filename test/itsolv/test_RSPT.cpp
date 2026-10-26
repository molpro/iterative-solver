#include "test.h"
#include <Eigen/Dense>
#include <cmath>
#include <complex>
#include <deque>
#include <fstream>
#include <iomanip>
#include <map>
#include <molpro/iostream.h>
#include <numeric>
#include <regex>
#include <sstream>
#include <vector>

#include "vector_types.h"
#include <molpro/linalg/itsolv/CastOptions.h>
#include <molpro/linalg/itsolv/SolverFactory-implementation.h>
#include <molpro/linalg/itsolv/helper.h>

// Find lowest eigensolution of a matrix obtained from an external file
// Storage of vectors in-memory via class Rvector

using molpro::linalg::array::Span;
using molpro::linalg::itsolv::CastOptions;
using molpro::linalg::itsolv::CVecRef;
using molpro::linalg::itsolv::cwrap;
using molpro::linalg::itsolv::cwrap_arg;
using molpro::linalg::itsolv::VecRef;
using molpro::linalg::itsolv::wrap;
using molpro::linalg::itsolv::wrap_arg;

struct RSPT : ::testing::Test {
  using MatrixXdr = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
  using MatrixXdc = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;
  using vectorP = std::vector<scalar>;

  size_t n = 0;
  size_t verbosity = 0;
  MatrixXdc hmat;
  Eigen::VectorXd h0;

  void load_matrix(const std::string &file, double degeneracy_split = 0) {
    std::ifstream f(std::string{"./"} + file + ".hamiltonian");
    f >> n;
    hmat.resize(n, n);
    for (size_t i = 0; i < n; i++)
      for (size_t j = 0; j < n; j++)
        f >> hmat(i, j);
    //    std::cout << "hmat\n" << hmat << std::endl;
    // split degeneracies
    for (size_t i = 0; i < n; i++)
      hmat(i, i) += degeneracy_split * i;
    //    std::cout << "hmat\n" << hmat << std::endl;
    h0.resize(n);
    std::ifstream f0(std::string{"./"} + file + ".h0");
    for (size_t i = 0; i < n; i++)
      f0 >> h0(i);
    //    std::cout << "h0 " << h0.adjoint() << std::endl;
  }

  void action(const Rvector &psx, Rvector &outputs) {
    auto x = Eigen::Map<const Eigen::VectorXd>(psx.data(), psx.size());
    auto r = Eigen::Map<Eigen::VectorXd>(outputs.data(), psx.size());
    r = hmat * x;
  }

  //! The hermitian inner product <a|b>, matching the convention of the array handlers
  template <class scalar>
  scalar dot(const std::vector<scalar> &a, const std::vector<scalar> &b) {
    return std::inner_product(std::begin(a), std::end(a), std::begin(b), scalar(0), std::plus<scalar>{},
                              [](const scalar &x, const scalar &y) { return molpro::linalg::conjugate(x) * y; });
  }

  void update(Rvector &psg) {
    double e0 = 1e50;
    for (size_t i = 0; i < n; i++)
      if (h0(i) < e0)
        e0 = h0(i);
    for (size_t i = 0; i < n; i++) {
      psg[i] = psg[i] / (1e-12 - e0 + h0(i));
    }
  }

  auto initial_guess(Rvector &x) {
    std::vector<double> diagonals;
    diagonals.reserve(n);
    for (size_t i = 0; i < n; i++)
      diagonals.push_back(h0(i));
    x[(std::min_element(diagonals.begin(), diagonals.end()) - diagonals.begin())] = 1; // initial guess
  }

  class testProblem : molpro::linalg::itsolv::Problem<Rvector> {
    void action(const CVecRef<Rvector> &parameters, const VecRef<Rvector> &action) const override {
      for (size_t i = 0; i < parameters.size(); ++i) {
        const Rvector& parameter1 = parameters[i].get();
        Rvector& action1 = action[i].get();
        auto x = Eigen::Map<const Eigen::VectorXd>(parameter1.data(), parameter1.size());
        auto r = Eigen::Map<Eigen::VectorXd>(action1.data(), action1.size());
        r = m_hmat * x;
      }
    }
    const MatrixXdc m_hmat;
    const Eigen::VectorXd m_h0;
  };

  void test_eigen(const std::string &title = "", const int n_working_vectors_max = 0) {
    {
      molpro::cout << "\n\n*** " << title << " by perturbation theory" << std::endl;
      auto solver = molpro::linalg::itsolv::create_LinearEigensystem<Rvector, Qvector, Pvector>("RSPT");
//      int nwork = 1;
      Rvector x(n, 0);
      Rvector g(n, 0);
      initial_guess(x);
      for (auto iter = 1; iter < 10; iter++) {
        action(x, g);
        //        std::cout << "x " << x << std::endl;
        //        std::cout << "g " << g << std::endl;
//        nwork =
        solver->add_vector(x, g);
        //        std::cout << "g after add_vector " << g << std::endl;
        update(g);
        //        std::cout << "g after update " << g << std::endl;
        //        std::cout << "x after update " << x << std::endl;
//        nwork =
        solver->end_iteration(x, g);
        //        std::cout << "g after end_iteration " << g << std::endl;
        //        std::cout << "x after end_iteration " << x << std::endl;
        solver->report();
      }
    }
  }

  enum method_e { BFGS, DIIS };

  template <method_e method = BFGS>
  double test_Hylleraas(const std::string &title = "", const int n_working_vectors_max = 0, bool precondition = true,
                        double expected = double(0)) {
    //        molpro::cout << "\n\n*** " << title << " Hylleraas minimum by " << (method == BFGS ? "BFGS" : "DIIS")
    //                     << std::endl;
    auto optimizer = molpro::linalg::itsolv::create_Optimize<Rvector, Qvector, Pvector>("BFGS");
    auto nonlinear_solver = molpro::linalg::itsolv::create_NonLinearEquations<Rvector, Qvector, Pvector>("DIIS");
    molpro::linalg::itsolv::IterativeSolver<Rvector, Qvector, Pvector> *solver;
    if (method == BFGS)
      solver = dynamic_cast<molpro::linalg::itsolv::IterativeSolver<Rvector, Qvector, Pvector> *>(optimizer.get());
    if (method == DIIS)
      solver =
          dynamic_cast<molpro::linalg::itsolv::IterativeSolver<Rvector, Qvector, Pvector> *>(nonlinear_solver.get());
    //            std::cout << "h\n" << hmat << std::endl;
    //            std::cout << "h0\n" << h0 << std::endl;
    auto ham0 = hmat;
    for (size_t i = 0; i < n; i++) {
      for (size_t j = 0; j < n; j++)
        ham0(i, j) = 0;
      ham0(i, i) = h0(i);
    }
    auto ham1 = hmat - ham0;
    //            std::cout << "ham0\n" << ham0 << std::endl;
    //            std::cout << "ham1\n" << ham1 << std::endl;
    int nwork = 1;
    Rvector x(n, 0);
    Rvector x0(n, 0);
    Rvector g(n, 0);
    auto xv = Eigen::Map<const Eigen::VectorXd>(x.data(), x.size());
    auto rv = Eigen::Map<Eigen::VectorXd>(g.data(), g.size());
    initial_guess(x0);
    auto xv0 = Eigen::Map<const Eigen::VectorXd>(x0.data(), x0.size());
    auto e0 = h0.dot(xv0);
    auto e1 = xv0.dot(ham1 * xv0);
    //            std::cout << "x0=" << xv0.transpose() << std::endl;
    //            std::cout << "e0=" << e0 << std::endl;
    //            std::cout << "e1=" << e1 << std::endl;
    double e2;
    for (auto iter = 1; iter < 20; iter++) {
      e2 = 2 * xv0.dot(ham1 * xv - e1 * xv) + xv.dot(ham0 * xv - e0 * xv);
      rv = ham1 * xv0 - e1 * xv0 + ham0 * xv - e0 * xv;
      //                std::cout << "x " << x << std::endl;
      //                std::cout << "g " << g << std::endl;
      //                std::cout << "e2 " << e2 << std::endl;
      if constexpr (method == BFGS)
        nwork = optimizer->add_vector(x, g, e2 / 2);
      else
        nwork = nonlinear_solver->add_vector(x, g);
      //        std::cout << "g after add_vector " << g << std::endl;
      if (precondition)
        update(g);
      //        std::cout << "g after update " << g << std::endl;
      //        std::cout << "x after update " << x << std::endl;
      nwork = solver->end_iteration(x, g);
      //        std::cout << "g after end_iteration " << g << std::endl;
      //        std::cout << "x after end_iteration " << x << std::endl;
      //            solver->report();
      if (nwork < 1)
        break;
    }
    if (expected != 0) {
      EXPECT_NEAR(e2, expected, 1e-11);
    }
    return e2;
  }
};

TEST_F(RSPT, file_eigen) {
  for (const auto &file : std::vector<std::string>{"he", "bh", "hf"}) {
    load_matrix(file, file == "phenol" ? 0 : 1e-8);
    test_eigen(file);
  }
}

TEST_F(RSPT, file_Hylleraas_BFGS) {
  for (const auto &file : std::vector<std::string>{"he", "bh", "hf"}) {
    load_matrix(file, file == "phenol" ? 0 : 1e-8);
    auto expected = test_Hylleraas(file, 0, true, 0);
    test_Hylleraas<BFGS>(file, 0, false, expected);
    test_Hylleraas<DIIS>(file, 0, true, expected);
    //        test_Hylleraas_BFGS<DIIS>(file,0,false, expected);
  }
}

namespace {
//! Hermitian matrix with H0 its diagonal, to be solved by perturbation theory in any value type
template <typename scalar>
struct PerturbedProblem : molpro::linalg::itsolv::Problem<std::vector<scalar>> {
  using R = std::vector<scalar>;
  size_t n;
  explicit PerturbedProblem(size_t n) : n(n) {}

  scalar matrix(size_t i, size_t j) const {
    if (i == j)
      return scalar(i + 1);
    const double re = 0.02 * (double(i) + double(j));
    if constexpr (molpro::linalg::itsolv::is_complex<scalar>{})
      return scalar(re, 0.01 * (double(j) - double(i)));
    else
      return scalar(re);
  }

  bool diagonals(R &d) const override {
    for (size_t i = 0; i < n; i++)
      d[i] = matrix(i, i);
    return true;
  }

  void action(const CVecRef<R> &parameters, const VecRef<R> &actions) const override {
    for (size_t k = 0; k < parameters.size(); k++) {
      const auto &v = parameters[k].get();
      auto &a = actions[k].get();
      for (size_t i = 0; i < n; i++) {
        a[i] = 0;
        for (size_t j = 0; j < n; j++)
          a[i] += matrix(i, j) * v[j];
      }
    }
  }

  //! Applies the zeroth-order resolvent (H0 - E0)^-1, whatever the shift
  void precondition(const VecRef<R> &residual, const std::vector<scalar> &shift, const R &diagonals) const override {
    for (auto &r : residual)
      for (size_t i = 0; i < n; i++)
        r.get()[i] /= (diagonals[i] - matrix(0, 0) + scalar(1e-12));
  }
};

//! The perturbation series from solve() matches the one from driving add_vector() and end_iteration() by hand
template <typename scalar>
void check_solve_matches_manual_iterations() {
  using R = std::vector<scalar>;
  const size_t n = 12;
  const int niter = 6;
  PerturbedProblem<scalar> problem(n);

  auto manual =
      molpro::linalg::itsolv::create_LinearEigensystem<R, std::deque<scalar>, std::map<size_t, scalar>>("RSPT");
  R x(n, scalar(0)), g(n, scalar(0)), diagonals(n, scalar(0));
  x[0] = 1;
  problem.diagonals(diagonals);
  for (int iter = 0; iter < niter; iter++) {
    problem.action(cwrap_arg(x), wrap_arg(g));
    manual->add_vector(x, g);
    problem.precondition(wrap_arg(g), {}, diagonals);
    manual->end_iteration(x, g);
  }

  auto driven =
      molpro::linalg::itsolv::create_LinearEigensystem<R, std::deque<scalar>, std::map<size_t, scalar>>("RSPT");
  driven->set_max_iter(niter);
  driven->set_convergence_threshold(0);
  driven->set_verbosity(molpro::linalg::itsolv::Verbosity::None);
  R y(n, scalar(0)), h(n, scalar(0));
  y[0] = 1;
  driven->solve(y, h, problem);

  std::ostringstream manual_energies, driven_energies;
  manual->report(manual_energies);
  driven->report(driven_energies);
  EXPECT_EQ(driven_energies.str(), manual_energies.str());
  for (size_t i = 0; i < n; i++)
    EXPECT_NEAR(std::abs(y[i] - x[i]), 0, 1e-12) << "element " << i;
}
} // namespace

TEST(RSPT_solve, matches_manual_iterations_double) { check_solve_matches_manual_iterations<double>(); }
TEST(RSPT_solve, matches_manual_iterations_complex) { check_solve_matches_manual_iterations<std::complex<double>>(); }
TEST(RSPT_solve, matches_manual_iterations_long_double) { check_solve_matches_manual_iterations<long double>(); }
