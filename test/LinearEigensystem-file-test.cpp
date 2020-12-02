#include "create_solver.h"
#include "test.h"
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <molpro/iostream.h>
#include <numeric>
#include <regex>
#include <vector>

// Find lowest eigensolution of a matrix obtained from an external file
// Storage of vectors in-memory via class Rvector
using scalar = double;
namespace {
size_t n;
std::vector<double> hmat;
std::vector<double> expected_eigenvalues;
} // namespace

void load_matrix(int dimension, const std::string& type = "", double param = 1, bool hermitian = true) {
  n = dimension;
  hmat.resize(n * n);
  hmat.assign(n * n, 1);
  for (int i = 0; i < n; i++)
    hmat[i * (n + 1)] = i * param;
  if (not hermitian)
    for (int i = 0; i < n; i++)
      for (int j = 0; j < i; j++)
        hmat[i * n + j] *= .95;
}
void load_matrix(const std::string& file, double degeneracy_split = 0) {
  std::ifstream f(std::string{"./"} + file + ".hamiltonian");
  f >> n;
  hmat.resize(n * n);
  for (auto i = 0; i < n * n; i++)
    f >> hmat[i];
  // split degeneracies
  for (auto i = 0; i < n; i++)
    hmat[i * (n + 1)] += degeneracy_split * i;
}

using pv = Rvector;

void action(const std::vector<Rvector>& psx, std::vector<Rvector>& outputs) {
  for (size_t k = 0; k < psx.size(); k++) {
    for (size_t i = 0; i < n; i++) {
      outputs[k][i] = 0;
      for (size_t j = 0; j < n; j++) {
        outputs[k][i] += hmat[i + n * j] * psx[k][j];
      }
    }
  }
}
template <class scalar>
scalar dot(const std::vector<scalar>& a, const std::vector<scalar>& b) {
  return std::inner_product(std::begin(a), std::end(a), std::begin(b), scalar(0));
}
std::vector<double> residual(const std::vector<Rvector>& psx, std::vector<Rvector>& outputs) {
  std::vector<double> evals;
  action(psx, outputs);
  for (size_t k = 0; k < psx.size(); k++) {
    auto eval = dot(outputs[k], psx[k]) / dot(psx[k], psx[k]);
    for (size_t i = 0; i < n; i++)
      outputs[k][i] -= eval * psx[k][i];
    evals.push_back(eval);
  }
  return evals;
}

void update(std::vector<Rvector>& psg, std::vector<scalar> shift = std::vector<scalar>()) {
  //  for (size_t k = 0; k < psc.size(); k++) {
  //   molpro::cout << "update root "<<k<<", shift="<<shift[k]<<": ";
  //    for (size_t i = 0; i < n; i++)
  //      molpro::cout <<" "<< -psg[k][i] / (1e-12-shift[k] + matrix(i,i));
  //    molpro::cout<<std::endl;
  //  }
  for (size_t k = 0; k < psg.size() && k < shift.size(); k++)
    for (size_t i = 0; i < n; i++) {
      psg[k][i] = -psg[k][i] / (1e-12 - shift[k] + hmat[i + n * i]);
    }
}

std::vector<double> phase_normalise(const std::vector<double>& in) {
  double phase = 1;
  double maxcomp = 1e-10;
  double norm = 0;
  for (const auto& v : in) {
    norm += v * v;
    if (std::abs(v) <= maxcomp)
      continue;
    phase = v / std::abs(v);
    maxcomp = std::abs(v);
  }
  if (norm == 0)
    throw std::runtime_error("zero vector");
  norm = 1 / std::sqrt(norm);
  std::vector<double> out;
  for (const auto& v : in)
    out.push_back(v * norm * phase);
  return out;
}

void test_eigen(const std::string& title = "") {
  std::map<double, std::vector<double>> expected_eigensolutions;
  bool hermitian = std::abs(hmat[0 + n * 1] - hmat[1 + n * 0]) < 1e-10;
  std::cout << hmat[0 + n * 1] << hmat[1 + n * 0] << std::endl;
  {
    Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>> he(hmat.data(), n, n);
    Eigen::EigenSolver<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>> esolver(he);
    auto ev = esolver.eigenvalues().real();
    expected_eigenvalues.clear();
    auto evecreal = esolver.eigenvectors().real().eval();
    for (int i = 0; i < n; i++)
      expected_eigenvalues.push_back(ev[i]);
    for (int i = 0; i < n; i++)
      for (int j = 0; j < n; j++)
        expected_eigensolutions[ev[i]].push_back(evecreal(j, i)); // won't work for degenerate eigenvalues!
    ASSERT_EQ(expected_eigensolutions.size(), n);
    //    for (const auto& ee : expected_eigensolutions)
    //      std::cout << "expected eigensolution " << ee.first << std::endl<<ee.second<<std::endl;
  }
  std::sort(expected_eigenvalues.begin(), expected_eigenvalues.end());
  std::cout << "expected eigenvalues " << expected_eigenvalues << std::endl;
  { // testing the test: that sort of eigenvalues is the same thing as std::map sorting of keys
    std::vector<double> testing;
    for (const auto& ee : expected_eigensolutions)
      testing.push_back(ee.first);
    ASSERT_THAT(expected_eigenvalues, ::testing::Pointwise(::testing::DoubleNear(1e-13), testing));
  }
  std::cout << "prep done" << std::endl;
  for (int nroot = 1; nroot <= n && nroot <= 28; nroot++) {
    for (auto np = 0; np <= n && np <= 50 && (hermitian or np == 0); np += std::max(nroot, int(n) / 10)) {
      molpro::cout << "\n\n*** " << title << ", " << nroot << " roots, problem dimension " << n << ", pspace dimension "
                   << np << std::endl;

      auto [solver, logger] = molpro::test::create_LinearEigensystem();
      auto options = CastOptions::LinearEigensystem(solver->get_options());
      options->n_roots = nroot;
      options->convergence_threshold = 1.0e-10;
      options->norm_thresh = 1.0e-14;
      options->svd_thresh = 1.0e-10;
      options->max_size_qspace = std::max(3 * nroot, std::min(int(n), std::min(1000, 3 * nroot)) - np);
      options->reset_D = 4;
      options->hermiticity = hermitian;
      solver->set_options(options);
      options = CastOptions::LinearEigensystem(solver->get_options());
      molpro::cout << "convergence threshold = " << options->convergence_threshold.value() << ", svd thresh"
                   << options->svd_thresh.value() << ", norm thresh" << options->norm_thresh.value()
                   << ", max size of Q = " << options->max_size_qspace.value()
                   << ", reset D = " << options->reset_D.value() << std::endl;
      logger->max_trace_level = molpro::linalg::itsolv::Logger::None;
      logger->max_warn_level = molpro::linalg::itsolv::Logger::Error;
      logger->data_dump = false;
      std::vector<Rvector> g;
      std::vector<Rvector> x;

      std::vector<size_t> guess;
      {
        std::vector<double> diagonals;
        for (auto i = 0; i < n; i++) {
          diagonals.push_back(hmat[i + n * i]);
        }
        for (size_t root = 0; root < solver->n_roots(); root++) {
          x.emplace_back(n, 0);
          g.emplace_back(n);
          guess.push_back(std::min_element(diagonals.begin(), diagonals.end()) - diagonals.begin()); // initial guess
          *std::min_element(diagonals.begin(), diagonals.end()) = 1e99;
          x.back()[guess.back()] = 1; // initial guess
        }
      }
      //        std::cout << "guess: " << guess << std::endl;
      int nwork = solver->n_roots();

      std::vector<scalar> PP;
      std::vector<Pvector> pspace;
      using vectorP = std::vector<scalar>;
      using molpro::linalg::itsolv::CVecRef;
      using molpro::linalg::itsolv::VecRef;
      //          using fapply_on_p_type = std::function<void(const std::vector<VectorP>&, const CVecRef<P>&, const
      //          VecRef<R>&)>;
      std::function<void(const std::vector<vectorP>&, const CVecRef<Pvector>&, const VecRef<Rvector>&)> apply_p =
          [](const std::vector<vectorP>& pvectors, const CVecRef<Pvector>& pspace, const VecRef<Rvector>& action) {
            for (size_t i = 0; i < pvectors.size(); i++) {
              auto& actioni = (action[i]).get();
              for (size_t pi = 0; pi < pspace.size(); pi++) {
                const auto& p = pspace[pi].get();
                for (const auto& pel : p)
                  for (size_t j = 0; j < n; j++)
                    actioni[j] += hmat[j + n * pel.first] * pel.second * pvectors[i][pi];
              }
            }
          };

      for (auto iter = 0; iter < 100; iter++) {
        if (iter == 0 && np > 0) {
          std::vector<double> diagonals;
          for (auto i = 0; i < n; i++) {
            diagonals.push_back(hmat[i + n * i]);
          }
          for (size_t p = 0; p < np; p++) {
            std::map<size_t, scalar> pp;
            pp.insert({std::min_element(diagonals.begin(), diagonals.end()) - diagonals.begin(), 1});
            pspace.push_back(pp);
            *std::min_element(diagonals.begin(), diagonals.end()) = 1e99;
          }
          //            std::cout << "P space: " << pspace << std::endl;
          PP.reserve(np * np);
          for (const auto& i : pspace)
            for (const auto& j : pspace) {
              const size_t kI = i.begin()->first;
              const size_t kJ = j.begin()->first;
              PP.push_back(hmat[kI + n * kJ]);
            }
          //            std::cout << "PP: " << PP << std::endl;

          nwork = solver->add_p(molpro::linalg::itsolv::cwrap(pspace),
                                molpro::linalg::array::span::Span<double>(PP.data(), PP.size()),
                                molpro::linalg::itsolv::wrap(x), molpro::linalg::itsolv::wrap(g), apply_p);
        } else {
          action(x, g);
          //          for (auto root = 0; root < nwork; root++) {
          //            std::cout << "before addVector() x:";
          //            for (auto i = 0; i < n; i++)
          //              std::cout << " " << x[root][i];
          //            std::cout << std::endl;
          //            std::cout << "before addVector() g:";
          //            for (auto i = 0; i < n; i++)
          //              std::cout << " " << g[root][i];
          //            std::cout << std::endl;
          //          }
          nwork = solver->add_vector(x, g, apply_p);
          std::cout << "solver.add_vector returns nwork=" << nwork << std::endl;
        }
        if (nwork == 0)
          break;
        //        for (auto root = 0; root < nwork; root++) {
        //          std::cout << "after addVector() or addP()"
        //                    << " eigenvalue=" << solver.working_set_eigenvalues()[root] << " error=" <<
        //                    solver.errors()[root]
        //                    << "\nx:";
        //          for (auto i = 0; i < n; i++)
        //            std::cout << " " << x[root][i];
        //          std::cout << std::endl;
        //          std::cout << "after addVector() g:";
        //          for (auto i = 0; i < n; i++)
        //            std::cout << " " << g[root][i];
        //          std::cout << std::endl;
        //        }
        //        x.resize(nwork);
        //        g.resize(nwork);
        //        for (auto k = 0; k < solver.working_set().size(); k++) {
        //          for (auto p = 0; p < np; p++)
        //            for (const auto& pc : pspace[p])
        //              for (auto i = 0; i < n; i++)
        //                g[k][i] += matrix(i, pc.first) * pc.second * Pcoeff[k][p];
        //          //            molpro::cout << "old error " << solver.m_errors[solver.working_set()[k]];
        //          //            solver.m_errors[solver.working_set()[k]] = std::sqrt(g[k].dot(g[k]));
        //          //            molpro::cout << "; new error " << solver.m_errors[solver.working_set()[k]] <<
        //          std::endl;
        //        }
        //          for (auto root = 0; root < nwork; root++) {
        //            std::cout << "after p gradient g:";
        //            for (auto i = 0; i < n; i++)
        //              std::cout << " " << g[root][i];
        //            std::cout << std::endl;
        //          }
        update(g, solver->working_set_eigenvalues());
        //          for (auto root = 0; root < nwork; root++) {
        //            std::cout << "after update() x:";
        //            for (auto i = 0; i < n; i++)
        //              std::cout << " " << x[root][i];
        //            std::cout << std::endl;
        //            std::cout << "after update() g:";
        //            for (auto i = 0; i < n; i++)
        //              std::cout << " " << g[root][i];
        //            std::cout << std::endl;
        //          }
        solver->report();
        //          if (*std::max_element(solver.errors().begin(), solver.errors().end()) < solver.m_thresh)
        nwork = solver->end_iteration(x, g);
        std::cout << "solver.add_vector returns nwork=" << nwork << std::endl;
        if (nwork == 0)
          break;
      }
      std::cout << "Error={ ";
      for (const auto& e : solver->errors())
        std::cout << e << " ";
      std::cout << "} after " << solver->statistics().iterations << " iterations, " << solver->statistics().r_creations
                << " R vectors" << std::endl;
      //        for (size_t root = 0; root < solver.n_roots(); root++) {
      //          std::cout << "Eigenvalue " << std::fixed << std::setprecision(9) << solver.eigenvalues()[root] <<
      //          std::endl;
      //        solver.solution(root, x.front(), g.front(), Pcoeff.front());
      //        std::cout << "Eigenvector: (norm=" << std::sqrt(x[0].dot(x[0])) << "): ";
      //        for (size_t k = 0; k < n; k++)
      //          std::cout << " " << (x[0])[k];
      //        std::cout << std::endl;
      //        }
      EXPECT_THAT(solver->errors(), ::testing::Pointwise(::testing::DoubleNear(2 * solver->convergence_threshold()),
                                                         std::vector<double>(nroot, double(0))));
      std::cout << "expected eigenvalues "
                << std::vector<double>(expected_eigenvalues.data(), expected_eigenvalues.data() + solver->n_roots())
                << std::endl;
      std::cout << "obtained eigenvalues " << solver->eigenvalues() << std::endl;
      EXPECT_THAT(solver->eigenvalues(),
                  ::testing::Pointwise(::testing::DoubleNear(2e-9),
                                       std::vector<double>(expected_eigenvalues.data(),
                                                           expected_eigenvalues.data() + solver->n_roots())));
      EXPECT_LE(solver->statistics().r_creations, (nroot + 1) * 15);
      std::vector<std::vector<double>> parameters, residuals;
      std::vector<int> roots;
      for (int root = 0; root < solver->n_roots(); root++) {
        parameters.emplace_back(n);
        residuals.emplace_back(n);
        roots.push_back(root);
      }
      solver->solution(roots, parameters, residuals);
      auto evalexpec = residual(parameters, residuals);
      for (const auto& r : residuals)
        EXPECT_LE(dot(r, r), 1e-15);
      int root = 0;
      for (const auto& ee : expected_eigensolutions) {
        EXPECT_NEAR(ee.first, solver->eigenvalues()[root], 1e-10);
        EXPECT_THAT(phase_normalise(parameters[root]),
                    ::testing::Pointwise(::testing::DoubleNear(1e-8), phase_normalise(ee.second)));
        //        std::cout << phase_normalise(parameters) << std::endl;
        //        std::cout << phase_normalise(ee.second) << std::endl;
        root++;
        if (root >= solver->n_roots())
          break;
      }
    }
  }
}

TEST(IterativeSolver, file_eigen) {
  for (const auto& file : std::vector<std::string>{"bh", "hf", "phenol"}) {
//  for (const auto& file : std::vector<std::string>{"bh", "hf"}) {
    load_matrix(file, file == "phenol" ? 0 : 1e-8);
    test_eigen(file);
  }
}

TEST(IterativeSolver, n_eigen) {
  size_t n = 100;
  double param = 1;
  //  for (auto param : std::vector<double>{.01, .1, 1, 10, 100}) {
  //  for (auto param : std::vector<double>{.01, .1, 1}) {
  for (auto param : std::vector<double>{1}) {
    load_matrix(n, "", param);
    test_eigen(std::to_string(n) + "/" + std::to_string(param));
  }
}
TEST(IterativeSolver, nonhermitian_eigen) {
  size_t n = 30;
  double param = 1;
  //  for (auto param : std::vector<double>{.01, .1, 1, 10, 100}) {
  //  for (auto param : std::vector<double>{.01, .1, 1}) {
  for (auto param : std::vector<double>{1}) {
    load_matrix(n, "", param, false);
    //    std::cout << "matrix " << hmat << std::endl;
    test_eigen(std::to_string(n) + "/" + std::to_string(param) + ", non-hermitian");
  }
}

TEST(IterativeSolver, small_eigen) {
  for (int n = 1; n < 20; n++) {
    double param = 1;
    load_matrix(n, "", param);
    test_eigen(std::to_string(n) + "/" + std::to_string(param));
  }
}

TEST(IterativeSolver, symmetry_eigen) {
  for (int n = 1; n < 10; n++) {
    double param = 1;
    load_matrix(n, "", param);
    for (auto i = 0; i < n; i++)
      for (auto j = 0; j < n; j++)
        if (((i % 3) == 0 and (j % 3) != 0) or ((j % 3) == 0 and (i % 3) != 0))
          hmat[j + i * n] = 0;
    if (false) {
      std::cout << "hmat " << std::endl;
      for (auto i = 0; i < n; i++) {
        std::cout << "i%3" << i % 3 << std::endl;
        for (auto j = 0; j < n; j++)
          std::cout << " " << hmat[j + i * n];
        std::cout << std::endl;
      }
    }
    test_eigen(std::to_string(n) + "/sym/" + std::to_string(param));
  }
}
