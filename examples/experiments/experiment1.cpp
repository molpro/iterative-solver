#include "Problem1.h"
#include <iostream>
#include <map>
#include <molpro/linalg/itsolv/LinearEigensystemDavidson.h>
#include <molpro/linalg/itsolv/SolverFactory.h>
#include <vector>

using Problem = Problem1;
const size_t roots = 100;
const size_t dimension = 1000;
const Problem::value_t off_diagonal = 100;

using R = Problem::container_t;
using Q = R;
using P = std::map<size_t, Problem::value_t>;

int main() {
  auto problem = Problem(dimension, off_diagonal);
  molpro::linalg::itsolv::ArrayHandlers<R, Q, P> handlers;
  molpro::linalg::itsolv::LinearEigensystemDavidson<R, Q, P> solver(
      std::make_shared<molpro::linalg::itsolv::ArrayHandlers<R, Q, P>>());
  //  auto solver = molpro::linalg::itsolv::create_LinearEigensystem<R, Q, P>("Davidson","max_size_qspace=150");
  std::vector<R> c, g;
  for (size_t root = 0; root < roots; ++root) {
    c.emplace_back(dimension, 0);
    g.emplace_back(dimension, 0);
    c.back()[root] = 1;
  }
  solver.set_hermiticity(true);
  solver.set_reset_D(5);
  solver.set_n_roots(roots);
  solver.set_max_iter(1000);
  solver.set_convergence_threshold(1e-11);
  using molpro::linalg::itsolv::wrap;
  solver.solve(wrap(c), wrap(g), problem);
  std::cout << solver.statistics() << std::endl;
  std::vector<int> root_list(roots);
  std::iota(root_list.begin(), root_list.end(), 0);
  solver.solution(root_list, c, g);
  for (size_t i = 0; i < roots; ++i) {
    auto resid = std::sqrt(std::inner_product(g[i].begin(), g[i].end(), g[i].begin(), double(0)));
    problem.action({c[i]}, {g[i]});
    auto norm = std::inner_product(c[i].begin(), c[i].end(), c[i].begin(), double(0));
    auto chc = std::inner_product(c[i].begin(), c[i].end(), g[i].begin(), double(0));
    auto expectation_value = chc / norm;
    std::cout << "Eigensolution " << root_list[i] << " norm=" << norm << " expectation value=" << expectation_value
              << " eigenvalue=" << solver.eigenvalues()[root_list[i]] << " residual=" << resid << std::endl;
  }
  return 0;
}