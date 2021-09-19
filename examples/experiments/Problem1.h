#ifndef ITERATIVESOLVER_PAPER_CALCULATIONS__PROBLEM1_H_
#define ITERATIVESOLVER_PAPER_CALCULATIONS__PROBLEM1_H_
#include <molpro/linalg/itsolv/IterativeSolver.h>
class Problem1 : public molpro::linalg::itsolv::Problem<std::vector<double>> {
protected:
  double matrix(int i, int j) const {
    return i == j ? i + 1 : offdiagonal
        //  * ((i + j) % n)
        ;
  }

public:
  using Problem::container_t;
  using Problem::value_t;
  const size_t n;
  const value_t offdiagonal;
  Problem1(size_t n = 10, value_t offdiagonal = 0.001) : n(n), offdiagonal(offdiagonal) {}

  bool diagonals(container_t &d) const override {
    for (size_t i = 0; i < d.size(); i++)
      d[i] = matrix(i, i);
    return true;
  }

  value_t residual(const container_t &v, container_t &a) const override {
    value_t value = 0;
    for (size_t i = 0; i < a.size(); i++) {
      a[i] = 0;
      for (size_t j = 0; j < a.size(); j++)
        a[i] += matrix(i, j) * (v[j] - 1);
      value += 0.5 * a[i] * (v[i] - 1);
    }
    return value;
  }

  void action(const CVecRef<container_t> &parameters, const VecRef<container_t> &actions) const override {
    for (size_t k = 0; k < parameters.size(); k++) {
      const auto &v = parameters[k].get();
      auto &a = actions[k].get();
      for (size_t i = 0; i < a.size(); i++) {
        a[i] = 0;
        for (size_t j = 0; j < a.size(); j++)
          a[i] += matrix(i, j) * v[j];
      }
    }
  }
};
#endif // ITERATIVESOLVER_PAPER_CALCULATIONS__PROBLEM1_H_
