#include "IterativeSolver.h"
#include "PagedVector.h"


//  typedef SimpleParameterVector pv;
using scalar = double;
using pv = LinearAlgebra::PagedVector<scalar>;

static double alpha;
static double anharmonicity;
static double n;

void anharmonic_residual(const pv& psx, pv& outputs) {
  std::vector<scalar> psxk(n);
  std::vector<scalar> output(n);
  psx.get(psxk.data(), n, 0);
  for (size_t i = 0; i < n; i++) {
    output[i] = (alpha * (i + 1) + anharmonicity * psxk[i]) * psxk[i];
    for (size_t j = 0; j < n; j++)
      output[i] += (i + j) * psxk[j];
  }
  outputs.put(output.data(), n, 0);
}

void update(pv& psc, const pv& psg) {
  std::vector<scalar> psck(n);
  std::vector<scalar> psgk(n);
  psg.get(psgk.data(), n, 0);
  psc.get(psck.data(), n, 0);
  for (size_t i = 0; i < n; i++)
    psck[i] -= psgk[i] / (alpha * (i + 1));
  psc.put(psck.data(), n, 0);
}

int main(int argc, char* argv[]) {
  alpha = 1;
  n = 100;
  anharmonicity = .5;
  LinearAlgebra::DIIS<pv> solver;
  solver.m_verbosity = 1;
  solver.m_maxIterations = 100;
  pv g(n);
  pv x(n);
  x.scal(0);
  scalar one = 1;
  x.put(&one, 1, 0);  // initial guess
  for (size_t iter = 0; iter < solver.m_maxIterations; ++iter) {
    anharmonic_residual(x, g);
    solver.addVector(x, g);
    update(x, g);
    if (solver.endIteration(x, g)) break;
  }
  std::cout << "Distance of solution from origin: " << std::sqrt(x.dot(x)) << std::endl;
  std::cout << "Error=" << solver.errors().front() << " after " << solver.iterations() << " iterations" << std::endl;
}
