#ifndef RSPT_H
#define RSPT_H
#include "IterativeSolver.h"

namespace IterativeSolver{

class RSPT : public IterativeSolverBase
{
  public:
    RSPT(const ParameterSetTransformation residualFunction, const ParameterSetTransformation preconditionerFunction=&IterativeSolver::steepestDescent);
//    static void test(size_t dimension, size_t roots=1, int verbosity=0, int problem=0);
  protected:
    virtual void extrapolate(ParameterVectorSet & residual, ParameterVectorSet & solution, ParameterVectorSet & other, const optionMap options=optionMap());
  public:
    int m_roots; ///< How many roots to calculate (defaults to size of solution and residual vectors)
    int m_order; ///< Up to what order of perturbation theory should the energy be obtained.
    std::vector<double> incremental_energies(); ///< The incremental energies order by order.
    std::vector<double> energies(); ///< The total energies order by order.
    std::vector<double> eigenvalues(); ///< The variatonally calculated eigenvalues
  private:
    RSPT();
};

}

#endif // RSPT_H
