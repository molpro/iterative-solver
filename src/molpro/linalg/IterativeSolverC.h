#ifndef LINEARALGEBRA_SRC_MOLPRO_LINALG_ITERATIVESOLVER_ITERATIVESOLVERC_H_
#define LINEARALGEBRA_SRC_MOLPRO_LINALG_ITERATIVESOLVER_ITERATIVESOLVERC_H_
#include <cstdint>
#include <stddef.h>


extern "C" void IterativeSolverLinearEigensystemInitialize(size_t nQ, size_t nroot, size_t* range_begin,
                                                           size_t* range_end, double thresh, double thresh_value,
                                                           int hermitian, int verbosity, const char* fname,
                                                           int64_t fcomm, const char* algorithm, const char* options);

extern "C" void IterativeSolverLinearEquationsInitialize(size_t n, size_t nroot, size_t* range_begin, size_t* range_end,
                                                         const double* rhs, double aughes, double thresh,
                                                         double thresh_value, int hermitian, int verbosity,
                                                         const char* fname, int64_t fcomm, const char* algorithm,
                                                         const char* options);

extern "C" void IterativeSolverNonLinearEquationsInitialize(size_t n, size_t* range_begin, size_t* range_end,
                                                            double thresh, int verbosity, const char* fname,
                                                            int64_t fcomm, const char* algorithm, const char* options);

extern "C" void IterativeSolverOptimizeInitialize(size_t n, size_t* range_begin, size_t* range_end, double thresh,
                                                  double thresh_value, int verbosity, int minimize, const char* fname,
                                                  int64_t fcomm, const char* algorithm, const char* options);

//! Remove the current solver instance. The most recently created remaining instance becomes current.
extern "C" void IterativeSolverFinalize();

//! Remove all solver instances
extern "C" void IterativeSolverFinalizeAll();

//! Handle of the current solver instance, or -1 if there is none. Call immediately after an Initialize function to
//! obtain the handle of the new instance.
extern "C" int64_t IterativeSolverCurrentHandle();

//! Make the instance with the given handle current, so that subsequent calls act on it. @return 0 on success, nonzero
//! if there is no instance with this handle
extern "C" int IterativeSolverSelect(int64_t handle);

//! Remove the instance with the given handle, whether or not it is current. Does nothing for an unknown handle.
extern "C" void IterativeSolverFinalizeHandle(int64_t handle);

//! Number of roots (solutions) of the current instance
extern "C" size_t IterativeSolverNRoots();

//! Dimension of the parameter space of the current instance
extern "C" size_t IterativeSolverDimension();

extern "C" size_t IterativeSolverAddVector(size_t buffer_size, double* parameters, double* action, int sync);

extern "C" void IterativeSolverSolution(int nroot, int* roots, double* parameters, double* action, int sync);

extern "C" size_t IterativeSolverAddValue(double value, double* parameters, double* action, int sync);

extern "C" size_t IterativeSolverEndIteration(size_t buffer_size, double* solution, double* residual, int sync);

extern "C" int IterativeSolverEndIterationNeeded();

typedef void (*cheesefunc)(char *name, void *user_data);
extern "C" void find_cheeses(cheesefunc user_func, void *user_data);

typedef void (*apply_on_p_t)(const double*, double*, const size_t, const size_t*);
extern "C" size_t IterativeSolverAddP(size_t buffer_size, size_t nP, const size_t* offsets, const size_t* indices,
                                      const double* coefficients, const double* pp, double* parameters, double* action,
                                      int sync, apply_on_p_t func);

extern "C" void IterativeSolverErrors(double* errors);

extern "C" int IterativeSolverConverged();

extern "C" void IterativeSolverEigenvalues(double* eigenvalues);

extern "C" void IterativeSolverWorkingSetEigenvalues(double* eigenvalues);

extern "C" size_t IterativeSolverSuggestP(const double* solution, const double* residual, size_t maximumNumber,
                                          double threshold, size_t* indices);

extern "C" void IterativeSolverPrintStatistics();

extern "C" int IterativeSolverNonLinear();

extern "C" int IterativeSolverHasValues();
extern "C" int IterativeSolverHasEigenvalues();

extern "C" void IterativeSolverSetDiagonals(const double* diagonals);

extern "C" void IterativeSolverDiagonals(double* diagonals);

extern "C" double IterativeSolverValue();

extern "C" int IterativeSolverVerbosity();

extern "C" int IterativeSolverMaxIter();
extern "C" void IterativeSolverSetMaxIter(int max_iter);

extern "C" int64_t mpicomm_self();

extern "C" int64_t mpicomm_global();

extern "C" int64_t IterativeSolver_mpicomm_global();
extern "C" int64_t IterativeSolver_mpicomm_self();
#endif // LINEARALGEBRA_SRC_MOLPRO_LINALG_ITERATIVESOLVER_ITERATIVESOLVERC_H_
