#ifndef LINEARALGEBRA_SRC_MOLPRO_LINALG_ITERATIVESOLVER_ITERATIVESOLVERC_H_
#define LINEARALGEBRA_SRC_MOLPRO_LINALG_ITERATIVESOLVER_ITERATIVESOLVERC_H_
#include <stddef.h>
#include <cstdint>
extern "C" void IterativeSolverLinearEigensystemInitialize(size_t nQ, size_t nroot, double thresh,
                                                           unsigned int maxIterations, int verbosity, const char* fname,
                                                           int64_t fcomm, int lmppx);

extern "C" void IterativeSolverLinearEquationsInitialize(size_t n, size_t nroot, const double* rhs, double aughes,
                                                         double thresh, unsigned int maxIterations, int verbosity);

extern "C" void IterativeSolverDIISInitialize(size_t n, double thresh, unsigned int maxIterations, int verbosity);

extern "C" void IterativeSolverOptimizeInitialize(size_t n, double thresh, unsigned int maxIterations, int verbosity,
                                                  char* algorithm, int minimize);

extern "C" void IterativeSolverFinalize();

extern "C" int IterativeSolverAddVector(double* parameters, double* action, double* parametersP, int sync, int lmppx);

extern "C" int IterativeSolverAddValue(double* parameters, double value, double* action, int sync, int lmppx);

extern "C" int IterativeSolverEndIteration(double* c, double* g, double* error, int lmppx);

extern "C" void IterativeSolverAddP(size_t nP, const size_t* offsets, const size_t* indices, const double* coefficients,
                                    const double* pp, double* parameters, double* action, double* parametersP,
                                    int lmppx);

extern "C" void IterativeSolverEigenvalues(double* eigenvalues);

extern "C" void IterativeSolverOption(const char* key, const char* val);

extern "C" size_t IterativeSolverSuggestP(const double* solution, const double* residual, size_t maximumNumber,
                                          double threshold, size_t* indices, int lmppx);
#endif // LINEARALGEBRA_SRC_MOLPRO_LINALG_ITERATIVESOLVER_ITERATIVESOLVERC_H_