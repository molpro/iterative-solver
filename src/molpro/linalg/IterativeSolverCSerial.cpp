#include "IterativeSolverC.h"

void IterativeSolverLinearEigensystemInitialize(size_t nQ, size_t nroot, size_t range_begin, size_t range_end,
                                                double thresh, int hermitian, int verbosity, const char* fname,
                                                int64_t fcomm, int lmppx) {}

void IterativeSolverLinearEquationsInitialize(size_t n, size_t nroot, size_t range_begin, size_t range_end,
                                              const double* rhs, double aughes, double thresh, int hermitian,
                                              int verbosity, const char* fname, int64_t fcomm, int lmppx) {}

void IterativeSolverDIISInitialize(size_t n, size_t range_begin, size_t range_end, double thresh, int verbosity,
                                   const char* fname, int64_t fcomm, int lmppx) {}

void IterativeSolverOptimizeInitialize(size_t n, size_t range_begin, size_t range_end, double thresh, int verbosity,
                                       char* algorithm, int minimize, const char* fname, int64_t fcomm, int lmppx) {}

void IterativeSolverFinalize() {}

size_t IterativeSolverAddVector(double* parameters, double* action, double* parametersP, int sync, int lmppx) {
  return {};
}

size_t IterativeSolverAddValue(double value, double* parameters, double* action, int sync, int lmppx) { return {}; }

int IterativeSolverEndIteration(double* c, double* g, double* error, int lmppx) { return {}; }

size_t IterativeSolverAddP(size_t nP, const size_t* offsets, const size_t* indices, const double* coefficients,
                           const double* pp, double* parameters, double* action, double* parametersP, int sync,
                           int lmppx) {return 0;}

void IterativeSolverEigenvalues(double* eigenvalues) {}

void IterativeSolverWorkingSetEigenvalues(double* eigenvalues) {}

void IterativeSolverOption(const char* key, const char* val) {}

size_t IterativeSolverSuggestP(const double* solution, const double* residual, size_t maximumNumber, double threshold,
                               size_t* indices, int lmppx) {
  return {};
}
