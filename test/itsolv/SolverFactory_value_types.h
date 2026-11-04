#ifndef LINEARALGEBRA_TEST_ITSOLV_SOLVERFACTORY_VALUE_TYPES_H_
#define LINEARALGEBRA_TEST_ITSOLV_SOLVERFACTORY_VALUE_TYPES_H_
/*!
 * @file
 * @brief The solver factories that several tests use: std::vector R and std::map P with std::deque Q in each value
 * type, and the double ones with std::vector or DistrArrayFile Q.
 *
 * Instantiating a factory instantiates every solver, which takes several GB of compiler memory under the sanitizers.
 * The tests that use these containers include this header so that each factory is compiled once, in
 * SolverFactory_implementation.cpp or one of the other SolverFactory_*.cpp, rather than in every one of them.
 */
#include <complex>
#include <cstddef>
#include <deque>
#include <map>
#include <molpro/linalg/array/DistrArrayFile.h>
#include <molpro/linalg/itsolv/SolverFactory.h>
#include <vector>

namespace molpro::linalg::itsolv {
extern template class SolverFactory<std::vector<double>, std::vector<double>, std::map<std::size_t, double>>;
extern template class SolverFactory<std::vector<double>, array::DistrArrayFile, std::map<std::size_t, double>>;
extern template class SolverFactory<std::vector<double>, std::deque<double>, std::map<std::size_t, double>>;
extern template class SolverFactory<std::vector<std::complex<double>>, std::deque<std::complex<double>>,
                                    std::map<std::size_t, std::complex<double>>>;
extern template class SolverFactory<std::vector<long double>, std::deque<long double>,
                                    std::map<std::size_t, long double>>;
} // namespace molpro::linalg::itsolv

#endif // LINEARALGEBRA_TEST_ITSOLV_SOLVERFACTORY_VALUE_TYPES_H_
