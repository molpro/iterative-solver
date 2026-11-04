#include "SolverFactory_value_types.h"
#include <molpro/linalg/itsolv/SolverFactory-implementation.h>

template class molpro::linalg::itsolv::SolverFactory<std::vector<double>, std::vector<double>,
                                                     std::map<std::size_t, double>>;
template class molpro::linalg::itsolv::SolverFactory<std::vector<double>, molpro::linalg::array::DistrArrayFile,
                                                     std::map<std::size_t, double>>;
