#include "SolverFactory_value_types.h"
#include <molpro/linalg/itsolv/SolverFactory-implementation.h>

template class molpro::linalg::itsolv::SolverFactory<std::vector<long double>, std::deque<long double>,
                                                     std::map<std::size_t, long double>>;
