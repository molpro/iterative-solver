#include "SolverFactory_value_types.h"
#include <molpro/linalg/itsolv/SolverFactory-implementation.h>

template class molpro::linalg::itsolv::SolverFactory<
    std::vector<std::complex<double>>, std::deque<std::complex<double>>, std::map<std::size_t, std::complex<double>>>;
