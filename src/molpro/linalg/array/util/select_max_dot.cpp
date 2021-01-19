#include "select_max_dot.h"

namespace molpro::linalg::array::util {

template std::map<size_t, double>
select_max_dot<Span<double>, Span<double>, double, double>(size_t n, const Span<double>& x, const Span<double>& y);

template std::map<size_t, double>
select_max_dot_iter_sparse<Span<double>, std::map<size_t, double>, double, double>(size_t n, const Span<double>& x,
                                                                                   const std::map<size_t, double>& y);
template std::map<size_t, double>
select_max_dot_sparse<std::map<size_t, double>, std::map<size_t, double>, double, double>(
    size_t n, const std::map<size_t, double>& x, const std::map<size_t, double>& y);

} // namespace molpro::linalg::array::util