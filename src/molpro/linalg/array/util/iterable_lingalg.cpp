#include "iterable_lingalg.h"

namespace molpro::linalg::array::util {
template void scal(double alpha, Span<double> &x);
template void add(double alpha, Span<double> &x);
template void times(Span<double> &x, const Span<double> &y);
template void times(Span<double> &x, const Span<double> &y, const Span<double> &z);
template void fill(double alpha, Span<double> &x);
template void axpy(double alpha, const Span<double> &x, Span<double> &y);
template void axpy_sparse(double alpha, const std::map<size_t, double> &x, Span<double> &y);
template double dot(const Span<double> &x, const Span<double> &y);
template double dot_sparse(const Span<double> &x, const std::map<size_t, double> &y);
} // namespace molpro::linalg::array::util