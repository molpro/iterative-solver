#include <molpro/linalg/itsolv/Interpolate-implementation.h>

namespace molpro::linalg::itsolv {

template class Interpolator<double>;
// Interpolator<long double> is instantiated once the solver stack it nests is precision-generic:
// the Morse interpolant fits its parameters with a NonLinearEquations solve.

} // namespace molpro::linalg::itsolv
