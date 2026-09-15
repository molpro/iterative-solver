#ifndef LINEARALGEBRA_SRC_MOLPRO_LINALG_SCALAR_TRAITS_H_
#define LINEARALGEBRA_SRC_MOLPRO_LINALG_SCALAR_TRAITS_H_

/*!
 * @file
 * @brief Traits and small helpers that let the same code serve real and complex scalar types.
 *
 * Kept free of any dependency on the rest of the library so that both molpro::linalg::array and
 * molpro::linalg::itsolv can use it.
 */

#include <complex>
#include <limits>
#include <type_traits>

namespace molpro::linalg {

template <typename T>
struct is_complex : std::false_type {};

template <typename T>
struct is_complex<std::complex<T>> : std::true_type {};

//! The real type underlying a (possibly complex) scalar type
template <typename T>
struct real_type {
  using type = T;
};

template <typename T>
struct real_type<std::complex<T>> {
  using type = T;
};

/*!
 * @brief The real type underlying @p T, i.e. @p T itself for a real type and @c U for @c std::complex<U>.
 *
 * Quantities that are real by construction -- eigenvalues of a hermitian matrix, singular values,
 * norms and the thresholds they are compared against -- are expressed in this type rather than in
 * the scalar type of the problem.
 */
template <typename T>
using real_type_t = typename real_type<T>::type;

/*!
 * @brief Rescale a tolerance that was calibrated for IEEE double precision to the working precision.
 *
 * A tolerance of the form "this quantity vanishes to within rounding error" is a fixed multiple of
 * the machine epsilon of the arithmetic used. The constants in this library were chosen for double
 * precision; this function preserves their safety margin, measured in machine epsilons, at any other
 * precision, and returns the constant unchanged when the working precision is double.
 */
template <typename value_type>
real_type_t<value_type> precision_scaled(double tolerance_for_double) {
  using real_t = real_type_t<value_type>;
  if constexpr (std::is_same_v<real_t, double>) {
    return tolerance_for_double;
  } else {
    return real_t(tolerance_for_double) *
           (std::numeric_limits<real_t>::epsilon() / real_t(std::numeric_limits<double>::epsilon()));
  }
}

} // namespace molpro::linalg

#endif // LINEARALGEBRA_SRC_MOLPRO_LINALG_SCALAR_TRAITS_H_
