#ifndef LINEARALGEBRA_SRC_MOLPRO_LINALG_ITSOLV_INTERPOLATE_H_
#define LINEARALGEBRA_SRC_MOLPRO_LINALG_ITSOLV_INTERPOLATE_H_
#include <molpro/linalg/scalar_traits.h>

#include <cmath>
#include <limits>
#include <ostream>
#include <string>
#include <vector>

/*!
 * @brief 4-parameter interpolation of a 1-dimensional function given two points for which function values and first
 * derivatives are known.
 */
namespace molpro::linalg::itsolv {

/*!
 * @brief The interpolant, in the precision of the problem it is used for.
 *
 * The parameter is a *real* type, and names a precision rather than a scalar type: minimising a
 * function presupposes that its values are ordered, so a complex instantiation is meaningless and is
 * rejected below. What varies is how accurately the line search is carried out. Optimize passes
 * value_type_abs, which for a std::complex<double> container is double, and for a long double one is
 * long double -- the case this template exists for, since narrowing the function values and gradients
 * to double would cap the line search at double precision however the solver was instantiated.
 *
 * @see Interpolate for the double-precision specialisation, which is the historical name and the one
 * used throughout the double-precision API.
 */
template <typename value_type = double>
class Interpolator {
  static_assert(!molpro::linalg::is_complex<value_type>{},
                "the interpolant orders function values in order to minimise them, so it is defined "
                "only for a real type; pass the real type underlying your scalar, as Optimize does");

public:
  using value_t = value_type;

  struct point {
    value_type x;                                                     //< abscissa
    value_type f = std::numeric_limits<value_type>::quiet_NaN();      //< function value at x
    value_type f1 = std::numeric_limits<value_type>::quiet_NaN();     //< function first gradient at x
    value_type f2 = std::numeric_limits<value_type>::quiet_NaN();     //< function second gradient at x
  };
  /*!
   * @brief Construct the interpolant
   * @param p0 Defining point
   * @param p1 Defining point
   * @param interpolant The interpolation method. An exception is thrown if it is not one of the implemented values.
   * @param verbosity Values greater than zero show information on constructing and using the interpolant.
   */
  explicit Interpolator(point p0, point p1, std::string interpolant = "cubic", int verbosity = 0);
  /*!
   * @brief Evaluate the interpolant and its derivative at a given point
   * @param x
   * @return
   */
  point operator()(value_type x) const;
  /*!
   * @brief Find the minimum of the interpolant within a range
   * @param xa first bound of range
   * @param xb second bound of range
   * @param bracket_grid number of intervals in |xa-xb|, to be considered in initial bracketing of the minimum. Large
   * values result in many function evaluations, but if set too small in cases of multiple minima, the global minimum
   * may not be found.
   * @return The minimum point. The result may be one of Interpolate(xa), Interpolate(xb) with non-zero first
   * derivative, if no other minimum was found in the interval
   */
  point minimize(value_type xa, value_type xb, size_t bracket_grid = 100, size_t max_bracket_grid = 100000,
                 bool analytic = true) const;
  point minimize_cubic() const;
  static std::vector<std::string> interpolants();

  const std::vector<value_type>& parameters() const { return m_parameters; }

private:
  const point m_p0, m_p1;
  const std::string m_interpolant;
  std::vector<value_type> m_parameters;
};

//! The double-precision interpolant
using Interpolate = Interpolator<double>;

template <typename value_type>
bool operator==(const typename Interpolator<value_type>::point& lhs,
                const typename Interpolator<value_type>::point& rhs) {
  return lhs.x == rhs.x && lhs.f == rhs.f && lhs.f1 == rhs.f1;
}

inline bool operator==(const Interpolate::point& lhs, const Interpolate::point& rhs) {
  return lhs.x == rhs.x && lhs.f == rhs.f && lhs.f1 == rhs.f1;
}

template <typename value_type>
std::ostream& operator<<(std::ostream& os, const Interpolator<value_type>& interpolant) {
  for (const auto& parameter : interpolant.parameters())
    os << " " << parameter;
  return os;
}

template <typename value_type>
std::ostream& operator<<(std::ostream& os, const typename Interpolator<value_type>::point& p) {
  os << "x=" << p.x << ", value=" << p.f << ", gradient=" << p.f1 << ", curvature=" << p.f2;
  return os;
}

inline std::ostream& operator<<(std::ostream& os, const Interpolate::point& p) {
  os << "x=" << p.x << ", value=" << p.f << ", gradient=" << p.f1 << ", curvature=" << p.f2;
  return os;
}

extern template class Interpolator<double>;
extern template class Interpolator<long double>;

} // namespace molpro::linalg::itsolv
#endif // LINEARALGEBRA_SRC_MOLPRO_LINALG_ITSOLV_INTERPOLATE_H_
