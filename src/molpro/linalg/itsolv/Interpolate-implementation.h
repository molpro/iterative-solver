#ifndef LINEARALGEBRA_SRC_MOLPRO_LINALG_ITSOLV_INTERPOLATE_IMPLEMENTATION_H_
#define LINEARALGEBRA_SRC_MOLPRO_LINALG_ITSOLV_INTERPOLATE_IMPLEMENTATION_H_

#include <molpro/linalg/itsolv/Interpolate.h>

#include <molpro/linalg/itsolv/IterativeSolver.h>
#include <molpro/linalg/itsolv/SolverFactory-implementation.h>
#include <molpro/linalg/itsolv/SolverFactory.h>
#include <molpro/linalg/itsolv/helper.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

namespace molpro::linalg::itsolv {
namespace interpolate_detail {

//! The Morse interpolant and its first two derivatives at y
template <typename value_type>
typename Interpolator<value_type>::point Morse(value_type y, const std::vector<value_type>& parameters) {
  using std::exp;
  using std::pow;
  typename Interpolator<value_type>::point result;
  result.x = y;
  result.f = parameters[0] +
             (parameters[1] / 2) * pow((1 - exp(-parameters[2] * (y - parameters[3]))) / parameters[2], 2);
  result.f1 = (parameters[1] / parameters[2]) * exp(-parameters[2] * (y - parameters[3])) *
              (1 - exp(-parameters[2] * (y - parameters[3])));
  result.f2 = -parameters[1] * (1 - 2 * exp(-parameters[2] * (y - parameters[3])));
  return result;
}

//! Fits the four Morse parameters to the two defining points
template <typename value_type>
class Morse_problem : public Problem<std::vector<value_type>> {
  using R = std::vector<value_type>;
  using point = typename Interpolator<value_type>::point;
  point p0, p1;

public:
  Morse_problem(point p0, point p1) {
    this->p0 = p0;
    this->p1 = p1;
  }

  typename Problem<R>::value_t residual(const R& parameters, R& residual) const override {
    auto pp0 = Morse<value_type>(p0.x, parameters);
    auto pp1 = Morse<value_type>(p1.x, parameters);
    residual[0] = pp0.f - p0.f;
    residual[1] = pp1.f - p1.f;
    residual[2] = pp0.f1 - p0.f1;
    residual[3] = pp1.f1 - p1.f1;
    return 0;
  }
};

} // namespace interpolate_detail

template <typename value_type>
Interpolator<value_type>::Interpolator(point p0, point p1, std::string interpolant, int verbosity)
    : m_p0(std::move(p0)), m_p1(std::move(p1)), m_interpolant(std::move(interpolant)), m_parameters(4) {
  using std::pow;
  if (m_interpolant == "cubic") {
    // c0 + c1(x-xbar) + c2(x-xbar)^2 + c3(x-xbar)^3 where xbar=(x0+x1)/2
    auto x1mx0 = m_p1.x - m_p0.x;
    auto f1pf0 = m_p1.f + m_p0.f;
    auto f1mf0 = m_p1.f - m_p0.f;
    auto g1pg0 = m_p1.f1 + m_p0.f1;
    auto g1mg0 = m_p1.f1 - m_p0.f1;
    m_parameters[0] = value_type(0.5) * f1pf0 - value_type(0.125) * g1mg0 * x1mx0;
    m_parameters[1] = value_type(-0.25) * g1pg0 + value_type(1.5) * f1mf0 / x1mx0;
    m_parameters[2] = value_type(0.5) * g1mg0 / x1mx0;
    m_parameters[3] = (-2 * f1mf0 + g1pg0 * x1mx0) / pow(x1mx0, 3);
  } else if (m_interpolant == "morse") {
    // L0 + (k/2a^2)*(1-exp(-a(y-y0)))^2
    // m_parameters: L0, k, a, y0
    using R = std::vector<value_type>;
    R residual(4);
    auto solver = molpro::linalg::itsolv::create_NonLinearEquations<R>("DIIS");
    auto cubic = Interpolator<value_type>(p0, p1, "cubic", 0);
    auto cubic_minimum = cubic.minimize(p0.x, p1.x);
    auto cubic_at_minimum = cubic(cubic_minimum.x);
    m_parameters[1] = cubic_at_minimum.f2;
    m_parameters[2] = -3 * cubic.parameters()[3] / (cubic_at_minimum.f2);
    m_parameters[3] = cubic_minimum.x;
    m_parameters[0] = cubic_at_minimum.f;
    interpolate_detail::Morse_problem<value_type> problem(p0, p1);
    solver->set_verbosity(verbosity);
    if (!solver->solve(m_parameters, residual, problem))
      throw std::runtime_error("Cannot find Morse interpolant");
    solver->solution(m_parameters, residual);
  } else
    throw std::runtime_error("Unknown interpolant: " + m_interpolant);
}

template <typename value_type>
std::vector<std::string> Interpolator<value_type>::interpolants() {
  return std::vector<std::string>{"cubic", "morse"};
}

template <typename value_type>
typename Interpolator<value_type>::point Interpolator<value_type>::operator()(value_type x) const {
  if (m_interpolant == "cubic") {
    auto xbar = value_type(0.5) * (m_p1.x + m_p0.x);
    auto f = m_parameters[0] +
             (x - xbar) * (m_parameters[1] + (x - xbar) * (m_parameters[2] + (x - xbar) * m_parameters[3]));
    auto f1 = m_parameters[1] + (x - xbar) * (2 * m_parameters[2] + 3 * (x - xbar) * m_parameters[3]);
    auto f2 = 2 * m_parameters[2] + 6 * (x - xbar) * m_parameters[3];
    return point{x, f, f1, f2};
  } else if (m_interpolant == "morse") {
    return interpolate_detail::Morse<value_type>(x, m_parameters);
  }
  throw std::logic_error("Unknown interpolant: " + m_interpolant);
}

template <typename value_type>
typename Interpolator<value_type>::point Interpolator<value_type>::minimize_cubic() const {
  using std::abs;
  using std::isnan;
  using std::sqrt;
  if (m_interpolant != "cubic")
    throw std::logic_error("minimize_cubic called with non-cubic interpolant");
  const auto c = m_parameters[1];
  const auto b = 2 * m_parameters[2];
  const auto a = 3 * m_parameters[3];
  // the criterion for "the cubic term is negligible" is a relative one, so it scales with the precision
  if (abs(a) < precision_scaled<value_type>(1e-10) * std::max(abs(c), abs(b))) { // quadratic not cubic
    return point{-c / b};
  }
  auto discriminant = b * b / (4 * a * a) - c / a;
  if (isnan(discriminant) || discriminant < 0)
    return {std::numeric_limits<value_type>::quiet_NaN()};
  auto xbar = value_type(0.5) * (m_p1.x + m_p0.x);
  point pm = (*this)(xbar - (b / (2 * a)) + sqrt(discriminant));
  point pp = (*this)(xbar - (b / (2 * a)) - sqrt(discriminant));
  return pm.f < pp.f ? pm : pp;
}

template <typename value_type>
typename Interpolator<value_type>::point Interpolator<value_type>::minimize(value_type xa, value_type xb,
                                                                           size_t bracket_grid,
                                                                           size_t max_bracket_grid,
                                                                           bool analytic) const {
  using std::abs;
  if (xa > xb)
    std::swap(xa, xb);
  if (analytic && m_interpolant == "cubic")
    return minimize_cubic();
  for (size_t ngrid = bracket_grid; ngrid < std::max(bracket_grid, max_bracket_grid) + 1; ngrid *= 2) {
    auto gridstep = (xb - xa) / ngrid;
    auto plow = (*this)(xa);
    auto p0 = (*this)(xa).f > (*this)(xb).f ? plow : (*this)(xb);
    auto p1 = p0;
    for (size_t igrid = 0; igrid < ngrid; igrid++) {
      auto phigh = (*this)(plow.x + gridstep);
      if (std::min(phigh.f, plow.f) < p0.f and plow.f1 <= 0 and phigh.f1 >= 0) {
        p1 = phigh;
        p0 = plow;
      }
      std::swap(plow, phigh);
    }
    if (p0.f1 < 0 and p1.f1 > 0) {
      auto pnew = p1;
      // a couple of units in the last place of pnew.x, expressed without std::nextafter so that it
      // is also defined for arbitrary-precision types
      auto tolerance =
          2 * std::numeric_limits<value_type>::epsilon() * std::max(value_type(1), abs(pnew.x));
      while (abs(p0.x - pnew.x) > tolerance) {
        pnew = (*this)((p1.x * p0.f1 - p0.x * p1.f1) / (p0.f1 - p1.f1));
        if (pnew.f1 * p0.f1 < 0)
          std::swap(p0, p1);
        std::swap(p0, pnew);
      }
      return p0;
    }
  }
  // nothing found; return lowest end point
  return (*this)(xa).f > (*this)(xb).f ? (*this)(xb) : (*this)(xa);
}

} // namespace molpro::linalg::itsolv

#endif // LINEARALGEBRA_SRC_MOLPRO_LINALG_ITSOLV_INTERPOLATE_IMPLEMENTATION_H_
