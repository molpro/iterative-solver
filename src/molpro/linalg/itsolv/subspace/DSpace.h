#ifndef LINEARALGEBRA_SRC_MOLPRO_LINALG_ITSOLV_SUBSPACE_DSPACE_H
#define LINEARALGEBRA_SRC_MOLPRO_LINALG_ITSOLV_SUBSPACE_DSPACE_H

#include <cassert>
#include <memory>
#include <molpro/linalg/array/ArrayHandler.h>
#include <molpro/linalg/itsolv/Logger.h>
#include <molpro/linalg/itsolv/subspace/Matrix.h>
#include <molpro/linalg/itsolv/wrap.h>

namespace molpro::linalg::itsolv::subspace {

/*!
 * @brief D space stores complement of P+Q+R subspace that is necessary to reproduce the previous solutions
 */
template <class Qt>
class DSpace {
public:
  using Q = Qt;
  using value_type = typename array::ArrayHandler<Q, Q>::value_type;
  using value_type_abs = typename array::ArrayHandler<Q, Q>::value_type_abs;

  explicit DSpace(std::shared_ptr<Logger> logger) : m_logger(std::move(logger)) {}

  //! Clears current D space and moves params and action, and any actions of the metric, inside.
  void update(VecRef<Q>& params, VecRef<Q>& actions, const VecRef<Q>& metric_actions = {}) {
    assert(params.size() == actions.size());
    assert(metric_actions.empty() || metric_actions.size() == params.size());
    clear();
    for (size_t i = 0; i < params.size(); ++i) {
      m_params.emplace_back(std::move(params[i].get()));
      m_actions.emplace_back(std::move(actions[i].get()));
    }
    for (auto& m : metric_actions)
      m_metric_actions.emplace_back(std::move(m.get()));
  }

  void clear() {
    m_params.clear();
    m_actions.clear();
    m_metric_actions.clear();
  }

  //! Erases parameter i. @param i index in the current space
  void erase(size_t i) {
    assert(m_params.size() > i);
    m_params.erase(begin(m_params) + i);
    m_actions.erase(begin(m_actions) + i);
    if (!m_metric_actions.empty())
      m_metric_actions.erase(begin(m_metric_actions) + i);
  }

  size_t size() const { return m_params.size(); }

  VecRef<Q> params() { return wrap(m_params); }
  CVecRef<Q> params() const { return wrap(m_params); }
  CVecRef<Q> cparams() const { return params(); }

  VecRef<Q> actions() { return wrap(m_actions); }
  CVecRef<Q> actions() const { return wrap(m_actions); }
  CVecRef<Q> cactions() const { return actions(); }

  //! Actions of the metric, empty unless there is a metric
  CVecRef<Q> cmetric_actions() const { return wrap(m_metric_actions); }

protected:
  std::shared_ptr<Logger> m_logger;
  std::vector<Q> m_params;         //!< D space parameters
  std::vector<Q> m_actions;        //!< D space actions
  std::vector<Q> m_metric_actions; //!< D space actions of the metric, empty unless there is a metric
};

} // namespace molpro::linalg::itsolv::subspace

#endif // LINEARALGEBRA_SRC_MOLPRO_LINALG_ITSOLV_SUBSPACE_DSPACE_H
