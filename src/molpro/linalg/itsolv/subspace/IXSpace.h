#ifndef LINEARALGEBRA_SRC_MOLPRO_LINALG_ITSOLV_SUBSPACE_IXSPACE_H
#define LINEARALGEBRA_SRC_MOLPRO_LINALG_ITSOLV_SUBSPACE_IXSPACE_H
#include <molpro/linalg/array/ArrayHandler.h>
#include <molpro/linalg/array/Span.h>
#include <molpro/linalg/itsolv/subspace/Dimensions.h>
#include <molpro/linalg/itsolv/subspace/SubspaceData.h>
#include <molpro/linalg/itsolv/wrap.h>

#include <memory>
#include <vector>

namespace molpro::linalg::itsolv {

class Logger;

namespace subspace {

//! Full subspace
template <class RT, class QT, class PT>
class IXSpace {
public:
  using R = RT;
  using Q = QT;
  using P = PT;
  using value_type = typename array::ArrayHandler<R, R>::value_type;
  using value_type_abs = typename array::ArrayHandler<R, R>::value_type_abs;
  IXSpace() = default;
  virtual ~IXSpace() = default;
  SubspaceData<value_type> data; //!< Equation data in the subspace

  //! Number of vectors forming the subspace
  size_t size() const { return dimensions().nX(); }

  //! Removes parameter i from the full subspace
  virtual void erase(size_t i) = 0;
  //! Removes parameter i from Q subspace
  virtual void eraseq(size_t i) = 0;
  //! Removes parameter i from P subspace
  virtual void erasep(size_t i) = 0;
  //! Removes parameter i from D subspace
  virtual void erased(size_t i) = 0;

  //! Adds parameters to the P space
  virtual void update_pspace(const CVecRef<P>& params, const array::Span<value_type>& pp_action_matrix) = 0;

  /*!
   * @brief Declares whether the subspace has a metric, which creates or removes the metric matrix EqnData::M. Only
   * possible while the subspace is empty, unless it does not change anything.
   */
  virtual void set_metric(bool metric) = 0;

  /*!
   * @brief Adds parameters to the Q space
   * @param params new parameters
   * @param actions their actions
   * @param metric_actions their actions of the metric, given exactly if there is a metric
   */
  virtual void update_qspace(const CVecRef<R>& params, const CVecRef<R>& actions,
                             const CVecRef<R>& metric_actions = {}) = 0;

  //! Updates D space with the new parameters, their actions and, if there is a metric, their actions of the metric
  virtual void update_dspace(VecRef<Q>& params, VecRef<Q>& actions, const VecRef<Q>& metric_actions = {}) = 0;

  virtual VecRef<P> paramsp() = 0;
  virtual VecRef<Q> paramsq() = 0;
  virtual VecRef<Q> actionsq() = 0;
  virtual VecRef<Q> paramsd() = 0;
  virtual VecRef<Q> actionsd() = 0;

  virtual CVecRef<P> paramsp() const = 0;
  virtual CVecRef<Q> paramsq() const = 0;
  virtual CVecRef<Q> actionsq() const = 0;
  virtual CVecRef<Q> paramsd() const = 0;
  virtual CVecRef<Q> actionsd() const = 0;

  virtual CVecRef<P> cparamsp() const = 0;
  virtual CVecRef<Q> cparamsq() const = 0;
  virtual CVecRef<Q> cactionsq() const = 0;
  virtual CVecRef<Q> cparamsd() const = 0;
  virtual CVecRef<Q> cactionsd() const = 0;

  //! Actions of the metric on the Q space parameters, empty unless there is a metric
  virtual CVecRef<Q> cmetric_actionsq() const = 0;
  //! Actions of the metric on the D space parameters, empty unless there is a metric
  virtual CVecRef<Q> cmetric_actionsd() const = 0;
  //! Whether the subspace holds the metric matrix, EqnData::M, of a generalised eigenproblem
  bool has_metric() const { return data.count(EqnData::M) > 0; }

  virtual const Dimensions& dimensions() const = 0;

  virtual void set_logger(std::shared_ptr<Logger> logger) = 0;
};

} // namespace subspace
} // namespace molpro::linalg::itsolv

#endif // LINEARALGEBRA_SRC_MOLPRO_LINALG_ITSOLV_SUBSPACE_IXSPACE_H
