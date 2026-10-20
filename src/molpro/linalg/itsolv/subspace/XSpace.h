#ifndef LINEARALGEBRA_SRC_MOLPRO_LINALG_ITSOLV_SUBSPACE_XSPACE_H
#define LINEARALGEBRA_SRC_MOLPRO_LINALG_ITSOLV_SUBSPACE_XSPACE_H
#include <Eigen/Dense>
#include <molpro/linalg/itsolv/helper.h>
#include <molpro/linalg/itsolv/subspace/DSpace.h>
#include <molpro/linalg/itsolv/subspace/Dimensions.h>
#include <molpro/linalg/itsolv/subspace/IXSpace.h>
#include <molpro/linalg/itsolv/subspace/PSpace.h>
#include <molpro/linalg/itsolv/subspace/QSpace.h>
#include <molpro/linalg/itsolv/subspace/util.h>

#include <algorithm>
#include <cassert>
#include <functional>
#include <memory>
#include <stdexcept>
#include <vector>

namespace molpro::linalg::itsolv::subspace {
namespace xspace {
//! New sections of equation data
template <typename T>
struct NewData {
  //! @param metric whether there are also blocks of the metric matrix, EqnData::M
  NewData(size_t nQnew, size_t nX, size_t nRHS, bool metric = false) {
    if (metric)
      for (auto* blocks : {&qq, &qx, &xq})
        (*blocks)[EqnData::M] = {};
    for (auto d : square_blocks(qq)) {
      qq[d].resize({nQnew, nQnew});
      qx[d].resize({nQnew, nX});
      xq[d].resize({nX, nQnew});
    }
    qq[EqnData::rhs].resize({nQnew, nRHS});
  }

  SubspaceData<T> qq = null_data<T, EqnData::H, EqnData::S>(); //!< data block between new paramters
  SubspaceData<T> qx = null_data<T, EqnData::H, EqnData::S>(); //!< data block between new parameters and current X space
  SubspaceData<T> xq = null_data<T, EqnData::H, EqnData::S>(); //!< data block between current X space and new parameters
};

/*!
 * @brief Returns new sections of equation data
 *
 * If metric_actions is not empty, the blocks of the metric matrix, EqnData::M, are calculated too, from the actions of
 * the metric on the new parameters alone: as the metric M is Hermitian, <x|M b> = <M x|b> for a new parameter x and
 * any subspace vector b. These are calculated together with the overlaps <x|b>, so that each b is read once.
 */
template <class R, class Q, class P>
auto update_qspace_data(const CVecRef<R>& params, const CVecRef<R>& actions, const CVecRef<P>& pparams,
                        const CVecRef<Q>& qparams, const CVecRef<Q>& qactions, const CVecRef<Q>& dparams,
                        const CVecRef<Q>& dactions, const CVecRef<Q>& rhs, const Dimensions& dims,
                        ArrayHandlers<R, Q, P>& handlers, Logger& logger, bool hermitian = false,
                        bool action_dot_action = false, const CVecRef<R>& metric_actions = {}) {
  auto nQnew = params.size();
  const bool metric = !metric_actions.empty();
  auto data = NewData<typename array::ArrayHandler<R, R>::value_type>(nQnew, dims.nX(), rhs.size(), metric);
  auto& qq = data.qq;
  auto& qx = data.qx;
  auto& xq = data.xq;
  qq[EqnData::S] = util::overlap(params, handlers.rr());
  // rows nQnew and beyond of the overlaps with the subspace vectors are those of the actions of the metric
  auto left = params;
  left.insert(left.end(), metric_actions.begin(), metric_actions.end());
  auto copy_overlaps = [&](const auto& overlaps, const size_t offset, const size_t size) {
    qx[EqnData::S].slice({0, offset}, {nQnew, offset + size}) = overlaps.slice({0, 0}, {nQnew, size});
    if (metric)
      qx[EqnData::M].slice({0, offset}, {nQnew, offset + size}) = overlaps.slice({nQnew, 0}, {2 * nQnew, size});
  };
  copy_overlaps(util::overlap(left, pparams, handlers.rp()), dims.oP(), dims.nP);
  copy_overlaps(util::overlap(left, qparams, handlers.rq()), dims.oQ(), dims.nQ);
  copy_overlaps(util::overlap(left, dparams, handlers.rq()), dims.oD(), dims.nD);
  qq[EqnData::H] =
      action_dot_action ? util::overlap(actions, handlers.rr()) : util::overlap(params, actions, handlers.rr());
  // left-hand vectors of the H matrix elements with existing actions
  const auto& h_left = action_dot_action ? actions : params;
  qx[EqnData::H].slice({0, dims.oQ()}, {nQnew, dims.oQ() + dims.nQ}) = util::overlap(h_left, qactions, handlers.rq());
  qx[EqnData::H].slice({0, dims.oD()}, {nQnew, dims.oD() + dims.nD}) = util::overlap(h_left, dactions, handlers.rq());
  if (hermitian) {
    xq[EqnData::H].slice({dims.oP(), 0}, {dims.oP() + dims.nP, nQnew}) = util::overlap(pparams, actions, handlers.rp());
    //    xq[EqnData::H].slice({dims.oQ(), 0}, {dims.oQ() + dims.nQ, nQnew}) = util::overlap(action_dot_action ? qactions :
    //    qparams, actions, handlers.qr());
    conjugate_transpose_copy(xq[EqnData::H].slice({dims.oQ(), 0}, {dims.oQ() + dims.nQ, nQnew}),
                   qx[EqnData::H].slice({0, dims.oQ()}, {nQnew, dims.oQ() + dims.nQ}));
    conjugate_transpose_copy(xq[EqnData::H].slice({dims.oD(), 0}, {dims.oD() + dims.nD, nQnew}),
                   qx[EqnData::H].slice({0, dims.oD()}, {nQnew, dims.oD() + dims.nD}));
    conjugate_transpose_copy(qx[EqnData::H].slice({0, dims.oP()}, {nQnew, dims.oP() + dims.nP}),
                   xq[EqnData::H].slice({dims.oP(), 0}, {dims.oP() + dims.nP, nQnew}));
  } else {
    xq[EqnData::H].slice({dims.oQ(), 0}, {dims.oQ() + dims.nQ, nQnew}) = util::overlap(qparams, actions, handlers.rq());
    xq[EqnData::H].slice({dims.oD(), 0}, {dims.oD() + dims.nD, nQnew}) = util::overlap(dparams, actions, handlers.rq());
  }
  if (metric) {
    qq[EqnData::M] = util::overlap(params, metric_actions, handlers.rr());
    conjugate_transpose_copy(xq[EqnData::M].slice(), qx[EqnData::M].slice());
  }
  qq[EqnData::rhs] = util::overlap(params, rhs, handlers.rq());
  conjugate_transpose_copy(xq[EqnData::S].slice({dims.oP(), 0}, {dims.oP() + dims.nP, nQnew}),
                 qx[EqnData::S].slice({0, dims.oP()}, {nQnew, dims.oP() + dims.nP}));
  conjugate_transpose_copy(xq[EqnData::S].slice({dims.oQ(), 0}, {dims.oQ() + dims.nQ, nQnew}),
                 qx[EqnData::S].slice({0, dims.oQ()}, {nQnew, dims.oQ() + dims.nQ}));
  conjugate_transpose_copy(xq[EqnData::S].slice({dims.oD(), 0}, {dims.oD() + dims.nD, nQnew}),
                 qx[EqnData::S].slice({0, dims.oD()}, {nQnew, dims.oD() + dims.nD}));
  logger.data_dump("xspace::update_qspace_data() nQnew = ", nQnew);
  logger.data_dump("Sqq = ", qq[EqnData::S]);
  logger.data_dump("Hqq = ", qq[EqnData::H]);
  logger.data_dump("Sqx = ", qx[EqnData::S]);
  logger.data_dump("Hqx = ", qx[EqnData::H]);
  logger.data_dump("Sxq = ", xq[EqnData::S]);
  logger.data_dump("Hxq = ", xq[EqnData::H]);
  if (metric) {
    logger.data_dump("Mqq = ", qq[EqnData::M]);
    logger.data_dump("Mqx = ", qx[EqnData::M]);
  }
  logger.data_dump("rhs_q = ", qq[EqnData::rhs]);
  return data;
}

//! Calculates overlap blocks between D space and the rest of the subspace
template <class Q, class P>
auto update_dspace_overlap_data(const CVecRef<P>& pparams, const CVecRef<Q>& qparams, const CVecRef<Q>& dparams,
                                const CVecRef<Q>& rhs, array::ArrayHandler<Q, P>& handler_qp,
                                array::ArrayHandler<Q, Q>& handler_qq, Logger& logger) {
  const auto nP = pparams.size();
  const auto nQ = qparams.size();
  const auto nX = nP + nQ;
  auto nD = dparams.size();
  const auto nRHS = rhs.size();
  auto data = NewData<typename array::ArrayHandler<Q, Q>::value_type>(nD, nX, nRHS);
  data.qq[EqnData::S].slice() = util::overlap(dparams, handler_qq);
  data.qx[EqnData::S].slice({0, 0}, {nD, nP}) = util::overlap(dparams, pparams, handler_qp);
  data.qx[EqnData::S].slice({0, nP}, {nD, nX}) = util::overlap(dparams, qparams, handler_qq);
  data.qq[EqnData::rhs].slice() = util::overlap(dparams, rhs, handler_qq);
  conjugate_transpose_copy(data.xq[EqnData::S].slice(), data.qx[EqnData::S].slice());
  logger.data_dump("xspace::update_dspace_overlap_data() nD = ", nD);
  logger.data_dump("Sdd = ", data.qq[EqnData::S]);
  logger.data_dump("Sdx = ", data.qx[EqnData::S]);
  logger.data_dump("rhs_d = ", data.qq[EqnData::rhs]);
  return data;
}

//! Calculates action blocks between D space and the rest of the subspace
template <class Q, class P>
auto update_dspace_action_data(const CVecRef<P>& pparams, const CVecRef<Q>& qparams, const CVecRef<Q>& qactions,
                               const CVecRef<Q>& dparams, const CVecRef<Q>& dactions,
                               array::ArrayHandler<Q, P>& handler_qp, array::ArrayHandler<Q, Q>& handler_qq,
                               Logger& logger) {
  const auto nP = pparams.size();
  const auto nQ = qparams.size();
  const auto nX = nP + nQ;
  auto nD = dparams.size();
  auto data = NewData<typename array::ArrayHandler<Q, Q>::value_type>(nD, nX, 0);
  const auto e = EqnData::H;
  data.qq[e].slice() = util::overlap(dparams, dactions, handler_qq);
  data.xq[e].slice({0, 0}, {nP, nD}) = util::overlap(pparams, dactions, handler_qp);
  data.xq[e].slice({nP, 0}, {nX, nD}) = util::overlap(qparams, dactions, handler_qq);
  data.qx[e].slice({0, nP}, {nD, nX}) = util::overlap(dparams, qactions, handler_qq);
  conjugate_transpose_copy(data.qx[e].slice({0, 0}, {nD, nP}), data.xq[e].slice({0, 0}, {nP, nD}));
  logger.data_dump("xspace::update_dspace_action_data() nD = ", nD);
  logger.data_dump("Hdd = ", data.qq[e]);
  logger.data_dump("Hdx = ", data.qx[e]);
  logger.data_dump("Hxd = ", data.xq[e]);
  return data;
}

/*!
 * @brief The metric matrix of the subspace after the deletion of Q vectors and their replacement, together with the
 * old D vectors, by new D vectors, obtained by transforming the current metric matrix rather than from the vectors
 * @param metric the current metric matrix
 * @param dims the current dimensions
 * @param q_delete indices of the Q vectors to be deleted
 * @param dcoefficients the coefficients of the new D vectors, as returned by construct_dspace()
 * @return the metric matrix of P, the remaining Q, and the new D vectors, in this order
 */
template <typename value_type>
Matrix<value_type> metric_with_new_dspace(const Matrix<value_type>& metric, const Dimensions& dims,
                                          const std::vector<std::size_t>& q_delete,
                                          const Matrix<value_type>& dcoefficients) {
  using matrix_type = Eigen::Matrix<value_type, Eigen::Dynamic, Eigen::Dynamic>;
  using row_major_type = Eigen::Matrix<value_type, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
  const auto nQnew = dims.nQ - q_delete.size();
  const auto nXnew = dims.nP + nQnew + dcoefficients.rows();
  // columns: the new subspace vectors in terms of the current ones
  matrix_type transformation = matrix_type::Zero(dims.nX(), nXnew);
  for (size_t i = 0; i < dims.nP; ++i)
    transformation(dims.oP() + i, i) = 1;
  for (size_t i = 0, k = 0; i < dims.nQ; ++i)
    if (std::find(q_delete.begin(), q_delete.end(), i) == q_delete.end())
      transformation(dims.oQ() + i, dims.nP + k++) = 1;
  for (size_t i = 0; i < dcoefficients.rows(); ++i) {
    for (size_t j = 0; j < q_delete.size(); ++j)
      transformation(dims.oQ() + q_delete[j], dims.nP + nQnew + i) = dcoefficients(i, j);
    for (size_t j = 0; j < dims.nD; ++j)
      transformation(dims.oD() + j, dims.nP + nQnew + i) = dcoefficients(i, q_delete.size() + j);
  }
  const matrix_type m = Eigen::Map<const row_major_type>(metric.data().data(), dims.nX(), dims.nX());
  const row_major_type m_new = transformation.adjoint() * m * transformation;
  return Matrix<value_type>(std::vector<value_type>(m_new.data(), m_new.data() + m_new.size()), {nXnew, nXnew});
}

template <typename T>
void copy_dspace_eqn_data(const NewData<T>& new_data, SubspaceData<T>& data, const subspace::EqnData e,
                          const Dimensions& dims) {
  const auto& dd = new_data.qq.at(e);
  const auto& dx = new_data.qx.at(e);
  const auto& xd = new_data.xq.at(e);
  data[e].slice({dims.oD(), dims.oD()}, {dims.oD() + dims.nD, dims.oD() + dims.nD}) = dd;
  data[e].slice({dims.oD(), dims.oP()}, {dims.oD() + dims.nD, dims.oP() + dims.nP}) =
      dx.slice({0, 0}, {dims.nD, dims.nP});
  data[e].slice({dims.oD(), dims.oQ()}, {dims.oD() + dims.nD, dims.oQ() + dims.nQ}) =
      dx.slice({0, dims.nP}, {dims.nD, dims.nP + dims.nQ});
  data[e].slice({dims.oP(), dims.oD()}, {dims.oP() + dims.nP, dims.oD() + dims.nD}) =
      xd.slice({0, 0}, {dims.nP, dims.nD});
  data[e].slice({dims.oQ(), dims.oD()}, {dims.oQ() + dims.nQ, dims.oD() + dims.nD}) =
      xd.slice({dims.nP, 0}, {dims.nP + dims.nQ, dims.nD});
}
} // namespace xspace

template <class R, class Q, class P>
class XSpace : public IXSpace<R, Q, P> {
public:
  using typename IXSpace<R, Q, P>::value_type;
  using typename IXSpace<R, Q, P>::value_type_abs;
  using IXSpace<R, Q, P>::data;
  using IXSpace<R, Q, P>::has_metric;
  using IXSpace<R, Q, P>::stores_metric_actions;

  explicit XSpace(const std::shared_ptr<ArrayHandlers<R, Q, P>>& handlers, const std::shared_ptr<Logger>& logger)
      : pspace(), qspace(handlers, logger), dspace(logger), m_handlers(handlers), m_logger(logger) {
    data = null_data<value_type, EqnData::H, EqnData::S, EqnData::rhs>();
  };

  void set_metric(MetricKind kind, MetricStorage storage = MetricStorage::Stored) override {
    if (kind == m_metric_kind && storage == m_metric_storage)
      return;
    if (m_dim.nX() != 0)
      throw std::logic_error("XSpace::set_metric: the metric cannot be changed once the subspace is not empty");
    m_metric_kind = kind;
    m_metric_storage = storage;
    if (has_metric())
      data[EqnData::M] = {};
    else
      data.erase(EqnData::M);
  }

  MetricKind metric_kind() const override { return m_metric_kind; }
  MetricStorage metric_storage() const override { return m_metric_storage; }

  //! Update parameters in Q space and corresponding equation data
  void update_qspace(const CVecRef<R>& params, const CVecRef<R>& actions,
                     const CVecRef<R>& metric_actions = {}) override {
    m_logger->trace("QSpace::update_qspace");
    if (metric_actions.size() != (has_metric() ? params.size() : 0))
      throw std::runtime_error(
          "XSpace::update_qspace: actions of the metric must be given for every parameter exactly when there is a "
          "metric");
    auto new_data = xspace::update_qspace_data(params, actions, cparamsp(), cparamsq(), cactionsq(), cparamsd(),
                                               cactionsd(), cwrap(m_rhs), m_dim, *m_handlers, *m_logger, m_hermitian,
                                               m_action_dot_action, metric_actions);
    qspace.update(params, actions, new_data.qq, new_data.qx, new_data.xq, m_dim, data,
                  stores_metric_actions() ? metric_actions : CVecRef<R>{});
    update_dimensions();
  }

  //! Replaces the old D space and the Q vectors q_delete by new D vectors, see IXSpace::update_dspace()
  void update_dspace(VecRef<Q>& params, VecRef<Q>& actions, const VecRef<Q>& metric_actions = {},
                     const std::vector<size_t>& q_delete = {}, const Matrix<value_type>& coefficients = {}) override {
    if (metric_actions.size() != (stores_metric_actions() ? params.size() : 0))
      throw std::invalid_argument(
          "XSpace::update_dspace: actions of the metric must be given exactly when they are stored");
    if (has_metric() && !params.empty() &&
        (coefficients.rows() != params.size() || coefficients.cols() != q_delete.size() + m_dim.nD))
      throw std::invalid_argument("XSpace::update_dspace: the coefficients of the new D vectors are needed");
    // the metric matrix of the new subspace is transformed from the current one, before the subspace changes
    const auto metric_matrix = has_metric() && !params.empty()
                                   ? xspace::metric_with_new_dspace(data.at(EqnData::M), m_dim, q_delete, coefficients)
                                   : Matrix<value_type>{};
    auto q_delete_descending = q_delete;
    std::sort(q_delete_descending.begin(), q_delete_descending.end(), std::greater<>());
    for (auto iq : q_delete_descending)
      eraseq(iq);
    dspace.update(params, actions, metric_actions);
    update_dimensions();
    for (auto e : square_blocks(data))
      data[e].resize({m_dim.nX(), m_dim.nX()});
    auto new_data = xspace::update_dspace_overlap_data(cparamsp(), cparamsq(), cparamsd(), cwrap(m_rhs),
                                                       m_handlers->qp(), m_handlers->qq(), *m_logger);
    xspace::copy_dspace_eqn_data(new_data, data, EqnData::S, m_dim);
    auto new_data_action = xspace::update_dspace_action_data(
        cparamsp(), cparamsq(), cactionsq(), cparamsd(), cactionsd(), m_handlers->qp(), m_handlers->qq(), *m_logger);
    xspace::copy_dspace_eqn_data(new_data_action, data, EqnData::H, m_dim);
    if (has_metric() && m_dim.nD > 0)
      data[EqnData::M] = metric_matrix;
    data[EqnData::rhs].resize({m_dim.nX(), m_dim.nRHS});
    data[EqnData::rhs].slice({m_dim.oD(), 0}, {m_dim.oD() + m_dim.nD, m_dim.nRHS}) = new_data.qq[EqnData::rhs].slice();
  }

  // FIXME this must be called when XSpace is empty
  //! @warning Subspace should be empty
  void update_pspace(const CVecRef<P>& params, const array::Span<value_type>& pp_action_matrix,
                     const array::Span<value_type>& pp_metric_matrix = {}) override {
    assert(m_dim.nX() == 0);
    if (!m_hermitian)
      throw std::runtime_error("P space can only be used with hermitian kernels");
    const size_t nPnew = params.size();
    if (pp_metric_matrix.size() != (has_metric() ? nPnew * nPnew : 0))
      throw std::runtime_error("XSpace::update_pspace: the metric matrix in the P space must be given exactly when "
                               "there is a metric");
    pspace.update(params, m_handlers->pp());
    update_dimensions();
    const size_t nP = m_dim.nP;
    update_rhs_with_pspace();
    data[EqnData::S].resize({nP, nP});
    data[EqnData::H].resize({nP, nP});
    data[EqnData::S].slice() = util::overlap(params, m_handlers->pp());
    for (size_t i = 0, ij = 0; i < nP; ++i)
      for (size_t j = 0; j < nP; ++j, ++ij)
        data[EqnData::H](i, j) = pp_action_matrix[ij];
    if (has_metric()) {
      data[EqnData::M].resize({nP, nP});
      for (size_t i = 0, ij = 0; i < nP; ++i)
        for (size_t j = 0; j < nP; ++j, ++ij)
          data[EqnData::M](i, j) = pp_metric_matrix[ij];
    }
  }

  //! For a system of linear equations Ax=b, adds rhs vectors b.
  void add_rhs_equations(const CVecRef<R>& rhs) {
    m_rhs.reserve(m_rhs.size() + rhs.size());
    for (const auto& r : rhs)
      m_rhs.emplace_back(this->m_handlers->qr().copy(r));
    for (const auto& r : rhs) {
      auto d = std::abs(this->m_handlers->rr().dot(r, r));
      if (d == 0)
        throw std::runtime_error("RHS vector cannot be zero");
      m_rhs_norm.emplace_back(std::sqrt(d));
    }
    update_dimensions();
    update_rhs_with_pspace();
  }

  //! Access RHS vectors in linear equations
  CVecRef<Q> rhs() const { return cwrap(m_rhs); }

  //! Norm of RHS vectors
  const std::vector<value_type_abs>& rhs_norm() const { return m_rhs_norm; }

  const Dimensions& dimensions() const override { return m_dim; }

  void erase(size_t i) override {
    if (i >= m_dim.oP() && i < m_dim.oP() + m_dim.nP) {
      erasep(i - m_dim.oP());
    } else if (i >= m_dim.oQ() && i < m_dim.oQ() + m_dim.nQ) {
      eraseq(i - m_dim.oQ());
    } else if (i >= m_dim.oD() && i < m_dim.oD() + m_dim.nD) {
      erased(i - m_dim.oD());
    }
  }

  void eraseq(size_t i) override {
    qspace.erase(i);
    remove_data(m_dim.oQ() + i);
    update_dimensions();
  }

  void erasep(size_t i) override {
    pspace.erase(i);
    remove_data(m_dim.oP() + i);
    update_dimensions();
  }

  void erased(size_t i) override {
    dspace.erase(i);
    remove_data(m_dim.oD() + i);
    update_dimensions();
  }

  VecRef<P> paramsp() override { return pspace.params(); }
  VecRef<Q> paramsq() override { return qspace.params(); }
  VecRef<Q> actionsq() override { return qspace.actions(); }
  VecRef<Q> paramsd() override { return dspace.params(); }
  VecRef<Q> actionsd() override { return dspace.actions(); }

  CVecRef<P> paramsp() const override { return pspace.params(); }
  CVecRef<Q> paramsq() const override { return qspace.params(); }
  CVecRef<Q> actionsq() const override { return qspace.actions(); }
  CVecRef<Q> paramsd() const override { return dspace.cparams(); }
  CVecRef<Q> actionsd() const override { return dspace.cactions(); }

  CVecRef<P> cparamsp() const override { return pspace.cparams(); }
  CVecRef<Q> cparamsq() const override { return qspace.cparams(); }
  CVecRef<Q> cactionsq() const override { return qspace.cactions(); }
  CVecRef<Q> cparamsd() const override { return dspace.cparams(); }
  CVecRef<Q> cactionsd() const override { return dspace.cactions(); }

  CVecRef<Q> cmetric_actionsq() const override { return qspace.cmetric_actions(); }
  CVecRef<Q> cmetric_actionsd() const override { return dspace.cmetric_actions(); }

  void set_logger(std::shared_ptr<Logger> logger) override { m_logger = std::move(logger); }

  //! Set Hermiticity of the subspace. P space can only be used with Hermitian problems
  void set_hermiticity(bool hermitian) { m_hermitian = hermitian; }
  bool get_hermiticity() { return m_hermitian; }
  void set_action_action() { m_action_dot_action = true; }

  PSpace<R, P> pspace;
  QSpace<R, Q, P> qspace;
  DSpace<Q> dspace;

protected:
  void update_dimensions() {
    m_dim = Dimensions(pspace.size(), qspace.size(), dspace.size());
    m_dim.nRHS = m_rhs.size();
  }

  //! Update projection of RHS data onto P space. @warning Subspace should contain only P space
  auto update_rhs_with_pspace() {
    data[EqnData::rhs].resize({m_dim.nP, m_dim.nRHS});
    data[EqnData::rhs].slice() = util::overlap(cparamsp(), rhs(), m_handlers->qp());
  };

  void remove_data(size_t i) {
    for (auto d : square_blocks(data))
      data[d].remove_row_col(i, i);
    if (data.find(EqnData::rhs) != std::end(data) && !data[EqnData::rhs].empty())
      data[EqnData::rhs].remove_row(i);
    if (data.find(EqnData::value) != std::end(data) && !data[EqnData::value].empty())
      data[EqnData::value].remove_row(i);
  }
  std::shared_ptr<ArrayHandlers<R, Q, P>> m_handlers;
  std::shared_ptr<Logger> m_logger;
  Dimensions m_dim;
  std::vector<Q> m_rhs;                   //!< Right hand side vectors
  std::vector<value_type_abs> m_rhs_norm; //!< norm of RHS vectors
  bool m_hermitian = false;               //!< whether the matrix is Hermitian
  bool m_action_dot_action = false;       //!< whether the H matrix is action.action instead of action.parameter
  MetricKind m_metric_kind = MetricKind::Identity;        //!< the metric declared with set_metric()
  MetricStorage m_metric_storage = MetricStorage::Stored; //!< whether the actions of the metric are kept
};

} // namespace molpro::linalg::itsolv::subspace
#endif // LINEARALGEBRA_SRC_MOLPRO_LINALG_ITSOLV_SUBSPACE_XSPACE_H
