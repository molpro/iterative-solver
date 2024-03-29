#ifndef LINEARALGEBRA_SRC_MOLPRO_LINALG_ARRAY_ARRAYHANDLERDISTRSPARSE_H
#define LINEARALGEBRA_SRC_MOLPRO_LINALG_ARRAY_ARRAYHANDLERDISTRSPARSE_H
#include "molpro/linalg/array/ArrayHandler.h"
#include <molpro/linalg/array/util/gemm.h>
#include <stdexcept>

using molpro::linalg::array::util::gemm_inner_distr_sparse;
using molpro::linalg::array::util::gemm_outer_distr_sparse;

namespace molpro::linalg::array {
template <typename AL, typename AR, bool = has_mapped_type_v<AR>>
class ArrayHandlerDistrSparse : public ArrayHandler<AL, AR> {};

/*!
 * @brief Array handler between an iterable and a sparse arrays. Iterable container must implement access operator.
 */
template <typename AL, typename AR>
class ArrayHandlerDistrSparse<AL, AR, true> : public ArrayHandler<AL, AR> {
public:
  using typename ArrayHandler<AL, AR>::value_type_L;
  using typename ArrayHandler<AL, AR>::value_type_R;
  using typename ArrayHandler<AL, AR>::value_type;
  using typename ArrayHandler<AL, AR>::value_type_abs;
  using typename ArrayHandler<AL, AR>::ProxyHandle;

  AL copy(const AR &source) override {
    throw std::logic_error("General construction of sparse from dense is ill-defined");
  };

  void copy(AL &x, const AR &y) override {
    //    static_assert(true, "General copy from sparse to dense is ill-defined");
    x.fill(0);
    auto lb = x.local_buffer();
    auto start = lb->start();
    for (const auto &v : y)
      if (v.first >= start and v.first < start + lb->size())
        (*lb)[v.first - start] = v.second;
  };

  void scal(value_type alpha, AL &x) override { static_assert(true, "Use ArrayHandlerDistr for unary operations"); };

  void fill(value_type alpha, AL &x) override { static_assert(true, "Use ArrayHandlerDistr for unary operations"); };

  // TODO For now these functions are in DistrArray, but they should be moved into the handler.
  void axpy(value_type alpha, const AR &x, AL &y) override {
    this->m_counter->axpy++;
    y.axpy(alpha, x);
  };

  value_type dot(const AL &x, const AR &y) override {
    this->m_counter->dot++;
    return x.dot(y);
  };

  void gemm_outer(const Matrix<value_type> alphas, const CVecRef<AR> &xx, const VecRef<AL> &yy) override {
    this->m_counter->gemm_outer++;
    gemm_outer_distr_sparse(alphas, xx, yy);
  }

  Matrix<value_type> gemm_inner(const CVecRef<AL> &xx, const CVecRef<AR> &yy) override {
    this->m_counter->gemm_inner++;
    return gemm_inner_distr_sparse(xx, yy);
  }

  std::map<size_t, value_type_abs> select_max_dot(size_t n, const AL &x, const AR &y) override {
    return x.select_max_dot(n, y);
  }

  std::map<size_t, value_type_abs> select(size_t n, const AL &x, bool max = false, bool ignore_sign = false) override {
    return x.select(n, max, ignore_sign);
  }

  ProxyHandle lazy_handle() override { return this->lazy_handle(*this); };

protected:
  using ArrayHandler<AL, AR>::error;
  using ArrayHandler<AL, AR>::lazy_handle;
  using ArrayHandler<AL, AR>::m_lazy_handles;
};

} // namespace molpro::linalg::array

#endif // LINEARALGEBRA_SRC_MOLPRO_LINALG_ARRAY_ARRAYHANDLERDISTRSPARSE_H
