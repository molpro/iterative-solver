#ifndef LINEARALGEBRA_SRC_MOLPRO_LINALG_ITSOLV_RSPACE_OPTIONS_H
#define LINEARALGEBRA_SRC_MOLPRO_LINALG_ITSOLV_RSPACE_OPTIONS_H

#include <molpro/linalg/itsolv/helper.h>

namespace molpro::linalg::itsolv {

/*!
 * @brief Thresholds used when proposing the next set of R-space vectors.
 *
 * The defaults were calibrated for double precision and are rescaled to the working precision: left
 * at their double-precision values they would declare new directions null long before the arithmetic
 * had run out of accuracy.
 */
template <typename value_type_abs = double>
struct RSpaceOptions {
  /// vectors with norm less than threshold can be considered null.
  value_type_abs norm_thresh = precision_scaled<value_type_abs>(1e-10);
  /// the smallest singular value in the subspace that can be allowed when
  /// constructing the working set. Smaller singular values will lead to
  /// deletion of parameters from the Q space
  value_type_abs svd_thresh = precision_scaled<value_type_abs>(1e-12);
};

} // namespace molpro::linalg::itsolv

#endif // LINEARALGEBRA_SRC_MOLPRO_LINALG_ITSOLV_RSPACE_OPTIONS_H
