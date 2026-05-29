#ifndef LINEARALGEBRA_SRC_MOLPRO_LINALG_ITSOLV_RSPACE_OPTIONS_H
#define LINEARALGEBRA_SRC_MOLPRO_LINALG_ITSOLV_RSPACE_OPTIONS_H

namespace molpro::linalg::itsolv {

struct RSpaceOptions {
  /// vectors with norm less than threshold can be considered null.
  double norm_thresh = 1e-10;
  /// the smallest singular value in the subspace that can be allowed when
  /// constructing the working set. Smaller singular values will lead to
  /// deletion of parameters from the Q space
  double svd_thresh = 1e-12;
};

} // namespace molpro::linalg::itsolv

#endif // LINEARALGEBRA_SRC_MOLPRO_LINALG_ITSOLV_RSPACE_OPTIONS_H
