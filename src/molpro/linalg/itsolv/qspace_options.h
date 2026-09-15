#ifndef LINEARALGEBRA_SRC_MOLPRO_LINALG_ITSOLV_QSPACE_OPTIONS_H
#define LINEARALGEBRA_SRC_MOLPRO_LINALG_ITSOLV_QSPACE_OPTIONS_H

#include <cstddef>
#include <limits>

namespace molpro::linalg::itsolv {

struct QSpaceOptions {
  /// maximum size of Q space
  std::size_t max_size = std::numeric_limits<std::size_t>::max();
  /// vectors having a smaller max. contribution to any of the current solutions
  /// will be considered redundant
  double contrib_thresh = 0;
  /// minimum size of Q space - if the Q space's size would fall below this
  /// value by deleting a vector, the deletion will be cancelled.
  std::size_t min_size = 0;
};

} // namespace molpro::linalg::itsolv

#endif // LINEARALGEBRA_SRC_MOLPRO_LINALG_ITSOLV_QSPACE_OPTIONS_H
