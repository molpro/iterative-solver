#ifndef LINEARALGEBRA_SRC_MOLPRO_LINALG_ITSOLV_QSPACE_OPTIONS_H
#define LINEARALGEBRA_SRC_MOLPRO_LINALG_ITSOLV_QSPACE_OPTIONS_H

#include <cstddef>
#include <limits>

namespace molpro::linalg::itsolv {

struct QSpaceOptions {
  /// maximum size of Q space
  std::size_t max_size = std::numeric_limits<std::size_t>::max();
};

} // namespace molpro::linalg::itsolv

#endif // LINEARALGEBRA_SRC_MOLPRO_LINALG_ITSOLV_QSPACE_OPTIONS_H
