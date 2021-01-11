#ifndef LINEARALGEBRA_SRC_MOLPRO_LINALG_ARRAY_UTIL_SCOPELOCK_H
#define LINEARALGEBRA_SRC_MOLPRO_LINALG_ARRAY_UTIL_SCOPELOCK_H
#include <molpro/linalg/array/util/LockMPI3.h>
namespace molpro::linalg::array::util {

//! Utility object that locks on creation and unlocks on destruction
class ScopeLock {
public:
  explicit ScopeLock(MPI_Comm comm) : lock{comm}, l{lock.scope()} {}

protected:
  LockMPI3 lock;
  decltype(std::declval<LockMPI3>().scope()) l;
};

} // namespace molpro::linalg::array::util

#endif // LINEARALGEBRA_SRC_MOLPRO_LINALG_ARRAY_UTIL_SCOPELOCK_H
