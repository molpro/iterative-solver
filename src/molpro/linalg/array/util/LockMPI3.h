#ifndef LINEARALGEBRA_SRC_MOLPRO_LINALG_ARRAY_UTIL_LOCKMPI3_H
#define LINEARALGEBRA_SRC_MOLPRO_LINALG_ARRAY_UTIL_LOCKMPI3_H
#include <memory>
#include <mpi.h>

namespace molpro::linalg::array::util {

//! Atomic lock allowing only one process to acquire it. Implemented using MPI3 RMA.
class LockMPI3 {
protected:
  MPI_Comm m_comm = MPI_COMM_NULL; //!< MPI communicator
  MPI_Win m_win = MPI_WIN_NULL;    //!< empty window handle
  bool m_locked = false;           //!< whether lock is active

public:
  //! Create the lock without acquiring it. This is collective and must be called by all processes in the communicator.
  LockMPI3(MPI_Comm comm);
  //! Release the lock and destroy it. This is collective and must be called by all processes in the communicator.
  ~LockMPI3();

  LockMPI3() = delete;
  LockMPI3(const LockMPI3 &) = delete;
  LockMPI3 &operator=(const LockMPI3 &) = delete;

  //! Acquire exclusive lock
  void lock();

  //! Release the lock
  void unlock();

protected:
  //! Proxy that locks on creation and unlocks on destruction. Useful for locking a scope.
  struct Proxy {
    LockMPI3 &m_lock;
    explicit Proxy(LockMPI3 &);
    Proxy() = delete;
    ~Proxy();

    bool m_deleted = false; //!< whether the lock was already deleted
  };

  //! Keep track of proxy object so that if lock is deleted, the proxy does not try to unlock.
  std::weak_ptr<Proxy> m_proxy;

public:
  //! Return a proxy object that acquires the lock on creation and releases it on destruction. Useful for locking a
  //! scope.
  std::shared_ptr<Proxy> scope();
};
} // namespace molpro::linalg::array::util

#endif // LINEARALGEBRA_SRC_MOLPRO_LINALG_ARRAY_UTIL_LOCKMPI3_H
