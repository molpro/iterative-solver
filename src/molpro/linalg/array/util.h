#ifndef LINEARALGEBRA_SRC_MOLPRO_LINALG_ARRAY_UTIL_H
#define LINEARALGEBRA_SRC_MOLPRO_LINALG_ARRAY_UTIL_H
#include <atomic>
#include <future>
#include <memory>
#include <string>
#include <vector>

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

//! Utility object that locks on creation and unlocks on destruction
class ScopeLock {
public:
  explicit ScopeLock(MPI_Comm comm) : lock{comm}, l{lock.scope()} {}

protected:
  LockMPI3 lock;
  decltype(std::declval<LockMPI3>().scope()) l;
};

template <typename Result = void>
class Task {
public:
  Task(std::future<Result> &&task) : m_task{std::move(task)} {};
  Task(Task &&other) = default;

  template <class Func, typename... Args>
  static Task create(Func &&f, Args &&...args) {
    return {std::async(std::launch::async, std::forward<Func>(f), std::forward<Args>(args)...)};
  }

  ~Task() { wait(); }

  //! Returns true if the task has completed
  bool test() { return m_task.wait_for(std::chrono::microseconds{1}) == std::future_status::ready; }
  //! wait for the task to complete and return its result
  Result wait() {
    if (!m_task.valid())
      throw std::future_error(std::future_errc::no_state);
    return m_task.get();
  }

protected:
  std::future<Result> m_task; //! Future holding the thread
};

} // namespace molpro::linalg::array::util

#endif // LINEARALGEBRA_SRC_MOLPRO_LINALG_ARRAY_UTIL_H
