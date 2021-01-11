#ifndef LINEARALGEBRA_SRC_MOLPRO_LINALG_ARRAY_UTIL_H
#define LINEARALGEBRA_SRC_MOLPRO_LINALG_ARRAY_UTIL_H
#include <atomic>
#include <future>

namespace molpro::linalg::array::util {

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
