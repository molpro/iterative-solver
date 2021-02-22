#ifndef LINEARALGEBRA_SRC_MOLPRO_LINALG_ARRAY_DISTRARRAYSPAN_H
#define LINEARALGEBRA_SRC_MOLPRO_LINALG_ARRAY_DISTRARRAYSPAN_H
#include <molpro/linalg/array/DistrArray.h>
#include <molpro/mpi.h>
using molpro::mpi::comm_global;

namespace molpro::linalg::array {
template <typename T = double>
class DistrArraySpan : public DistrArray<T> {
protected:
  std::unique_ptr<util::Distribution<size_t>> m_distribution;     //!< describes distribution of array among processes
  bool m_allocated = false;                         //!< whether the window has been created
public:
  using value_type = T;
  using size_type = typename DistrArray<T>::size_type;
  using index_type = typename DistrArray<T>::index_type;
protected:
  using LocalBuffer = typename DistrArray<T>::LocalBuffer;
  using Distribution = util::Distribution<size_type>;
  using DistrArray<T>::error;
  Span<value_type> m_span;                          //!< Span over provided buffer
  
public:
  DistrArraySpan();
  DistrArraySpan(size_t dimension, MPI_Comm commun = comm_global());
  DistrArraySpan(std::unique_ptr<Distribution> distribution, MPI_Comm commun = comm_global());
  DistrArraySpan(const DistrArraySpan &source);
  explicit DistrArraySpan(const DistrArray<T> &source);
  DistrArraySpan(DistrArraySpan &&source) noexcept;
  DistrArraySpan &operator=(const DistrArraySpan &source);
  DistrArraySpan &operator=(DistrArraySpan &&source) noexcept;
  ~DistrArraySpan() override;

  template <class U>
  friend void swap(DistrArraySpan<U> &a1, DistrArraySpan<U> &a2) noexcept;
  //void sync() const override;
  void allocate_buffer() override;
  void allocate_buffer(Span<value_type> buffer);
  void free_buffer() override;
  [[nodiscard]] bool empty() const override;
  
protected:
  struct LocalBufferSpan : public DistrArray<T>::LocalBuffer {
    explicit LocalBufferSpan(DistrArraySpan &source);
    explicit LocalBufferSpan(const DistrArraySpan &source);
  };

public:
  [[nodiscard]] const Distribution &distribution() const override;
  [[nodiscard]] std::unique_ptr<LocalBuffer> local_buffer() override;
  [[nodiscard]] std::unique_ptr<const LocalBuffer> local_buffer() const override;
  [[nodiscard]] value_type at(index_type ind) const override;
  void set(index_type ind, value_type val) override;
  void get(index_type lo, index_type hi, value_type *buf) const override;
  [[nodiscard]] std::vector<value_type> get(index_type lo, index_type hi) const override;
  void put(index_type lo, index_type hi, const value_type *data) override;
  void acc(index_type lo, index_type hi, const value_type *data) override;
  [[nodiscard]] std::vector<value_type> gather(const std::vector<index_type> &indices) const override;
  void scatter(const std::vector<index_type> &indices, const std::vector<value_type> &data) override;
  void scatter_acc(std::vector<index_type> &indices, const std::vector<value_type> &data) override;
  [[nodiscard]] std::vector<value_type> vec() const override;
};

} // molpro::linalg::array
#endif // LINEARALGEBRA_SRC_MOLPRO_LINALG_ARRAY_DISTRARRAYSPAN_H
