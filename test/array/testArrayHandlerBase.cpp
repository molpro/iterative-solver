#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <molpro/linalg/array/ArrayHandler.h>

#include <memory>
#include <numeric>
#include <stdexcept>
#include <vector>

// Tests of the lazy-evaluation machinery provided by the ArrayHandler base class, using a minimal handler that relies
// on the default fused_dot and fused_axpy and counts the operations it is asked to do.

using molpro::linalg::array::ArrayHandler;
using molpro::linalg::array::CVecRef;
using molpro::linalg::array::Matrix;
using molpro::linalg::array::VecRef;
using molpro::linalg::array::util::ArrayHandlerError;
using ::testing::ElementsAre;

namespace {
using V = std::vector<double>;

class CountingHandler : public ArrayHandler<V, V> {
public:
  int n_dot = 0;
  int n_axpy = 0;

  V copy(const V& source) override { return source; }
  void copy(V& x, const V& y) override { x = y; }
  void scal(double alpha, V& x) override {
    for (auto& e : x)
      e *= alpha;
  }
  void fill(double alpha, V& x) override { std::fill(x.begin(), x.end(), alpha); }
  void axpy(double alpha, const V& x, V& y) override {
    ++n_axpy;
    for (size_t i = 0; i < y.size(); ++i)
      y[i] += alpha * x[i];
  }
  double dot(const V& x, const V& y) override {
    ++n_dot;
    return std::inner_product(x.begin(), x.end(), y.begin(), 0.);
  }
  void gemm_outer(const Matrix<double>, const CVecRef<V>&, const VecRef<V>&) override {
    throw std::logic_error("not used");
  }
  Matrix<double> gemm_inner(const CVecRef<V>&, const CVecRef<V>&) override { throw std::logic_error("not used"); }
  std::map<size_t, double> select_max_dot(size_t, const V&, const V&) override { return {}; }
  std::map<size_t, double> select(size_t, const V&, bool, bool) override { return {}; }

  using ArrayHandler<V, V>::lazy_handle;
  ProxyHandle lazy_handle() override { return this->lazy_handle(*this); }

  //! number of slots in the registry of lazy handles, and how many of them refer to live handles
  size_t registered_slots() const { return m_lazy_handles.size(); }
  size_t live_handles() const {
    return std::count_if(m_lazy_handles.begin(), m_lazy_handles.end(), [](const auto& h) { return !h.expired(); });
  }
};
} // namespace

// The default fused_dot and fused_axpy forward each registered operation to dot and axpy
TEST(ArrayHandlerBase, default_fused_operations_forward_to_dot_and_axpy) {
  CountingHandler handler;
  V x{1, 2, 3}, y{4, 5, 6}, z{1, 1, 1};
  double xy = 0, xz = 0, yz = 0;
  {
    auto h = handler.lazy_handle();
    h.dot(x, y, xy);
    h.dot(x, z, xz);
    h.dot(y, z, yz);
    EXPECT_EQ(handler.n_dot, 0);
  }
  EXPECT_EQ(handler.n_dot, 3);
  EXPECT_EQ(xy, 32);
  EXPECT_EQ(xz, 6);
  EXPECT_EQ(yz, 15);
  {
    auto h = handler.lazy_handle();
    h.axpy(2, x, z);
    h.axpy(-1, y, x);
    EXPECT_EQ(handler.n_axpy, 0);
  }
  EXPECT_EQ(handler.n_axpy, 2);
  EXPECT_THAT(z, ElementsAre(3, 5, 7));
  EXPECT_THAT(x, ElementsAre(-3, -3, -3));
}

namespace {
//! Records the arguments that the lazy handle passes to fused_dot
class RecordingHandler : public CountingHandler {
public:
  size_t n_ops = 0, n_unique_x = 0, n_unique_y = 0;

protected:
  void fused_dot(const std::vector<std::tuple<size_t, size_t, size_t>>& reg,
                 const std::vector<std::reference_wrapper<const V>>& xx,
                 const std::vector<std::reference_wrapper<const V>>& yy,
                 std::vector<std::reference_wrapper<double>>& out) override {
    n_ops = reg.size();
    n_unique_x = xx.size();
    n_unique_y = yy.size();
    CountingHandler::fused_dot(reg, xx, yy, out);
  }
};
} // namespace

// Before fused_dot is called, repeated references to the same array are merged so that each array is passed once,
// while every registered operation is kept
TEST(ArrayHandlerBase, arrays_passed_once_to_fused_operation) {
  RecordingHandler handler;
  V x{1, 2}, y1{1, 0}, y2{0, 1}, y3{1, 1};
  std::vector<double> out(3);
  {
    auto h = handler.lazy_handle();
    h.dot(x, y1, out[0]);
    h.dot(x, y2, out[1]);
    h.dot(x, y3, out[2]);
  }
  EXPECT_EQ(handler.n_ops, 3);
  EXPECT_EQ(handler.n_unique_x, 1);
  EXPECT_EQ(handler.n_unique_y, 3);
  EXPECT_THAT(out, ElementsAre(1, 2, 3));
}

// lazy_handle() records a weak_ptr, which expires with the handle, and expired slots are reused
TEST(ArrayHandlerBase, lazy_handle_registers_weak_ptr) {
  CountingHandler handler;
  EXPECT_EQ(handler.registered_slots(), 0);
  {
    auto h = handler.lazy_handle();
    EXPECT_EQ(handler.live_handles(), 1);
    {
      auto h2 = handler.lazy_handle();
      EXPECT_EQ(handler.live_handles(), 2);
    }
    EXPECT_EQ(handler.live_handles(), 1);
  }
  EXPECT_EQ(handler.live_handles(), 0);
  EXPECT_EQ(handler.registered_slots(), 2);
  auto h3 = handler.lazy_handle();
  EXPECT_EQ(handler.live_handles(), 1);
  EXPECT_EQ(handler.registered_slots(), 2) << "an expired slot should have been reused";
}

// With lazy evaluation off, operations are done immediately; turning it back on defers them again
TEST(ArrayHandlerBase, proxy_on_off) {
  CountingHandler handler;
  V x{1, 1}, y{0, 0};
  auto h = handler.lazy_handle();
  EXPECT_FALSE(h.is_off());
  h.off();
  EXPECT_TRUE(h.is_off());
  h.axpy(1, x, y);
  EXPECT_EQ(handler.n_axpy, 1);
  h.on();
  EXPECT_FALSE(h.is_off());
  h.axpy(1, x, y);
  EXPECT_EQ(handler.n_axpy, 1);
  h.eval();
  EXPECT_EQ(handler.n_axpy, 2);
  EXPECT_THAT(y, ElementsAre(2, 2));
}

// A lazy handle holds operations of only one type at a time
TEST(ArrayHandlerBase, mixing_dot_and_axpy_is_refused) {
  CountingHandler handler;
  V x{1, 2}, y{3, 4};
  double xy = 0;
  auto h = handler.lazy_handle();
  h.dot(x, y, xy);
  EXPECT_THROW(h.axpy(1, x, y), ArrayHandlerError);
  h.eval();
  EXPECT_EQ(xy, 11);
  EXPECT_THAT(y, ElementsAre(3, 4));
}

// eval clears the registry, so a repeated eval does nothing and another operation type is then allowed
TEST(ArrayHandlerBase, eval_clears_registry) {
  CountingHandler handler;
  V x{1, 2}, y{3, 4};
  double xy = 0;
  auto h = handler.lazy_handle();
  h.dot(x, y, xy);
  h.eval();
  EXPECT_EQ(handler.n_dot, 1);
  h.eval();
  EXPECT_EQ(handler.n_dot, 1);
  EXPECT_NO_THROW(h.axpy(1, x, y));
  h.eval();
  EXPECT_EQ(handler.n_axpy, 1);
  EXPECT_THAT(y, ElementsAre(4, 6));
}

// An invalidated handle does not evaluate its registered operations
TEST(ArrayHandlerBase, invalidated_handle_does_not_evaluate) {
  CountingHandler handler;
  V x{1, 2}, y{3, 4};
  double xy = -1;
  {
    auto h = handler.lazy_handle();
    h.dot(x, y, xy);
    EXPECT_FALSE(h.invalid());
    h.invalidate();
    EXPECT_TRUE(h.invalid());
    h.eval();
  }
  EXPECT_EQ(handler.n_dot, 0);
  EXPECT_EQ(xy, -1);
}

// Destroying the handler invalidates its live lazy handles, which must then not use it
TEST(ArrayHandlerBase, handler_destroyed_before_lazy_handle) {
  auto handler = std::make_unique<CountingHandler>();
  V x{1, 2}, y{3, 4};
  double xy = -1;
  auto h = handler->lazy_handle();
  h.dot(x, y, xy);
  EXPECT_FALSE(h.invalid());
  handler.reset();
  EXPECT_TRUE(h.invalid());
  h.eval(); // must return without touching the destroyed handler
  EXPECT_EQ(xy, -1);
}
