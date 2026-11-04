#include "vector_types.h"
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <molpro/linalg/itsolv/IterativeSolverTemplate.h>
#include <molpro/linalg/itsolv/Logger.h>
#include <molpro/linalg/itsolv/SolverFactory.h>
#include <molpro/linalg/itsolv/subspace/Matrix.h>
#include <molpro/mpi.h>

#include <complex>
#include <string>
#include <vector>

using molpro::linalg::itsolv::CVecRef;
using molpro::linalg::itsolv::Logger;
using molpro::linalg::itsolv::VecRef;
namespace itlog = molpro::linalg::itsolv::log;
using ::testing::ElementsAre;

namespace {

struct TestEvent {
  static constexpr std::string_view name = "TestEvent";
  struct Payload {
    int iter;
    std::vector<double> values;
  };
  static std::string format(const Payload& p) { return std::to_string(p.iter); }
};

template <typename Scalar>
struct ScalarEvent {
  static constexpr std::string_view name = "ScalarEvent";
  struct Payload {
    Scalar value;
  };
};

//! Records the text output of the built-in handler
class TextRecorder : public Logger {
public:
  using Logger::Logger;
  mutable std::vector<std::string> texts;
  mutable std::vector<std::string> contexts;

protected:
  void default_message_handler(std::string_view ctx, itlog::Severity, itlog::Verbosity,
                               std::string_view msg) const override {
    contexts.emplace_back(ctx);
    texts.emplace_back(msg);
  }
};

} // namespace

TEST(Logger, typed_sink_receives_raw_payload_regardless_of_thresholds) {
  Logger logger(itlog::Severity::Warning, itlog::Verbosity::None);
  logger.set_default_output(false);
  std::vector<int> iters;
  std::vector<double> values;
  logger.on<TestEvent>([&](const itlog::Record& record, const TestEvent::Payload& p) {
    EXPECT_EQ(record.context, "TestEvent");
    EXPECT_EQ(record.message, "msg ");
    iters.push_back(p.iter);
    values = p.values;
  });
  logger.emit<TestEvent>(itlog::Severity::Normal, itlog::Verbosity::Trace, "msg ",
                         [] { return TestEvent::Payload{3, {1.5, 2.5}}; });
  EXPECT_THAT(iters, ElementsAre(3));
  EXPECT_THAT(values, ElementsAre(1.5, 2.5));
}

TEST(Logger, typed_sink_only_receives_its_context) {
  Logger logger;
  logger.set_default_output(false);
  int n_double = 0, n_float = 0;
  logger.on<ScalarEvent<double>>([&](const itlog::Record&, const ScalarEvent<double>::Payload&) { ++n_double; });
  logger.on<ScalarEvent<float>>([&](const itlog::Record&, const ScalarEvent<float>::Payload&) { ++n_float; });
  logger.emit<ScalarEvent<double>>(itlog::Severity::Normal, itlog::Verbosity::Info, "",
                                   [] { return ScalarEvent<double>::Payload{1.0}; });
  logger.info("free text");
  EXPECT_EQ(n_double, 1);
  EXPECT_EQ(n_float, 0);
}

TEST(Logger, untyped_sink_inherits_thresholds) {
  Logger logger;
  logger.set_default_output(false);
  std::vector<std::string> texts;
  logger.add_sink([&](const itlog::Record& record) { texts.push_back(record.text()); });
  logger.trace("trace");
  logger.info("info");
  logger.set_verbosity(itlog::Verbosity::Trace);
  logger.trace("trace again");
  logger.set_min_severity(itlog::Severity::Warning);
  logger.info("dropped");
  logger.warn("warning");
  EXPECT_THAT(texts, ElementsAre("info", "trace again", "warning"));
}

TEST(Logger, untyped_sink_with_own_filter) {
  Logger logger;
  logger.set_default_output(false);
  int n = 0;
  logger.add_sink([&](const itlog::Record&) { ++n; }, {.verbosity = itlog::Verbosity::Trace});
  logger.trace("trace");
  EXPECT_EQ(n, 1);
}

TEST(Logger, data_dumps) {
  Logger logger(itlog::Severity::Normal, itlog::Verbosity::Trace);
  logger.set_default_output(false);
  std::vector<std::string> untyped;
  std::vector<std::vector<double>> typed;
  logger.add_sink([&](const itlog::Record& record) { untyped.push_back(record.text()); });
  logger.on<itlog::DataDump<std::vector<double>>>(
      [&](const itlog::Record& record, const itlog::DataDump<std::vector<double>>::Payload& p) {
        EXPECT_TRUE(record.is_data_dump());
        typed.push_back(p.data);
      });
  const std::vector<double> data{1.23456, 2.0};
  logger.data_dump<3>("x = ", data);
  EXPECT_TRUE(untyped.empty());
  EXPECT_EQ(typed.size(), 1);
  logger.enable_data_dumps(true);
  logger.data_dump<3>("x = ", data);
  EXPECT_THAT(untyped, ElementsAre("x = [1.23, 2]"));
  EXPECT_EQ(typed.size(), 2);
  EXPECT_EQ(typed.back(), data);
}

TEST(Logger, payload_only_created_when_wanted) {
  Logger logger;
  bool made = false;
  auto make = [&] {
    made = true;
    return TestEvent::Payload{0, {}};
  };
  logger.emit<TestEvent>(itlog::Severity::Normal, itlog::Verbosity::Debug, "", make);
  EXPECT_FALSE(made) << "default output does not want Debug";
  logger.set_default_output(false);
  logger.set_verbosity(itlog::Verbosity::Trace);
  logger.emit<TestEvent>(itlog::Severity::Normal, itlog::Verbosity::Debug, "", make);
  EXPECT_FALSE(made) << "nobody listens";
  logger.on<TestEvent>([](const itlog::Record&, const TestEvent::Payload&) {});
  logger.emit<TestEvent>(itlog::Severity::Normal, itlog::Verbosity::Debug, "", make);
  EXPECT_TRUE(made);
}

TEST(Logger, record_access) {
  Logger logger;
  logger.set_default_output(false);
  int n = 0;
  logger.add_sink([&](const itlog::Record& record) {
    ++n;
    EXPECT_TRUE(record.is<TestEvent>());
    EXPECT_FALSE(record.is<itlog::Generic>());
    EXPECT_EQ(record.get<itlog::Generic>(), nullptr);
    ASSERT_NE(record.get<TestEvent>(), nullptr);
    EXPECT_EQ(record.get<TestEvent>()->iter, 7);
    EXPECT_EQ(record.text(), "Iteration  ->  7");
  });
  logger.emit<TestEvent>(itlog::Severity::Normal, itlog::Verbosity::Info, "Iteration",
                         [] { return TestEvent::Payload{7, {}}; });
  EXPECT_EQ(n, 1);
}

TEST(Logger, remove_sink) {
  Logger logger;
  logger.set_default_output(false);
  int n = 0;
  auto id = logger.add_sink([&](const itlog::Record&) { ++n; });
  logger.info("a");
  logger.remove_sink(id);
  logger.info("b");
  EXPECT_EQ(n, 1);
}

TEST(Logger, empty_argument_keeps_separator) {
  TextRecorder logger;
  logger.info("x ", std::string(""), 5);
  EXPECT_THAT(logger.texts, ElementsAre("x , 5"));
}

TEST(Logger, default_message_handler_override) {
  TextRecorder logger;
  logger.info("value ", 1, 2.5);
  logger.info("no blank", std::vector<int>{1, 2});
  logger.emit<TestEvent>(itlog::Severity::Normal, itlog::Verbosity::Info, "event ",
                         [] { return TestEvent::Payload{4, {}}; });
  EXPECT_THAT(logger.texts, ElementsAre("value 1, 2.5", "no blank  ->  [1, 2]", "event 4"));
  EXPECT_THAT(logger.contexts, ElementsAre("Generic", "Generic", "TestEvent"));
  logger.set_default_output(false);
  logger.info("silenced");
  EXPECT_EQ(logger.texts.size(), 3);
}

TEST(Logger, stringify) {
  EXPECT_EQ(itlog::stringify(1.23456, 3), "1.23");
  EXPECT_EQ(itlog::stringify(42, 3), "42");
  EXPECT_EQ(itlog::stringify(std::complex<double>(1.23456, 2), 3), "(1.23,2)");
  EXPECT_EQ(itlog::stringify(std::vector<std::pair<std::size_t, double>>{{1, 0.5}}), "[{ 1: 0.5 }]");
  EXPECT_EQ(itlog::stringify(std::string("text"), 3), "text");
  static_assert(itlog::stringifiable<std::vector<std::pair<int, double>>>);
  static_assert(!itlog::stringifiable<std::vector<std::vector<itlog::Generic::Payload>>>);
  molpro::linalg::itsolv::subspace::Matrix<double> m({1, 2});
  m(0, 0) = 1.23456;
  m(0, 1) = 2;
  EXPECT_EQ(itlog::stringify(m, 3), "\n[[1.23, 2]]");
}

namespace {

//! Symmetric matrix whose first unit vector is an exact eigenvector, so that root 0 converges immediately
class DecoupledProblem : public molpro::linalg::itsolv::Problem<Rvector> {
public:
  const size_t n;
  mutable std::vector<size_t> action_sizes; //!< number of vectors passed to each call of action()
  explicit DecoupledProblem(size_t n) : n(n) {}

  double matrix(size_t i, size_t j) const {
    if (i == j)
      return double(i + 1);
    return (i == 0 || j == 0) ? 0.0 : 0.01 * double(i + j);
  }

  bool diagonals(container_t& d) const override {
    for (size_t i = 0; i < d.size(); i++)
      d[i] = matrix(i, i);
    return true;
  }

  void action(const CVecRef<Rvector>& parameters, const VecRef<Rvector>& actions) const override {
    action_sizes.push_back(parameters.size());
    for (size_t k = 0; k < parameters.size(); k++) {
      const auto& v = parameters[k].get();
      auto& a = actions[k].get();
      for (size_t i = 0; i < a.size(); i++) {
        a[i] = 0;
        for (size_t j = 0; j < a.size(); j++)
          a[i] += matrix(i, j) * v[j];
      }
    }
  }
};

} // namespace

//! A logger set after construction reaches all parts of the solver, including the Q space
TEST(Logger, set_logger_propagates) {
  molpro::mpi::init();
  auto solver = molpro::linalg::itsolv::create_LinearEigensystem<Rvector, Qvector>("Davidson");
  solver->set_n_roots(2);
  auto logger = std::make_shared<Logger>(itlog::Severity::Normal, itlog::Verbosity::Trace, true);
  logger->set_default_output(false);
  std::vector<std::string> messages;
  logger->add_sink([&](const itlog::Record& record) { messages.emplace_back(record.message); });
  solver->set_logger(logger);
  DecoupledProblem problem(10);
  std::vector<Rvector> c(2, Rvector(problem.n)), g(2, Rvector(problem.n));
  EXPECT_TRUE(solver->solve(c, g, problem, true));
  EXPECT_THAT(messages, ::testing::Contains("QSpace::update"));
}

//! The solver reports its progress through the lifecycle events
TEST(Logger, solver_lifecycle_events) {
  molpro::mpi::init();
  constexpr size_t nroots = 3;
  auto solver = molpro::linalg::itsolv::create_LinearEigensystem<Rvector, Qvector>("Davidson");
  solver->set_n_roots(nroots);
  solver->set_convergence_threshold(1e-8);
  auto logger = std::make_shared<Logger>();
  logger->set_default_output(false);
  int n_started = 0, n_finished = 0;
  std::vector<std::vector<int>> new_iteration_roots, completed_roots;
  std::vector<std::size_t> n_values;
  bool finished_converged = false;
  logger->on<itlog::SolveStarted<double>>([&](const itlog::Record&, const itlog::SolveStarted<double>::Payload& p) {
    ++n_started;
    EXPECT_EQ(p.n_roots, nroots);
    EXPECT_DOUBLE_EQ(p.convergence_threshold, 1e-8);
    EXPECT_TRUE(p.initial_guess_generated);
  });
  logger->on<itlog::NewIteration<double>>([&](const itlog::Record&, const itlog::NewIteration<double>::Payload& p) {
    EXPECT_EQ(p.iter, int(new_iteration_roots.size()));
    new_iteration_roots.push_back(p.roots);
  });
  logger->on<itlog::IterationCompleted<double>>(
      [&](const itlog::Record&, const itlog::IterationCompleted<double>::Payload& p) {
        completed_roots.push_back(p.roots);
        EXPECT_EQ(p.errors.size(), nroots);
        n_values.push_back(p.values.size());
      });
  logger->on<itlog::SolveFinished<double>>([&](const itlog::Record&, const itlog::SolveFinished<double>::Payload& p) {
    ++n_finished;
    finished_converged = p.converged;
    EXPECT_EQ(p.iterations, int(completed_roots.size()));
    ASSERT_EQ(p.values.size(), nroots);
    EXPECT_DOUBLE_EQ(p.values[0], 1.0);
  });
  int n_float = 0;
  logger->on<itlog::IterationCompleted<float>>(
      [&](const itlog::Record&, const itlog::IterationCompleted<float>::Payload&) { ++n_float; });
  solver->set_logger(logger);

  DecoupledProblem problem(12);
  std::vector<Rvector> c(nroots, Rvector(problem.n)), g(nroots, Rvector(problem.n));
  const bool converged = solver->solve(c, g, problem, true);
  EXPECT_TRUE(converged);
  EXPECT_EQ(n_started, 1);
  EXPECT_EQ(n_finished, 1);
  EXPECT_EQ(finished_converged, converged);
  EXPECT_EQ(n_float, 0);
  ASSERT_GE(new_iteration_roots.size(), 2);
  EXPECT_EQ(new_iteration_roots, completed_roots);
  EXPECT_THAT(n_values, ::testing::Each(nroots));
  // the roots worked on are exactly the vectors passed to action()
  std::vector<size_t> sizes;
  for (const auto& roots : new_iteration_roots)
    if (!roots.empty())
      sizes.push_back(roots.size());
  EXPECT_EQ(sizes, problem.action_sizes);
  // root 0 is exact after the first iteration, so it is no longer worked on afterwards
  EXPECT_THAT(new_iteration_roots.front(), ElementsAre(0, 1, 2));
  EXPECT_LT(new_iteration_roots[1].size(), nroots);
  EXPECT_THAT(new_iteration_roots[1], ::testing::Not(::testing::Contains(0)));
}

//! At the default verbosity, the text output shows the completed iterations but not the Debug lifecycle events
TEST(Logger, iteration_completed_in_default_text_output) {
  molpro::mpi::init();
  auto solver = molpro::linalg::itsolv::create_LinearEigensystem<Rvector, Qvector>("Davidson");
  solver->set_n_roots(2);
  auto logger = std::make_shared<TextRecorder>();
  solver->set_logger(logger);
  DecoupledProblem problem(10);
  std::vector<Rvector> c(2, Rvector(problem.n)), g(2, Rvector(problem.n));
  EXPECT_TRUE(solver->solve(c, g, problem, true));
  EXPECT_THAT(logger->contexts, ::testing::Contains("IterationCompleted"));
  EXPECT_THAT(logger->contexts, ::testing::Not(::testing::Contains("NewIteration")));
  EXPECT_THAT(logger->contexts, ::testing::Not(::testing::Contains("SolveStarted")));
}
