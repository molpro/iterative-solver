#include "ExampleProblem.h"

#include <molpro/linalg/itsolv/IterativeSolverTemplate.h>
#include <molpro/linalg/itsolv/Logger.h>
#include <molpro/linalg/itsolv/SolverFactory.h>
#include <molpro/linalg/itsolv/subspace/Matrix.h>
#include <molpro/mpi.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <format>
#include <iostream>
#include <numeric>
#include <string_view>

using namespace molpro::linalg::itsolv;

// Everything the solvers log is an event of some context (log::NewIteration<Scalar>, log::DataDump<T>, ...).
// There are three ways of observing events:
//
//  1. Typed subscriptions, Logger::on<Ctx>(f): f receives the raw payload of every event of context Ctx, independent
//     of the logger's verbosity and data-dump settings. Contexts carrying floating-point data are templated on the
//     scalar type of the solver, so subscribe for the type your solver uses.
//  2. Untyped sinks, Logger::add_sink(f, filter): f receives every event passing the filter (by default the logger's
//     own thresholds) as a log::Record, which gives the formatted text and, via get<Ctx>(), the payload.
//  3. Overriding Logger::default_message_handler(), which receives the formatted text of everything the built-in
//     text output would print.
//
// Sinks are registered on a logger instance at runtime, so they also work for solvers compiled elsewhere.

// Approach 3: redirect the built-in text output
class MyLogger : public Logger {
protected:
  void default_message_handler(std::string_view ctx, log::Severity severity, log::Verbosity verbosity,
                               std::string_view msg) const override {
    std::cout << "[text output, context " << ctx << "] " << msg << "\n";
  }
};

int main(int argc, char* argv[]) {
  molpro::mpi::init();
  {
    auto problem = ExampleProblem(argc > 1 ? std::stoi(argv[1]) : 20);
    using Rvector = decltype(problem)::container_t;
    using scalar = double; // the solver's scalar_type
    auto solver = create_LinearEigensystem<Rvector>("Davidson");

    solver->set_n_roots(argc > 2 ? std::stoi(argv[2]) : 2);
    solver->set_max_iter(100);

    std::vector<Rvector> c, g;
    auto nbuffer = argc > 3 ? std::stoi(argv[3]) : solver->n_roots();
    for (size_t root = 0; root < nbuffer; root++) {
      c.emplace_back(problem.n);
      g.emplace_back(problem.n);
    }

    auto logger = std::make_shared<MyLogger>();
    // Only warnings make it to the text output
    logger->set_min_severity(log::Severity::Warning);

    // Approach 1: an iteration table built from the raw data of the lifecycle events
    auto start = std::chrono::steady_clock::now();
    logger->on<log::SolveStarted<scalar>>([&](const log::Record&, const log::SolveStarted<scalar>::Payload& p) {
      std::cout << "Solving for " << p.n_roots << " roots, threshold " << p.convergence_threshold << "\n"
                << " iter  max. error  vectors  time/s  eigenvalues\n";
      start = std::chrono::steady_clock::now();
    });
    logger->on<log::IterationCompleted<scalar>>(
        [&](const log::Record&, const log::IterationCompleted<scalar>::Payload& p) {
          const auto now = std::chrono::steady_clock::now();
          const double seconds = std::chrono::duration<double>(now - start).count();
          start = now;
          const auto max_error = p.errors.empty() ? 0.0 : *std::ranges::max_element(p.errors);
          // p.roots are the roots worked on in this iteration, which excludes the converged ones
          std::cout << std::format("{:5d}{:12.2e}{:9d}{:8.2f} ", p.iter, max_error, p.roots.size(), seconds);
          for (std::size_t i = 0; i < p.values.size(); ++i)
            std::cout << std::format(" {:.8f} ({:.0e})", p.values[i], p.errors[i]);
          std::cout << "\n";
        });
    logger->on<log::SolveFinished<scalar>>([](const log::Record&, const log::SolveFinished<scalar>::Payload& p) {
      std::cout << (p.converged ? "Converged" : "Not converged") << " after " << p.iterations << " iterations\n";
    });

    // Approach 1 for data dumps: the subspace matrices, without stringification (and independent of
    // enable_data_dumps(), which only steers the text output and untyped sinks)
    logger->on<log::DataDump<subspace::Matrix<scalar>>>(
        [](const log::Record& record, const log::DataDump<subspace::Matrix<scalar>>::Payload& p) {
          static std::size_t dimension = 0;
          if (record.message == "H = " && p.data.rows() != dimension)
            std::cout << "  subspace dimension " << (dimension = p.data.rows()) << "\n";
        });

    // Approach 2: count all debug messages, whatever their context
    int n_debug = 0;
    logger->add_sink(
        [&](const log::Record& record) {
          if (record.verbosity == log::Verbosity::Debug)
            ++n_debug;
        },
        {.min_severity = log::Severity::Normal, .verbosity = log::Verbosity::Debug});

    solver->set_logger(logger);
    solver->solve(c, g, problem, true);
    std::cout << n_debug << " debug messages were logged\n";

    std::vector<int> roots(solver->n_roots());
    std::iota(roots.begin(), roots.end(), 0);
    solver->solution(roots, c, g);
    for (const auto& ev : solver->eigenvalues())
      std::cout << "Final eigenvalue: " << ev << std::endl;
  }
  molpro::mpi::finalize();
}

#include <molpro/linalg/itsolv/SolverFactory-implementation.h>
#include <vector>
template class molpro::linalg::itsolv::SolverFactory<std::vector<double>, std::vector<double>,
                                                     std::map<size_t, double>>;
