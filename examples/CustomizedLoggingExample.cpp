#include "ExampleProblem.h"
#include <molpro/linalg/itsolv/SolverFactory.h>
#include <molpro/linalg/itsolv/Logger.h>
#include <molpro/linalg/itsolv/IterativeSolverTemplate.h>

#include <iostream>
#include <string_view>
#include <ranges>
#include <algorithm>

using namespace molpro::linalg::itsolv;

// There are 3 different ways of how we can customize logging. Two of them involve providing specializations
// of the LogHandler templates, which is what we demonstrate here. This has the big advantage that we get direct
// access to the logged parameters in their original (non-stringified) form, which allows custom processing.
//
// The specialization always is done on the specific log context that we want to catch. Entry point is always
// the handle(...) function. This can be implemented as either a static function or a const member function.
// In the member case, it is expected that the logger class publicly inherits from the LogHandler specialization
// and is mainly intended for handing the call to a virtual function that is implemented in the actual logger.
namespace molpro::linalg::itsolv::log {

  // This is an example for using the member variant. For this to work we have to provide a custom logger
  // implementation that publicly inherits from this handler!
  template<>
  struct LogHandler<NewIteration> {
    virtual void new_iteration(NewIteration::arg_t<NewIteration::iter> iter, NewIteration::arg_t<NewIteration::errors> errors) const = 0;

    template<typename ...Ts>
    // This requirement only makes sense for log contexts using a fixed argument set
    requires(NewIteration::num_args == sizeof...(Ts))
    void handle(Severity severity, Verbosity verbosity, std::string_view msg, Ts &&...args) const {
      // Forward to dedicated member function that will be overwritten in the logger class
	  // This only works in this fashion because the NewIteration uses a fixed argument set
      new_iteration(NewIteration::get_arg<NewIteration::iter>(args...), NewIteration::get_arg<NewIteration::errors>(args...));
    }
  };
  // This ensures that the interface is implemented correctly.
  // This requires hardcoded knowledge about the parameters of NewIteration, which the above implementation
  // partially avoided by using all that get_arg and arg_t constructs.
  static_assert(member_handler_exists<NewIteration, int, std::vector<double>>);


  // This is the static approach. Note the changed function call signature which now takes a Logger instance
  // as its first parameter!
  template<>
  struct LogHandler<DataDump> {
    template<typename ...Ts>
    static void handle(const Logger &logger, Severity severity, Verbosity verbosity, std::string_view msg, Ts &&...data) {
      std::cout << "Dumping data '" << msg << "'" << ((std::string() + ... + (stringify<DataDump>(std::forward<Ts>(data)) + ", "))) << "\n";
	}
  };
  // Again, assertion to ensure implementation is done correctly. Since DataDump doesn't use a fixed argument
  // set, we there is no point in specifying specific arguments in the concept as the implementation has to
  // deal with arbitrary arguments.
  static_assert(static_handler_exists<DataDump>);

}

// Note the inheritance from LogHandler<log::NewIteration>
// This demonstrates the third way to customize logging: overwriting virtual functions in the Logger class
class MyLogger : public molpro::linalg::itsolv::Logger, public molpro::linalg::itsolv::log::LogHandler<log::NewIteration> {
public:
protected:
  // This is the function that gets called whenever there is no dedicated LogHandler specialization available.
  // However, it will only receive a single, fully stringified version of the original message along with all
  // parameters. Hence, it can't (easily) access the original data parameters.
  // If all you want to do is to redirect the log messages, this function is the way to go though.
  void default_message_handler(std::string_view ctx, log::Severity severity, log::Verbosity verbosity, std::string_view msg) const override {
    std::cout << "Log message received via default message handler (ctx: " << ctx << "): " << msg << "\n";
  }

  // This function only serves debugging purposes. It allows you to get an overview of what log contexts
  // are being used (with how many arguments). Usually, you will want to leave this at its default
  // implementation (which does nothing).
  void log_ctx(std::string_view tag, std::size_t num_args) const override {
    std::cout << "Received log for context '" << tag << "' with " << num_args << " arguments\n";
  }

  // This overwrites the customization point we created for ourselves in LogHandler<NewIteration>
  void new_iteration(log::NewIteration::arg_t<log::NewIteration::iter> iteration,
      log::NewIteration::arg_t<log::NewIteration::errors> errors) const override {
    using std::ranges::begin;
    using std::ranges::end;

    std::cout << "Iteration " << iteration;

    if (begin(errors) != end(errors)) {
      std::cout << " (max. error is " << std::ranges::max(errors, std::less<>{}, [](auto val) { return std::abs(val); }) << ")";
    }

    std::cout << "\n";
  }
};


int main(int argc, char* argv[]) {
  {
    auto problem = ExampleProblem(argc > 1 ? std::stoi(argv[1]) : 20);
    using Rvector = decltype(problem)::container_t;
    auto solver = create_LinearEigensystem<Rvector>("Davidson");

    solver->set_n_roots(argc > 2 ? std::stoi(argv[2]) : 2);
    solver->set_max_iter(100);

    std::vector<Rvector> c, g;
    auto nbuffer = argc > 3 ? std::stoi(argv[3]) : solver->n_roots();
    for (size_t root = 0; root < nbuffer; root++) {
      c.emplace_back(problem.n);
      g.emplace_back(problem.n);
    }

	// Install our custom logger
    std::shared_ptr<MyLogger> logger = std::make_shared<MyLogger>();
    solver->set_logger(logger);

    solver->solve(c, g, problem, true);

    std::vector<int> roots;
    std::iota(roots.begin(), roots.end(), 0);
    solver->solution(roots, c, g);
    for (const auto& ev : solver->eigenvalues())
      std::cout << "Final eigenvalue: " << ev << std::endl;
  }
}

#include <molpro/linalg/itsolv/SolverFactory-implementation.h>
#include <vector>
template class molpro::linalg::itsolv::SolverFactory<std::vector<double>, std::vector<double>,
                                                     std::map<size_t, double>>;
