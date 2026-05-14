#include "Options.h"
#include "util.h"

namespace molpro::linalg::itsolv {

void Options::copy(const Options& options) {
  convergence_threshold = options.convergence_threshold;
  n_roots = options.n_roots;
  verbosity = options.verbosity;
  max_iter = options.max_iter;
  max_p = options.max_p;
  p_threshold = options.p_threshold;
}

Options::Options(const options_map& opt) {
  auto facet = util::StringFacet{};
  auto opt_upper = util::capitalize_keys(opt, facet);
  if (auto key = facet.toupper("n_roots"); opt_upper.count(key)) {
    n_roots = std::stoi(opt_upper.at(key));
  };
  if (auto key = facet.toupper("convergence_threshold"); opt_upper.count(key)) {
    convergence_threshold = std::stod(opt_upper.at(key));
  }
  if (auto key = facet.toupper("verbosity"); opt_upper.count(key)) {
    verbosity = static_cast<Verbosity>(std::stoi(opt_upper.at(key)));
  }
  if (auto key = facet.toupper("max_iter"); opt_upper.count(key)) {
    max_iter = std::stoi(opt_upper.at(key));
  }
  if (auto key = facet.toupper("max_p"); opt_upper.count(key)) {
    max_p = std::stoi(opt_upper.at(key));
  }
  if (auto key = facet.toupper("p_threshold"); opt_upper.count(key)) {
    p_threshold = std::stod(opt_upper.at(key));
  }
}

} // namespace molpro::linalg::itsolv