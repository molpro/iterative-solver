#include "LinearEquationsDavidsonOptions.h"
#include "DavidsonOptions.h"
#include "Options.h"
#include "util.h"

namespace molpro::linalg::itsolv {

LinearEquationsDavidsonOptions::LinearEquationsDavidsonOptions(const options_map& opt)
    : LinearEquationsOptions(opt), DavidsonOptions(opt) {
  auto facet = util::StringFacet{};
  auto opt_upper = util::capitalize_keys(opt, facet);
  if (auto key = facet.toupper("augmented_hessian"); opt_upper.count(key)) {
    augmented_hessian = std::stod(opt_upper.at(key));
  }
}

} // namespace molpro::linalg::itsolv
