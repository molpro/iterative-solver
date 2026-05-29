#include "LinearEigensystemDavidsonOptions.h"
#include "DavidsonOptions.h"
#include "Options.h"

namespace molpro::linalg::itsolv {
LinearEigensystemDavidsonOptions::LinearEigensystemDavidsonOptions(const std::map<std::string, std::string>& opt)
    : LinearEigensystemOptions(opt), DavidsonOptions(opt) {}

} // namespace molpro::linalg::itsolv
