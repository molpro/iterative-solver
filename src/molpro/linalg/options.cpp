#include "options.h"
namespace {
std::shared_ptr<molpro::Options>& s_options() {
  static std::shared_ptr<molpro::Options> instance;
  return instance;
}
} // namespace

std::shared_ptr<const molpro::Options> molpro::linalg::options() {
  auto& opt = s_options();
  if (opt.get() == nullptr) {
    opt = std::make_shared<molpro::Options>("ITERATIVE-SOLVER", "");
  }
  return opt;
}

void molpro::linalg::set_options(const molpro::Options& options) {
  s_options() = std::make_shared<molpro::Options>(options);
}