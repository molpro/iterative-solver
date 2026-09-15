#ifndef LINEARALGEBRA_SRC_MOLPRO_LINALG_ITSOLV_LINEAREQUATIONSDAVIDSONOPTIONS_H
#define LINEARALGEBRA_SRC_MOLPRO_LINALG_ITSOLV_LINEAREQUATIONSDAVIDSONOPTIONS_H
#include <molpro/linalg/itsolv/DavidsonOptions.h>
#include <molpro/linalg/itsolv/Options.h>
#include <molpro/linalg/itsolv/options_map.h>

#include <optional>

namespace molpro::linalg::itsolv {
/*!
 * @brief Allows setting and getting of options for LinearEquationsDavidson instance via IterativeSolver base class
 */
struct LinearEquationsDavidsonOptions : public LinearEquationsOptions, public DavidsonOptions {
  LinearEquationsDavidsonOptions() = default;
  LinearEquationsDavidsonOptions(const options_map& opt);

  std::optional<double> augmented_hessian;
};

} // namespace molpro::linalg::itsolv

#endif // LINEARALGEBRA_SRC_MOLPRO_LINALG_ITSOLV_LINEAREQUATIONSDAVIDSONOPTIONS_H
