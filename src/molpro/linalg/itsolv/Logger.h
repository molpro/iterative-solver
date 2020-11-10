#ifndef LINEARALGEBRA_SRC_MOLPRO_LINALG_ITSOLV_LOGGER_H
#define LINEARALGEBRA_SRC_MOLPRO_LINALG_ITSOLV_LOGGER_H
#include <bitset>
#include <numeric>
#include <string>

namespace molpro::linalg::itsolv {

/*!
 * @brief A dummy structured logger.
 *
 * This should be replaced by a fully functional logger from Profiler or similar.
 *
 * The log output is structured using JSON or HTML format for easy parsing.
 * Logging is split into sections
 * @code
 * //Log output
 * { // some initial state attributes
 * Log = [
 * {SectionName = "Section1",
 * Log = [
 * {SectionName = "Section1Child1"},
 * Log = [{ date = "year-mm-dd-hh:mm:ss", msg="log message", custom_entry_name="DataDump"}
 * ,{ date = "...", msg="another message"}
 * ]
 * ]
 * }
 * ,{SectionName = "Section2"
 * }
 * ]
 * }
 * @endcode
 *
 * The logger also handles output which might go to a separate stream.
 */
struct Logger {
  /*!
   * @brief Different levels of logging
   *
   * {Trace, Debug, Info} are hierarchical
   * {Warn, Error} are hierarchical
   * Fatal is always on
   * DataDump is optional
   */
  enum Level : short { None, Trace, Debug, Info, Warn, Error, Fatal };

  void msg(const std::string& message, Level log_lvl);

  template <typename ForwardIt>
  void msg(const std::string& message, ForwardIt begin, ForwardIt end, Level log_lvl) {
    msg(std::accumulate(begin, end, message, [](auto s, auto el) { return s + std::to_string(el) + ", "; }), log_lvl);
  }

  //! Converts double to a string in scientific notation
  static std::string scientific(double val);

  Level max_trace_level = None; //! highest level of trace message that can be logged
  Level max_warn_level = None;  //! highest level of warning/error that can be logged
  bool data_dump = false;       //! whether data dumps are allowed
};

} // namespace molpro::linalg::itsolv

#endif // LINEARALGEBRA_SRC_MOLPRO_LINALG_ITSOLV_LOGGER_H
