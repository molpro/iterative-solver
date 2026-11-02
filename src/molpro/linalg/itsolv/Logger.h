#ifndef LINEARALGEBRA_SRC_MOLPRO_LINALG_ITSOLV_LOGGER_H
#define LINEARALGEBRA_SRC_MOLPRO_LINALG_ITSOLV_LOGGER_H
#include <cmath>
#include <complex>
#include <concepts>
#include <cstddef>
#include <format>
#include <functional>
#include <iomanip>
#include <limits>
#include <optional>
#include <ranges>
#include <sstream>
#include <string>
#include <string_view>
#include <type_traits>
#include <typeinfo>
#include <utility>
#include <vector>

#include <molpro/iostream.h>

/*!
 * @file
 * @brief Logging facilities of the iterative solvers.
 *
 * Everything that is logged is an *event*. An event belongs to a *context*, a tag type that names the
 * event and describes the data (the *payload*) that comes with it:
 *
 * @code
 * struct MyEvent {
 *   static constexpr std::string_view name = "MyEvent";
 *   struct Payload { int iter; std::vector<double> errors; };
 *   static std::string format(const Payload &p); // optional: text appended to the message
 * };
 * @endcode
 *
 * Free-text messages (Logger::info(), Logger::debug(), ...) use the log::Generic context, whose payload is empty
 * and whose message already contains the stringified arguments.
 *
 * Downstream code observes events through *sinks* registered on a Logger instance at runtime:
 *  - Logger::on<Ctx>(f) calls f(record, payload) for every event of context Ctx, giving access to the raw payload.
 *    Typed subscriptions are not subject to the logger's verbosity/severity thresholds or the data-dump flag.
 *  - Logger::add_sink(f, filter) calls f(record) for every event passing the filter; log::Record::text() yields the
 *    formatted message and log::Record::get<Ctx>() the payload, if the event is of context Ctx.
 *  - The built-in text output calls the virtual Logger::default_message_handler() with the formatted message. It
 *    obeys the logger's thresholds and the data-dump flag and can be switched off with
 *    Logger::set_default_output().
 *
 * Context identity is the C++ type: events of log::IterationCompleted<double> are not delivered to a subscription
 * for log::IterationCompleted<float>. Contexts carrying floating-point data are templated on the scalar type of
 * the solver, so they also work for single-precision and complex-valued problems.
 *
 * Payloads may refer to data owned by the emitter and are only valid during the call of the sink. Sinks must copy
 * whatever they want to keep. Sinks must not add or remove sinks of the logger that is calling them.
 */

namespace molpro::linalg::itsolv::log {

/*!
 * @brief Different levels of logging
 */
enum class Verbosity : short {
  Trace,
  Debug,
  Info,
  None,
};

/*!
 * @brief Severity of a given message
 */
enum class Severity : short {
  Normal,
  Warning,
  Error,
  Fatal,
};

/*!
 * Constant indicating that formatting should be done with default precision
 */
constexpr const std::size_t default_precision = std::numeric_limits<std::size_t>::max();

/*!
 * The real type associated with a (possibly complex) scalar type, e.g. the type of norms and errors
 */
template <typename Scalar>
using real_t = decltype(std::abs(std::declval<Scalar>()));

namespace detail {
template <typename T>
struct is_complex : std::false_type {};
template <typename T>
struct is_complex<std::complex<T>> : std::true_type {};
} // namespace detail

/*!
 * Concept checking that a type is a complex number type
 */
template <typename T>
concept complex_number = detail::is_complex<std::remove_cvref_t<T>>::value;

/*!
 * Concept checking that a given type is pair-like object
 */
template <typename T>
concept pair_like = requires(const std::remove_cvref_t<T> &t) {
  t.first;
  t.second;
};

/*!
 * Concept checking that a given type supports operator<< into an output stream
 */
template <typename T>
concept streamable = requires(std::ostream &stream, const std::remove_cvref_t<T> &t) { stream << t; };

/*!
 * Concept checking that a given type can be formatted via std::format
 */
template <typename T>
concept formattable = requires(const std::remove_cvref_t<T> &t, std::format_context ctx) {
  std::formatter<std::remove_cvref_t<T>>().format(t, ctx);
};

/*!
 * Concept checking that a type is string-like, i.e. it is printed as is
 */
template <typename T>
concept string_like = std::convertible_to<const std::remove_cvref_t<T> &, std::string_view>;

namespace detail {
template <typename T>
consteval bool is_stringifiable() {
  using U = std::remove_cvref_t<T>;
  if constexpr (string_like<U> || formattable<U> || streamable<U>) {
    return true;
  } else if constexpr (pair_like<U>) {
    return is_stringifiable<decltype(std::declval<const U &>().first)>() &&
           is_stringifiable<decltype(std::declval<const U &>().second)>();
  } else if constexpr (std::ranges::input_range<const U>) {
    using E = std::ranges::range_reference_t<const U>;
    if constexpr (std::same_as<std::remove_cvref_t<E>, U>)
      return false;
    else
      return is_stringifiable<E>();
  } else {
    return false;
  }
}
} // namespace detail

namespace detail {
//! Whether T is a range whose elements are of a different type than T (to exclude e.g. std::filesystem::path)
template <typename T>
consteval bool is_printable_range() {
  if constexpr (std::ranges::input_range<const T>)
    return !std::same_as<std::remove_cvref_t<std::ranges::range_reference_t<const T>>, T>;
  else
    return false;
}
} // namespace detail

/*!
 * Concept checking that a type can be converted into a string by stringify()
 */
template <typename T>
concept stringifiable = detail::is_stringifiable<T>();

/*!
 * @brief Converts the given object into a string
 *
 * Ranges are printed as [a, b, ...], pair-like objects as { first: second }. Anything else is formatted via
 * std::format or, failing that, operator<<.
 *
 * @param precision Number of significant digits for floating-point values (and complex numbers of them). Types
 * whose std::formatter understands a ".N" precision specification (e.g. subspace::Matrix) receive it as well.
 */
template <typename T>
  requires stringifiable<T>
std::string stringify(const T &t, std::size_t precision = default_precision) {
  using U = std::remove_cvref_t<T>;
  const bool precise = precision != default_precision;

  if constexpr (string_like<U>) {
    return std::string(std::string_view(t));
  } else if constexpr (complex_number<U> || (std::floating_point<U> && !formattable<U>)) {
    std::ostringstream stream;
    if (precise)
      stream << std::setprecision(static_cast<int>(precision));
    stream << t;
    return stream.str();
  } else if constexpr (std::floating_point<U>) {
    return precise ? std::format("{:.{}}", t, precision) : std::format("{}", t);
  } else if constexpr (pair_like<U>) {
    return "{ " + stringify(t.first, precision) + ": " + stringify(t.second, precision) + " }";
  } else if constexpr (detail::is_printable_range<U>()) {
    std::string result = "[";
    bool first = true;
    for (auto &&element : t) {
      if (!first)
        result += ", ";
      first = false;
      result += stringify(element, precision);
    }
    return result + "]";
  } else if constexpr (formattable<U>) {
    if (precise) {
      try {
        return std::vformat("{:." + std::to_string(precision) + "}", std::make_format_args(t));
      } catch (const std::format_error &) {
        // the type's formatter has no notion of precision
      }
    }
    return std::format("{}", t);
  } else {
    std::ostringstream stream;
    if (precise)
      stream << std::setprecision(static_cast<int>(precision));
    stream << t;
    return stream.str();
  }
}

/*!
 * Concept checking whether a given type is an event context: it has a name and a payload type
 */
template <typename Ctx>
concept event_context = requires {
  { Ctx::name } -> std::convertible_to<std::string_view>;
  typename Ctx::Payload;
};

/*!
 * Concept checking whether a context provides a text representation of its payload
 */
template <typename Ctx>
concept formattable_context = event_context<Ctx> && requires(const typename Ctx::Payload &p) {
  { Ctx::format(p) } -> std::convertible_to<std::string>;
};

/*!
 * Concept checking whether a context represents a data dump (which is subject to Logger::data_dumps_enabled())
 */
template <typename Ctx>
concept data_dump_context = event_context<Ctx> && requires {
  requires Ctx::is_data_dump;
};

/*!
 * Context for free-text messages. Its message already contains all stringified arguments.
 */
struct Generic {
  static constexpr std::string_view name = "Generic";
  struct Payload {};
};

/*!
 * Context used when dumping data of type T
 */
template <typename T>
struct DataDump {
  static constexpr std::string_view name = "DataDump";
  static constexpr bool is_data_dump = true;
  struct Payload {
    const T &data;         //!< The dumped data
    std::size_t precision; //!< Requested precision (default_precision if unspecified)
  };
  static std::string format(const Payload &p)
    requires stringifiable<T>
  {
    return stringify(p.data, p.precision);
  }
};

/*!
 * @returns message with the given suffix appended. A separator is inserted if the message doesn't end in a blank.
 */
inline std::string compose(std::string_view message, std::string_view suffix) {
  if (suffix.empty())
    return std::string(message);
  if (!message.empty() && message.back() != ' ')
    return std::string(message) + "  ->  " + std::string(suffix);
  return std::string(message) + std::string(suffix);
}

/*!
 * @brief A single logged event, as passed to sinks
 */
class Record {
public:
  std::string_view context; //!< Name of the context
  Severity severity;
  Verbosity verbosity;
  std::string_view message; //!< The message without the payload's text representation

  /*!
   * @returns the type of the event's context
   */
  const std::type_info &type() const { return *m_type; }

  /*!
   * @returns Whether this event is of context Ctx
   */
  template <event_context Ctx>
  bool is() const {
    return *m_type == typeid(Ctx);
  }

  /*!
   * @returns The event's payload if the event is of context Ctx, nullptr otherwise
   */
  template <event_context Ctx>
  const typename Ctx::Payload *get() const {
    return is<Ctx>() ? static_cast<const typename Ctx::Payload *>(m_payload) : nullptr;
  }

  /*!
   * @returns The message followed by the text representation of the payload (if its context provides one)
   */
  std::string text() const { return m_format ? compose(message, m_format(m_payload)) : std::string(message); }

  /*!
   * @returns Whether this event is a data dump
   */
  bool is_data_dump() const { return m_data_dump; }

  template <event_context Ctx>
  static Record make(Severity severity, Verbosity verbosity, std::string_view message,
                     const typename Ctx::Payload &payload) {
    Record record{Ctx::name, severity, verbosity, message};
    record.m_type = &typeid(Ctx);
    record.m_payload = &payload;
    record.m_data_dump = data_dump_context<Ctx>;
    if constexpr (formattable_context<Ctx>)
      record.m_format = [](const void *p) {
        return std::string(Ctx::format(*static_cast<const typename Ctx::Payload *>(p)));
      };
    return record;
  }

private:
  Record(std::string_view context, Severity severity, Verbosity verbosity, std::string_view message)
      : context(context), severity(severity), verbosity(verbosity), message(message) {}

  const std::type_info *m_type = nullptr;
  const void *m_payload = nullptr;
  std::string (*m_format)(const void *) = nullptr;
  bool m_data_dump = false;
};

/*!
 * @brief Selects which events an untyped sink receives. Unset fields inherit the logger's current settings.
 */
struct Filter {
  std::optional<Severity> min_severity; //!< Lowest severity that is passed on
  std::optional<Verbosity> verbosity;   //!< Lowest verbosity that is passed on
  std::optional<bool> data_dumps;       //!< Whether data dumps are passed on

  //! A filter that lets everything pass
  static Filter all() { return {Severity::Normal, Verbosity::Trace, true}; }
};

} // namespace molpro::linalg::itsolv::log

namespace molpro::linalg::itsolv {

/*!
 * @brief Logger class implementation. Provides (customizable) logging functionalities.
 *
 * See Logger.h for an overview of how to observe logged events.
 */
class Logger {
public:
  using Sink = std::function<void(const log::Record &)>;
  using SinkId = std::size_t;

  Logger(log::Severity min_severity = log::Severity::Normal, log::Verbosity verbosity = log::Verbosity::Info,
         bool enable_data_dumps = false)
      : m_verbosity(verbosity), m_min_severity(min_severity), m_dump_data(enable_data_dumps) {}
  virtual ~Logger() = default;

  /*!
   * @brief Registers a sink receiving all events that pass the given filter
   * @returns An id that can be passed to remove_sink()
   */
  SinkId add_sink(Sink sink, log::Filter filter = {}) {
    m_sinks.push_back({m_next_id, filter, nullptr, std::move(sink)});
    return m_next_id++;
  }

  /*!
   * @brief Registers f(const log::Record &, const Ctx::Payload &) to be called for every event of context Ctx
   *
   * The subscription is not subject to the logger's thresholds or its data-dump flag.
   * @returns An id that can be passed to remove_sink()
   */
  template <log::event_context Ctx, typename F>
    requires std::invocable<F &, const log::Record &, const typename Ctx::Payload &>
  SinkId on(F &&f) {
    m_sinks.push_back({m_next_id, log::Filter::all(), &typeid(Ctx),
                       [f = std::forward<F>(f)](const log::Record &record) mutable { f(record, *record.get<Ctx>()); }});
    return m_next_id++;
  }

  /*!
   * @brief Removes a previously registered sink
   */
  void remove_sink(SinkId id) {
    std::erase_if(m_sinks, [id](const SinkEntry &entry) { return entry.id == id; });
  }

  /*!
   * @brief Emits an event of context Ctx
   *
   * @param make Callable creating the payload. It is only invoked if some sink is interested in the event.
   */
  template <log::event_context Ctx, typename MakePayload>
    requires std::invocable<MakePayload &>
  void emit(log::Severity severity, log::Verbosity verbosity, std::string_view message, MakePayload &&make) const {
    if (!wants(typeid(Ctx), severity, verbosity, log::data_dump_context<Ctx>))
      return;
    const typename Ctx::Payload payload = make();
    dispatch(log::Record::make<Ctx>(severity, verbosity, message, payload));
  }

  /*!
   * @brief Emits an event of a context whose payload is default-constructible (usually empty)
   */
  template <log::event_context Ctx>
    requires std::default_initializable<typename Ctx::Payload>
  void emit(log::Severity severity, log::Verbosity verbosity, std::string_view message) const {
    emit<Ctx>(severity, verbosity, message, [] { return typename Ctx::Payload{}; });
  }

  /*!
   * @brief Logs a free-text message (context log::Generic). The arguments are stringified and appended.
   */
  template <typename... Ts>
  void msg(log::Severity severity, log::Verbosity verbosity, std::string_view message, const Ts &...args) const {
    static_assert((log::stringifiable<Ts> && ...), "All types passed to logger must be stringify-able");
    if (!wants(typeid(log::Generic), severity, verbosity, false))
      return;
    const log::Generic::Payload payload{};
    if constexpr (sizeof...(args) > 0) {
      std::string arg_str;
      ((arg_str += (arg_str.empty() ? "" : ", ") + log::stringify(args)), ...);
      const std::string text = log::compose(message, arg_str);
      dispatch(log::Record::make<log::Generic>(severity, verbosity, text, payload));
    } else {
      dispatch(log::Record::make<log::Generic>(severity, verbosity, message, payload));
    }
  }

  template <typename... Ts>
  void trace(std::string_view message, const Ts &...args) const {
    msg(log::Severity::Normal, log::Verbosity::Trace, message, args...);
  }

  template <typename... Ts>
  void debug(std::string_view message, const Ts &...args) const {
    msg(log::Severity::Normal, log::Verbosity::Debug, message, args...);
  }

  template <typename... Ts>
  void info(std::string_view message, const Ts &...args) const {
    msg(log::Severity::Normal, log::Verbosity::Info, message, args...);
  }

  template <typename... Ts>
  void warn(std::string_view message, const Ts &...args) const {
    msg(log::Severity::Warning, log::Verbosity::None, message, args...);
  }

  template <typename... Ts>
  void error(std::string_view message, const Ts &...args) const {
    msg(log::Severity::Error, log::Verbosity::None, message, args...);
  }

  template <typename... Ts>
  void fatal(std::string_view message, const Ts &...args) const {
    msg(log::Severity::Fatal, log::Verbosity::None, message, args...);
  }

  /*!
   * @brief Dumps data (context log::DataDump<T>, verbosity Trace). Nothing is copied.
   */
  template <std::size_t precision = log::default_precision, typename T>
  void data_dump(std::string_view what, const T &data) const {
    emit<log::DataDump<T>>(log::Severity::Normal, log::Verbosity::Trace, what,
                           [&] { return typename log::DataDump<T>::Payload{data, precision}; });
  }

  log::Severity min_severity() const { return m_min_severity; }

  void set_min_severity(log::Severity severity) { m_min_severity = severity; }

  log::Verbosity verbosity() const { return m_verbosity; }

  void set_verbosity(log::Verbosity verbosity) { m_verbosity = verbosity; }

  bool data_dumps_enabled() const { return m_dump_data; }

  void enable_data_dumps(bool enable) { m_dump_data = enable; }

  bool default_output() const { return m_default_output; }

  /*!
   * @brief Switches the built-in text output (default_message_handler()) on or off
   */
  void set_default_output(bool enable) { m_default_output = enable; }

protected:
  /*!
   * @brief Built-in text output. Receives every event passing the logger's thresholds and data-dump flag.
   *
   * Override this to redirect the text output.
   */
  virtual void default_message_handler(std::string_view ctx, log::Severity severity, log::Verbosity verbosity,
                                       std::string_view msg) const {
    bool add_colon = false;
    switch (severity) {
    case log::Severity::Normal:
      break;
    case log::Severity::Warning:
      molpro::cout << "[WARNING]";
      add_colon = true;
      break;
    case log::Severity::Error:
      molpro::cout << "[ERROR]:";
      add_colon = true;
      break;
    case log::Severity::Fatal:
      molpro::cout << "[FATAL]:";
      add_colon = true;
      break;
    }
    switch (verbosity) {
    case log::Verbosity::None:
      break;
    case log::Verbosity::Info:
      molpro::cout << "[INFO]";
      add_colon = true;
      break;
    case log::Verbosity::Debug:
      molpro::cout << "[DEBUG]";
      add_colon = true;
      break;
    case log::Verbosity::Trace:
      molpro::cout << "[TRACE]";
      add_colon = true;
      break;
    };

    if (ctx != log::Generic::name) {
      molpro::cout << "[" << ctx << "]";
      add_colon = true;
    }

    molpro::cout << (add_colon ? ": " : "") << msg;
    if (severity >= log::Severity::Error) {
      molpro::cout << std::endl;
    } else {
      molpro::cout << "\n";
    }
  }

private:
  struct SinkEntry {
    SinkId id;
    log::Filter filter;
    const std::type_info *context; //!< only events of this context are passed on, unless nullptr
    Sink sink;
  };

  bool passes(const log::Filter &filter, log::Severity severity, log::Verbosity verbosity, bool data_dump) const {
    return severity >= filter.min_severity.value_or(m_min_severity) &&
           verbosity >= filter.verbosity.value_or(m_verbosity) &&
           (!data_dump || filter.data_dumps.value_or(m_dump_data));
  }

  bool wants(const std::type_info &context, log::Severity severity, log::Verbosity verbosity, bool data_dump) const {
    if (m_default_output && passes({}, severity, verbosity, data_dump))
      return true;
    for (const auto &entry : m_sinks)
      if ((!entry.context || *entry.context == context) && passes(entry.filter, severity, verbosity, data_dump))
        return true;
    return false;
  }

  void dispatch(const log::Record &record) const {
    const bool data_dump = record.is_data_dump();
    for (const auto &entry : m_sinks)
      if ((!entry.context || *entry.context == record.type()) &&
          passes(entry.filter, record.severity, record.verbosity, data_dump))
        entry.sink(record);
    if (m_default_output && passes({}, record.severity, record.verbosity, data_dump))
      default_message_handler(record.context, record.severity, record.verbosity, record.text());
  }

  /// Lowest verbosity that will still be logged
  log::Verbosity m_verbosity = log::Verbosity::Info;
  /// Lowest severity that will still be logged
  log::Severity m_min_severity = log::Severity::Normal;
  /// Whether data dumps shall be performed
  bool m_dump_data = false;
  /// Whether the built-in text output is active
  bool m_default_output = true;
  /// Registered sinks
  std::vector<SinkEntry> m_sinks;
  SinkId m_next_id = 1;
};

} // namespace molpro::linalg::itsolv

#endif // LINEARALGEBRA_SRC_MOLPRO_LINALG_ITSOLV_LOGGER_H
