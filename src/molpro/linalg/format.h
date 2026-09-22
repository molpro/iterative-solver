#ifndef LINEARALGEBRA_SRC_MOLPRO_LINALG_FORMAT_H_
#define LINEARALGEBRA_SRC_MOLPRO_LINALG_FORMAT_H_

/*!
 * @file
 * @brief Selects the text-formatting backend: the standard library's <format>, or {fmt}.
 *
 * std::format is a C++20 library feature that arrived late in the implementations: libstdc++ has it
 * from GCC 13, and libc++ from LLVM 17, so a compiler that accepts -std=c++20 does not necessarily
 * provide it. {fmt} is the library std::format was modelled on and offers the same interface, so it
 * serves as a drop-in alternative where the standard one is missing, or where a project prefers it.
 *
 * The backend is chosen in this order:
 *  - MOLPRO_LINALG_USE_FMT or MOLPRO_LINALG_USE_STD_FORMAT, if either is defined, decides it. CMake
 *    defines one of them according to the LINEARALGEBRA_FORMAT option.
 *  - otherwise the standard library is used if it provides a complete <format>,
 *  - otherwise {fmt} is used if it is available,
 *  - otherwise the build fails with a diagnostic rather than a wall of template errors.
 *
 * Use molpro::linalg::fmtlib::format and friends rather than naming either backend directly. To
 * specialise the formatter for your own type, open the backend's namespace with the
 * MOLPRO_LINALG_FORMAT_NAMESPACE macro and write parse() and format() as templates on their context
 * types, as subspace::Matrix does; those signatures are the ones both backends accept.
 */

#if !defined(MOLPRO_LINALG_USE_FMT) && !defined(MOLPRO_LINALG_USE_STD_FORMAT)
#if __has_include(<format>)
#include <format>
#if defined(__cpp_lib_format) && __cpp_lib_format >= 201907L
#define MOLPRO_LINALG_USE_STD_FORMAT
#endif
#endif
#if !defined(MOLPRO_LINALG_USE_STD_FORMAT)
#if __has_include(<fmt/format.h>)
#define MOLPRO_LINALG_USE_FMT
#else
#error "No text formatting backend: this compiler provides no usable <format>, and {fmt} was not found. \
Install {fmt} and configure with -DLINEARALGEBRA_FORMAT=fmt, or use a compiler with C++20 <format> support."
#endif
#endif
#endif

#ifdef MOLPRO_LINALG_USE_FMT
#include <fmt/format.h>
//! The namespace the formatter specialisations have to be opened in
#define MOLPRO_LINALG_FORMAT_NAMESPACE fmt
#else
#include <format>
#define MOLPRO_LINALG_FORMAT_NAMESPACE std
#endif

namespace molpro::linalg {

/*!
 * @brief The formatting backend in use, whichever it is.
 *
 * These names are for *using* the backend. A formatter specialisation has to be declared in the
 * backend's own namespace, for which there is MOLPRO_LINALG_FORMAT_NAMESPACE.
 */
namespace fmtlib {
using MOLPRO_LINALG_FORMAT_NAMESPACE::format;
using MOLPRO_LINALG_FORMAT_NAMESPACE::format_context;
using MOLPRO_LINALG_FORMAT_NAMESPACE::format_error;
using MOLPRO_LINALG_FORMAT_NAMESPACE::format_parse_context;
using MOLPRO_LINALG_FORMAT_NAMESPACE::formatter;
} // namespace fmtlib

} // namespace molpro::linalg

#endif // LINEARALGEBRA_SRC_MOLPRO_LINALG_FORMAT_H_
