#ifndef LINEARALGEBRA_SRC_MOLPRO_LINALG_ARRAY_DEFAULT_HANDLER_H
#define LINEARALGEBRA_SRC_MOLPRO_LINALG_ARRAY_DEFAULT_HANDLER_H
#include <molpro/linalg/array/ArrayHandlerDDisk.h>
#include <molpro/linalg/array/ArrayHandlerDDiskDistr.h>
#include <molpro/linalg/array/ArrayHandlerDDiskSparse.h>
#include <molpro/linalg/array/ArrayHandlerDefault.h>
#include <molpro/linalg/array/ArrayHandlerDistr.h>
#include <molpro/linalg/array/ArrayHandlerDistrDDisk.h>
#include <molpro/linalg/array/ArrayHandlerDistrSparse.h>
#include <molpro/linalg/array/ArrayHandlerIterable.h>
#include <molpro/linalg/array/ArrayHandlerIterableSparse.h>
#include <molpro/linalg/array/ArrayHandlerSparse.h>
#include <molpro/linalg/array/type_traits.h>

namespace molpro::linalg::array {

template <class T, class S, ArrayFamily = array_family_v<T>, ArrayFamily = array_family_v<S>>
struct default_handler {
  using value = ArrayHandlerDefault<T, S>;
};

template <class T, class S>
struct default_handler<T, S, ArrayFamily::Iterable, ArrayFamily::Iterable> {
  using value = ArrayHandlerIterable<T, S>;
};

template <class T, class S>
struct default_handler<T, S, ArrayFamily::Sparse, ArrayFamily::Sparse> {
  using value = ArrayHandlerSparse<T, S>;
};

template <class T, class S>
struct default_handler<T, S, ArrayFamily::Distributed, ArrayFamily::Distributed> {
  using value = ArrayHandlerDistr<T, S>;
};

template <class T, class S>
struct default_handler<T, S, ArrayFamily::Iterable, ArrayFamily::Sparse> {
  using value = ArrayHandlerIterableSparse<T, S>;
};

template <class T, class S>
struct default_handler<T, S, ArrayFamily::Distributed, ArrayFamily::Sparse> {
  using value = ArrayHandlerDistrSparse<T, S>;
};

template <class T, class S>
struct default_handler<T, S, ArrayFamily::DistributedDisk, ArrayFamily::Distributed> {
  using value = ArrayHandlerDDiskDistr<T, S>;
};

template <class T, class S>
struct default_handler<T, S, ArrayFamily::Distributed, ArrayFamily::DistributedDisk> {
  using value = ArrayHandlerDistrDDisk<T, S>;
};

template <class T, class S>
struct default_handler<T, S, ArrayFamily::DistributedDisk, ArrayFamily::Sparse> {
  using value = ArrayHandlerDDiskSparse<T, S>;
};

template <class T, class S>
struct default_handler<T, S, ArrayFamily::DistributedDisk, ArrayFamily::DistributedDisk> {
  using value = ArrayHandlerDDisk<T, S>;
};

template <class T, class S>
using default_handler_t = typename default_handler<T, S>::value;

namespace detail {
//! SFINAE functor for default handler creation allows for more customisation
template <class T, class S, typename = default_handler_t<T, S>>
struct create_default_handler {
  auto operator()() { return std::make_shared<default_handler_t<T, S>>(); }
};
} // namespace detail

/*!
 * @brief Creates an appropriate handler for the given array types
 */
template <class T, class S>
auto create_default_handler() {
  return detail::create_default_handler<T, S>{}();
}

} // namespace molpro::linalg::array
#endif // LINEARALGEBRA_SRC_MOLPRO_LINALG_ARRAY_DEFAULT_HANDLER_H
