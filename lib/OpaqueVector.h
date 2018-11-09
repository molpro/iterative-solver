#ifndef OPAQUEVECTOR_H
#define OPAQUEVECTOR_H
#ifndef _GNU_SOURCE
#define _GNU_SOURCE 1
#endif

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <climits>
#include <cstring>
#include <cstddef>
#include <cstdlib>
#include <unistd.h>
#include <cassert>
#include <fstream>
#include <sstream>
#include <ostream>
#include <map>
#include <vector>
#include <numeric>

namespace LinearAlgebra {

/*!
  * \brief A class that implements a vector container that has the following features:
  * - opaque implementation of BLAS including dot(), axpy(), scal() as required by IterativeSolver without P-space
  * - import and export of data ranges
  * \tparam T the type of elements of the vector
  * \tparam Allocator alternative to std::allocator
  */
template<class T=double,
    class Allocator =
    std::allocator<T>
>
class OpaqueVector {
  typedef double scalar_type; //TODO implement this properly from T
  std::vector<T, Allocator> m_buffer;
 public:
  typedef T value_type;
  explicit OpaqueVector(size_t length = 0, const T& value = T())
      : m_buffer(length, value) {}
  /*!
   * @brief Copy constructor
   * @param source
   * @param option
   */
  OpaqueVector<T, Allocator>(const OpaqueVector& source, unsigned int option = 0) : m_buffer(source.m_buffer) {}

  /*!
    * \brief Add a constant times a sparse vector to this object
    * \param a The factor to multiply.
    * \param other The object to be added to this.
    * \return
    */
  void axpy(scalar_type a, const std::map<size_t, T>& other) {
    for (const auto& o: other)
      m_buffer[o.first] += a * o.second;
  }

  /*!
   * \brief Scalar product of two objects.
   * \param other The object to be contracted with this.
   * \return
   */
  scalar_type dot(const OpaqueVector<T>& other) const {
    assert(this->m_buffer.size() == other.m_buffer.size());
    return std::inner_product(m_buffer.begin(), m_buffer.end(), other.m_buffer.begin(), (scalar_type) 0);
  }

  //TODO this function should be removed once IterativeSolver doesn't need it to compile correctly
  /*!
   * \brief Scalar product with a sparse vector
   * \param other The object to be contracted with this.
   * \return
   */
  scalar_type dot(const std::map<size_t, T>& other) const {
    scalar_type result = 0;
    for (const auto& o: other)
      result += o.second * m_buffer[o.first];
    return result;
  }

  /*!
     * \brief scal Scale the object by a factor.
     * \param a The factor to scale by. If a is zero, then the current contents of the object are ignored.
     */
  void scal(scalar_type a) {
    if (a != 0)
      std::transform(m_buffer.begin(), m_buffer.end(), m_buffer.begin(), [a](T& x) { return a * x; });
    else
      m_buffer.assign(m_buffer.size(), 0);
  }

  /*!
   * \brief Add a constant times another object to this object
   * \param a The factor to multiply.
   * \param other The object to be added to this.
   * \return
   */
  void axpy(scalar_type a, const OpaqueVector<T>& other) {
    assert(this->m_buffer.size() == other.m_buffer.size());
    std::transform(other.m_buffer.begin(),
                   other.m_buffer.end(),
                   m_buffer.begin(),
                   m_buffer.begin(),
                   [a](T x, T y) -> T { return y + a * x; });
  }

  // below here is the optional interface. Above here is what is needed for IterativeSolver
  /*!
   * @brief Return the number of elements of data
   * @return
   */
  size_t size() const { return m_buffer.size(); }

  /*!
   * \brief Update a range of the object data with the contents of a provided buffer
   * \param buffer
   * \param length
   * \param offset
   */
  void put(const T* buffer, size_t length, size_t offset) {
    std::copy(buffer, buffer + length, &m_buffer[offset]);
  }

  /*!
   * \brief Read a range of the object data into a provided buffer
   * @param buffer
   * @param length
   * @param offset
   */
  void get(T* buffer, size_t length, size_t offset) const {
    std::copy(&m_buffer[offset], &m_buffer[offset + length], buffer);
  }

};

}
#endif // OPAQUEVECTOR_H
