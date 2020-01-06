#ifndef OUTOFCOREARRAY_H
#define OUTOFCOREARRAY_H
#include "IterativeSolver-config.h"
#ifdef TIMING
#include <chrono>
#endif
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

#ifdef MOLPRO
#define HAVE_MPI_H
#endif

#ifdef HAVE_MPI_H
#include <mpi.h>
#ifdef MOLPRO
#include "ppidd.h"
#define MPI_COMM_COMPUTE MPI_Comm_f2c(PPIDD_Worker_comm())
#else
#define MPI_COMM_COMPUTE MPI_COMM_WORLD
#endif
#else
#define MPI_COMM_COMPUTE 0
using MPI_Comm = int;
#endif

namespace LinearAlgebra {

/*!
  * \brief A class that implements a fixed-size vector container that has the following public features:
  * - data held on backing store, distributed over MPI ranks, instead of in memory
  * - alternatively, mapping of an externally-owned buffer
  * - opaque implementation of BLAS including dot(), axpy(), scal()
  * - additional BLAS overloads to combine with a simple sparse vector represented in std::map
  * - efficient import and export of data ranges
  * - additional functions to find the largest elements, and to print the vector
  * - underlying type of elements
  * \tparam T the type of elements of the vector
  * \tparam Allocator alternative to std::allocator
  */
template<class T=double, class Allocator = std::allocator<T> >
class OutOfCoreArray {
  typedef double scalar_type; //TODO implement this properly from T
 public:
  typedef T value_type;
  /*!
   * \brief Construct an object without any data.
   */
  explicit OutOfCoreArray(size_t length = 0, unsigned int option = 0, MPI_Comm mpi_communicator = MPI_COMM_COMPUTE)
      : m_size(length),
        m_communicator(mpi_communicator), m_mpi_size(mpi_size()), m_mpi_rank(mpi_rank()),
        m_data(nullptr),
        m_sync(true),
        m_segment_offset(seg_offset()),
        m_segment_length(seg_length()) {
//   init(option);
//    std::cout <<" OutOfCoreArray constructor length="<<length<<", option "<<( option)<<std::endl;
//    std::cout <<"LINEARALGEBRA_OFFLINE & option "<<(LINEARALGEBRA_OFFLINE & option)<<std::endl;
//    std::cout <<"cache preferred length "<<m_cache.preferred_length<<std::endl;
//   std::cout << m_mpi_rank << " in constructor m_segment_length="<<m_segment_length<<", m_segment_offset="<<m_segment_offset<<std::endl;
  }
  /*!
   * @brief Copy constructor
   * @param source
   * @param option
   * @param mpi_communicator
   */
  OutOfCoreArray(const OutOfCoreArray& source)
      : m_size(source.m_size),
        m_communicator(source.m_communicator), m_mpi_size(mpi_size()), m_mpi_rank(mpi_rank()),
        m_sync(source.m_sync),
        m_segment_offset(seg_offset()),
        m_segment_length(seg_length()) {
#ifdef TIMING
    auto start=std::chrono::steady_clock::now();
#endif
    *this = source;
#ifdef TIMING
    auto end=std::chrono::steady_clock::now();
    std::cout <<" OutOfCoreArray copy constructor length="<<m_size<<", option "<<( option)
    <<", seconds="<<std::chrono::duration_cast<std::chrono::nanoseconds>(end-start).count()*1e-9
        <<", bandwidth="<<m_size*8*1e3/std::chrono::duration_cast<std::chrono::nanoseconds>(end-start).count()<<"MB/s"
    <<std::endl;
#endif
  }

  /*!
  * @brief Construct an object mapped from external data
   *
   * @param buffer
   * @param length
   * @param mpi_communicator Not used by this object, but can be specified so that it is taken when copy-constructing from this object
   */

  OutOfCoreArray(T* buffer, size_t length, MPI_Comm mpi_communicator = MPI_COMM_COMPUTE)
      : m_size(length),
        m_communicator(MPI_COMM_COMPUTE), m_mpi_size(mpi_size()), m_mpi_rank(mpi_rank()),
        m_data(buffer),
        m_sync(true),
        m_segment_offset(0),
        m_segment_length(m_size) {
//    std::cout << "OutOfCoreArray map constructor "<<buffer<<" : "<<&(*this)[0]<<std::endl;
  }

  ~OutOfCoreArray() = default;

  bool synchronised() const { return m_sync; }

 private:
  int mpi_size() {
#ifdef HAVE_MPI_H
    int result;
    MPI_Initialized(&result);
    if (!result) MPI_Init(0,nullptr);
    MPI_Comm_size(m_communicator, &result);
#else
    int result = 1;
#endif
    return result;
  }
  int mpi_rank() {
#ifdef HAVE_MPI_H
    int result;
    MPI_Comm_rank(m_communicator, &result);
#else
    int result = 0;
#endif
    return result;
  }

  size_t m_size; //!< How much data
  const MPI_Comm m_communicator; //!< the MPI communicator for distributed data
  int m_mpi_size;
  int m_mpi_rank;
  T* m_data; //!< the data buffer (if in an externally-provided buffer) or nullptr (if owned by this object)
  bool m_sync; //!< whether replicated data is in sync on all processes
  size_t m_segment_offset; //!< offset in the overall data object of this process' data
  size_t m_segment_length; //!< length of this process' data

  /*!
 * Calculate the segment length for asynchronous operations on replicated vectors
 * @return the length of a vector segment
 */
  size_t seg_length() const {
    size_t alpha = m_size / m_mpi_size;
    size_t beta = m_size % m_mpi_size;
    size_t segl = (m_mpi_rank < beta) ? alpha + 1 : alpha;
    return segl;
  }

  /*!
 * Calculate the segment offset for asynchronous operations on replicated vectors
 * @return the offset of a vector segment
 */
  size_t seg_offset() const {
    size_t alpha = m_size / m_mpi_size;
    size_t beta = m_size % m_mpi_size;
    size_t segoff = (m_mpi_rank < beta) ? (alpha + 1) * m_mpi_rank : alpha * m_mpi_rank + beta;
    return segoff;
  }

  /*!
 * Get a pointer to the entire data structure if possible
 * @return the pointer, or nullptr if caching/distribution means that the whole vector is not available
 */
  T* data() {
    return m_data;
  }

 public:
  /*!
   * \brief Update a range of the object data with the contents of a provided buffer
   * \param buffer
   * \param length
   * \param offset
   */
  void put(const T* buffer, size_t length, size_t offset) {
//   std::cout << "OutOfCoreArray::put() length="<<length<<", offset="<<offset<<std::endl;
//   for (size_t k=0; k<length; k++) std::cout << " "<<buffer[k]; std::cout << std::endl;
//   std::cout << "cache from "<<m_cache.offset<<" for "<<m_cache.length<<std::endl;
//   for (size_t k=0; k<m_cache.length; k++) std::cout << " "<<m_cache.buffer[k]; std::cout << std::endl;
//     std::cout << "m_segment_offset "<<m_segment_offset<<std::endl;
    // first of all, focus attention only on that part of buffer which appears in [m_segment_offset,m_segment_offset+m_segment_length)
    size_t buffer_offset = 0;
    if (offset < m_segment_offset) {
      buffer_offset = m_segment_offset - offset;
      offset = m_segment_offset;
      length -= m_segment_offset - offset;
    }
    if (offset + length > std::min(m_size, m_segment_offset + m_segment_length))
      length = std::min(m_size,
                        m_segment_offset + m_segment_length) - offset;

    // now make addresses relative to this mpi-rank's segment
    offset -= this->m_segment_offset; // the offset in the segment of the first usable element of buffer
//   buffer_offset += 0*this->m_segment_offset; // the offset in buffer of its first usable element

//   std::cout << "adjusted offset="<<offset<<", buffer_offset="<<buffer_offset<<std::endl;

    // first of all, process that part of the data in the initial cache window
    for (size_t k = std::max(offset, m_cache.offset); k < std::min(offset + length, m_cache.offset + m_cache.length);
         k++) { // k is offset in segment
      m_cache.buffer[k - m_cache.offset] = buffer[k + buffer_offset - offset];
      m_cache.dirty = true;
//    std::cout <<"in initial window, k="<<k<<", m_cache_buffer["<<k-m_cache.offset<<"]=buffer["<<k+buffer_offset-offset<<"]="<<buffer[k+buffer_offset-offset]<<std::endl;
    }

//   std::cout <<"after initial window"<<std::endl;
    // next, process the data appearing before the initial cache window
    size_t initial_cache_offset = m_cache.offset;
    size_t initial_cache_length = m_cache.length;
    for (m_cache.move(offset); m_cache.length && m_cache.offset < initial_cache_offset; ++m_cache) {
//    std::cout << "cache segment buffer range "<<std::max(offset,m_cache.offset)+buffer_offset-offset<<" : "<<m_cache.offset+m_cache.length+buffer_offset-offset<<std::endl;
      for (size_t k = std::max(offset, m_cache.offset);
           k < m_cache.offset + m_cache.length && k < initial_cache_offset && k + buffer_offset < offset + length;
           k++) { // k is offset in segment
//     if (k+buffer_offset<offset) std::cout <<"accessing data before buffer"<<std::endl;
//     if (k+buffer_offset-offset>length) std::cout <<"accessing data after buffer"<<std::endl;
        m_cache.buffer[k - m_cache.offset] = buffer[k + buffer_offset - offset];
      }
//    std::cout <<"processed preceding window"<<std::endl;
      m_cache.dirty = true;
    }

    // finally, process the data appearing after the initial cache window
//std::cout << "initial_cache_offset="<<initial_cache_offset<<std::endl;
    for (m_cache.move(initial_cache_offset + initial_cache_length); m_cache.length && m_cache.offset < offset + length;
         ++m_cache) {
      for (size_t k = m_cache.offset; k < m_cache.offset + m_cache.length && k < offset + length; k++)
        m_cache.buffer[k - m_cache.offset] = buffer[k + buffer_offset - offset];
//    std::cout <<"processed following window"<<std::endl;
      m_cache.dirty = true;
    }

//   std::cout << "OutOfCoreArray::put() ends length="<<length<<", offset="<<offset<<std::endl;
  }

  /*!
   * \brief Read a range of the object data into a provided buffer
   * @param buffer
   * @param length
   * @param offset
   */
  void get(T* buffer, size_t length, size_t offset) const {

    // first of all, focus attention only on that part of buffer which appears in [m_segment_offset,m_segment_offset+m_segment_length)
    size_t buffer_offset = 0;
    if (offset < m_segment_offset) {
      buffer_offset = m_segment_offset - offset;
      offset = m_segment_offset;
      length -= m_segment_offset - offset;
    }
    if (offset + length > std::min(m_size, m_segment_offset + m_segment_length))
      length = std::min(m_size,
                        m_segment_offset + m_segment_length) - offset;

    // now make addresses relative to this mpi-rank's segment
    offset -= this->m_segment_offset; // the offset in the segment of the first usable element of buffer
    buffer_offset += 0 * this->m_segment_offset; // the offset in buffer of its first usable element

//   std::cout << "adjusted offset="<<offset<<", buffer_offset="<<buffer_offset<<std::endl;

    // first of all, process that part of the data in the initial cache window
    for (size_t k = std::max(offset, m_cache.offset); k < std::min(offset + length, m_cache.offset + m_cache.length);
         k++) { // k is offset in segment
      buffer[k + buffer_offset - offset] = m_cache.buffer[k - m_cache.offset];
//    std::cout <<"in initial window, k="<<k<<", m_cache_buffer["<<k-m_cache.offset<<"]=buffer["<<k+buffer_offset-offset<<"]="<<buffer[k+buffer_offset-offset]<<std::endl;
    }

    // next, process the data appearing before the initial cache window
    size_t initial_cache_offset = m_cache.offset;
    size_t initial_cache_length = m_cache.length;
    for (m_cache.move(offset); m_cache.length && m_cache.offset < initial_cache_offset; ++m_cache) {
//    std::cout <<"new cache window offset="<<m_cache.offset<<", length="<<m_cache.length<<std::endl;
      for (size_t k = std::max(offset, m_cache.offset);
           k < m_cache.offset + m_cache.length && k < initial_cache_offset && k + buffer_offset < offset + length;
           k++) { // k is offset in segment
//     std::cout << "in loop k="<<k<<", buffer_offset="<<buffer_offset<<", offset="<<offset<<" index="<<k+buffer_offset-offset<<std::endl;
        buffer[k + buffer_offset - offset] = m_cache.buffer[k - m_cache.offset];
//    std::cout <<"in preceding window, k="<<k<<", m_cache_buffer["<<k-m_cache.offset<<"]=buffer["<<k+buffer_offset-offset<<"]="<<buffer[k+buffer_offset-offset]<<std::endl;
      }
//    std::cout <<"processed preceding window"<<std::endl;
    }

    // finally, process the data appearing after the initial cache window
    for (m_cache.move(initial_cache_offset + initial_cache_length); m_cache.length && m_cache.offset < offset + length;
         ++m_cache) {
      for (size_t k = m_cache.offset; k < m_cache.offset + m_cache.length && k < offset + length; k++)
        buffer[k + buffer_offset - offset] = m_cache.buffer[k - m_cache.offset];
//    std::cout <<"processed following window"<<std::endl;
    }

//   for (size_t k=0; k<length; k++) std::cout << " "<<buffer[k]; std::cout << std::endl;
//   std::cout << "OutOfCoreArray::get() ends, length="<<length<<", offset="<<offset<<std::endl;
  }

  /*!
   * @brief Return a reference to an element of the data
   * @param pos Offset of the data
   * @return
   */
  const T& operator[](size_t pos) const {
    if (pos >= m_cache.offset + m_segment_offset + m_cache.length
        || pos < m_cache.offset + m_segment_offset) { // cache not mapping right sector
//    std::cout << "cache miss"<<std::endl;
      if (pos >= m_segment_offset + m_segment_length || pos < m_segment_offset)
        throw std::logic_error("operator[] finds index out of range");
      m_cache.move(pos - m_segment_offset);
//    std::cout << "cache offset="<<m_cache.offset<<", cache length=" <<m_cache.length<<std::endl;
    }
    return m_cache.buffer[pos - m_cache.offset - m_segment_offset];
  }

  /*!
   * @brief Return a reference to an element of the data
   * @param pos Offset of the data
   * @return
   */
  T& operator[](size_t pos) {
    T* result;
    result = &const_cast<T&>(static_cast<const OutOfCoreArray*>(this)->operator[](pos));
    m_cache.dirty = true;
    return *result;
  }

  /*!
   * @brief Return the number of elements of data
   * @return
   */
  size_t size() const { return m_size; }

  /*!
   * @brief Produce a printable representation of the object
   * @param verbosity How much to print
   * @param columns Width in characters
   * @return
   */
  std::string str(int verbosity = 0, unsigned int columns = UINT_MAX) const {
    std::ostringstream os;
    os << "OutOfCoreArray object:";
    for (size_t k = 0; k < m_segment_length; k++)
      os << " " << (*this)[m_segment_offset + k];
    os << std::endl;
    return os.str();
  }

  /*
   * \brief replicate the vectors over the MPI ranks
   * \return
   */
  void sync() {
    if (m_replicated) {
#ifdef HAVE_MPI_H
      // std::cout <<m_mpi_rank<<"before broadcast this="<<*this<<std::endl;
            size_t alpha = m_size/m_mpi_size;
            size_t beta = m_size%m_mpi_size;
      //    std::cout <<m_mpi_rank<<"lenseg="<<lenseg<<std::endl;
            if (m_cache.io) { // ! this needs to be probably re-written ! - has not been touched after moving from axpy() (still MPI_Bcast())
              //size_t lenseg = ((m_size - 1) / m_mpi_size + 1);
              size_t lenseg = seg_length();;
              for (int rank = 0; rank < m_mpi_size; rank++) {
                for (m_cache.move(lenseg * rank, std::min(m_cache.preferred_length, lenseg));
                     m_cache.length && m_cache.offset < lenseg * (rank + 1); ++m_cache) {
                  size_t l = std::min(m_cache.length, (rank + 1) * lenseg - m_cache.offset);
      //      std::cout << m_mpi_rank<<"broadcast from "<<rank<<" offset="<<m_cache.offset<<", length="<<l<<std::endl;
      //      std::cout <<m_mpi_rank<<"before Bcast"; for (auto i=0; i<l; i++) std::cout <<" "<<m_cache.buffer[i]; std::cout <<std::endl;
                  if (l > 0)
                    MPI_Bcast(m_cache.buffer, l, MPI_DOUBLE, rank, m_communicator); // needs attention for non-double
      //      std::cout <<m_mpi_rank<<"after Bcast m_cache.length="<<m_cache.length<<", lenseg="<<lenseg<<", m_cache.offset="<<m_cache.offset<<std::endl;
      //      std::cout <<m_mpi_rank<<"after Bcast"; for (auto i=0; i<l; i++) std::cout <<" "<<m_cache.buffer[i]; std::cout <<std::endl;
                  if (rank != m_mpi_rank) m_cache.dirty = true;
      //       usleep(100);
                }
              }
            } else {
              int chunks[m_mpi_size];
              int displs[m_mpi_size];
              for (int i = 0; i < m_mpi_size; i++) {
                  chunks[i] = (i < beta) ? alpha+1 : alpha;
                  displs[i] = (i < beta) ? chunks[i]*i : chunks[i]*i + beta;
      //            if (m_mpi_rank == 0) std::cout<<"Chunk: "<<chunks[i]<<" Displ: "<<displs[i]<<std::endl;
              }
              MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL,&m_cache.buffer[0],chunks,displs,MPI_DOUBLE,m_communicator); // May want to try non-blocking
            }
#endif
      m_sync = true;
    }
  }

  /*!
   * \brief Add a constant times another object to this object
   * \param a The factor to multiply.
   * \param other The object to be added to this.
   * \return
   */
  void axpy(scalar_type a, const OutOfCoreArray<T>& other) {
    if (this->m_size != m_size) throw std::logic_error("mismatching lengths");
    if (this->m_replicated == other.m_replicated) {
      if (this->m_cache.io && other.m_cache.io) {
        size_t l = std::min(m_cache.length, other.m_cache.length);
        for (m_cache.move(0, l), other.m_cache.move(0, l); m_cache.length; ++m_cache, ++other.m_cache)
          for (size_t i = 0; i < m_cache.length; i++) {
            m_cache.buffer[i] += a * other.m_cache.buffer[i];
            m_cache.dirty = true;
          }
      } else if (other.m_cache.io) {
        size_t offset = 0;
        for (other.m_cache.ensure(0); other.m_cache.length; offset += other.m_cache.length, ++other.m_cache) {
          for (size_t i = 0; i < other.m_cache.length; i++) {
            m_cache.buffer[offset + i] += a * other.m_cache.buffer[i];
          }
        }
      } else if (this->m_cache.io) {
        size_t offset = 0;
        for (m_cache.ensure(0); m_cache.length; offset += m_cache.length, ++m_cache) {
          for (size_t i = 0; i < m_cache.length; i++)
            m_cache.buffer[i] += a * other.m_cache.buffer[offset + i];
          m_cache.dirty = true;
        }
      } else if (this->m_replicated) {
        size_t seglength = seg_length();
        size_t segoffset = seg_offset();
        for (size_t i = 0; i < seglength; i++)
          m_cache.buffer[segoffset + i] += a * other.m_cache.buffer[segoffset + i];
      } else {
        for (size_t i = 0; i < this->m_segment_length; i++)
          m_cache.buffer[i] += a * other.m_cache.buffer[i];
      }
    } else if (this->m_replicated) { // this is replicated, other is not replicated
      if (m_cache.io && other.m_cache.io) {
        auto l = std::min(m_cache.preferred_length, other.m_cache.preferred_length);
        for (other.m_cache.move(0, l), m_cache.move(other.m_segment_offset,
                                                    l); other.m_cache.length; ++m_cache, ++other.m_cache) {
          for (size_t i = 0; i < other.m_cache.length; i++)
            m_cache.buffer[i] += a * other.m_cache.buffer[i];
          m_cache.dirty = true;
        }
      } else if (m_cache.io) {
        for (m_cache.ensure(other.m_segment_offset);
             m_cache.offset < other.m_segment_offset + other.m_segment_length && m_cache.length; ++m_cache) {
          for (size_t i = 0; i < m_cache.length; i++)
            m_cache.buffer[i] += a * other.m_cache.buffer[m_cache.offset - other.m_segment_offset + i];
          m_cache.dirty = true;
        }
      } else if (other.m_cache.io) {
        for (other.m_cache.ensure(0); other.m_cache.length; ++other.m_cache)
          for (size_t i = 0; i < other.m_cache.length; i++)
            m_cache.buffer[other.m_cache.offset + other.m_segment_offset + i] += a * other.m_cache.buffer[i];
      } else {
        for (size_t i = 0; i < other.m_segment_length; i++)
          m_cache.buffer[other.m_segment_offset + i] += a * other.m_cache.buffer[i];
      }

    } else { // this is not replicated, other is replicated
      if (m_cache.io && other.m_cache.io) {
        auto l = std::min(m_cache.preferred_length, other.m_cache.preferred_length);
        for (other.m_cache.move(m_segment_offset, l), m_cache.move(0, l); m_cache.length; ++m_cache, ++other.m_cache) {
          for (size_t i = 0; i < m_cache.length; i++)
            m_cache.buffer[i] += a * other.m_cache.buffer[i];
          m_cache.dirty = true;
        }
      } else if (m_cache.io) {
        for (m_cache.ensure(0); m_cache.length; ++m_cache) {
          for (size_t i = 0; i < m_cache.length; i++)
            m_cache.buffer[i] += a * other.m_cache.buffer[m_cache.offset + m_segment_offset + i];
          m_cache.dirty = true;
        }
      } else if (other.m_cache.io) {
        for (other.m_cache.ensure(m_segment_offset);
             other.m_cache.offset < m_segment_offset + m_segment_length && other.m_cache.length; ++other.m_cache)
          for (size_t i = 0;
               i < std::min(other.m_cache.length, m_segment_offset + m_segment_length - other.m_cache.offset); i++)
            m_cache.buffer[other.m_cache.offset - m_segment_offset + i] += a * other.m_cache.buffer[i];
      } else {
        for (size_t i = 0; i < m_segment_length; i++)
          m_cache.buffer[i] += a * other.m_cache.buffer[m_segment_offset + i];
      }
    }
    if (m_replicated) m_sync = false;
  }

  /*!
    * \brief Add a constant times a sparse vector to this object
    * \param a The factor to multiply.
    * \param other The object to be added to this.
    * \return
    */
  void axpy(scalar_type a, const std::map<size_t, T>& other) {
    for (const auto& o: other)
      if (o.first >= m_segment_offset && o.first < m_segment_offset + m_segment_length)
        (*this)[o.first] += a * o.second;
  }

  /*!
   * \brief Scalar product of two objects.
   * \param other The object to be contracted with this.
   * \return
   */
  scalar_type dot(const OutOfCoreArray<T, default_offline_buffer_size>& other) const {
    if (this->m_size != m_size) throw std::logic_error("mismatching lengths");
    scalar_type result = 0;
    if (this == &other) {
      if (m_cache.io) {
        for (m_cache.ensure(0); m_cache.length; ++m_cache) {
          for (size_t i = 0; i < m_cache.length; i++) {
            result += m_cache.buffer[i] * m_cache.buffer[i];
          }
        }
      } else if (this->m_replicated) {
        size_t seglength = seg_length();
        size_t segoffset = seg_offset();
        for (size_t i = 0; i < seglength; i++)
          result += m_cache.buffer[segoffset + i] * m_cache.buffer[segoffset + i];
      } else {
        for (size_t i = 0; i < this->m_segment_length; i++)
          result += m_cache.buffer[i] * m_cache.buffer[i];
      }
    } else {
      if (this->m_replicated == other.m_replicated) {
        if (this->m_cache.io && other.m_cache.io) {
          size_t l = std::min(m_cache.length, other.m_cache.length);
          for (m_cache.move(0, l), other.m_cache.move(0, l); m_cache.length; ++m_cache, ++other.m_cache)
            for (size_t i = 0; i < m_cache.length; i++)
              result += m_cache.buffer[i] * other.m_cache.buffer[i];
        } else if (other.m_cache.io) {
          size_t offset = 0;
          for (other.m_cache.ensure(0); other.m_cache.length; offset += other.m_cache.length, ++other.m_cache) {
            for (size_t i = 0; i < other.m_cache.length; i++) {
              result += m_cache.buffer[offset + i] * other.m_cache.buffer[i];
            }
          }
        } else if (this->m_cache.io) {
          size_t offset = 0;
          for (m_cache.ensure(0); m_cache.length; offset += m_cache.length, ++m_cache)
            for (size_t i = 0; i < m_cache.length; i++)
              result += m_cache.buffer[i] * other.m_cache.buffer[offset + i];
        } else if (this->m_replicated) { // both are replicated
          size_t seglength = seg_length();
          size_t segoffset = seg_offset();
          for (size_t i = 0; i < seglength; i++)
            result += m_cache.buffer[segoffset + i] * other.m_cache.buffer[segoffset + i];
        } else { // both are distributed
//          size_t l = std::min(m_cache.length, other.m_cache.length); // FIXME puzzle as to what this is
          //std::cout<<"Rank "<<m_mpi_rank<<" "; for (auto i=0; i<m_segment_length; i++) std::cout <<" "<<m_cache.buffer[i]; std::cout <<std::endl;
          for (size_t i = 0; i < this->m_segment_length; i++)
            result += m_cache.buffer[i] * other.m_cache.buffer[i];
        }
      } else if (this->m_replicated) { // this is replicated, other is not replicated
        if (m_cache.io && other.m_cache.io) {
          auto l = std::min(m_cache.preferred_length, other.m_cache.preferred_length);
          for (other.m_cache.move(0, l), m_cache.move(other.m_segment_offset,
                                                      l); other.m_cache.length; ++m_cache, ++other.m_cache) {
            for (size_t i = 0; i < other.m_cache.length; i++)
              result += m_cache.buffer[i] * other.m_cache.buffer[i];
          }
        } else if (m_cache.io) {
          for (m_cache.ensure(other.m_segment_offset);
               m_cache.offset < other.m_segment_offset + other.m_segment_length && m_cache.length; ++m_cache) {
            for (size_t i = 0;
                 i < std::min(m_cache.length, other.m_segment_offset + other.m_segment_length - m_cache.offset); i++)
              result += m_cache.buffer[i] * other.m_cache.buffer[m_cache.offset - other.m_segment_offset + i];
          }
        } else if (other.m_cache.io) {
          for (other.m_cache.ensure(0); other.m_cache.length; ++other.m_cache)
            for (size_t i = 0; i < other.m_cache.length; i++)
              result += m_cache.buffer[other.m_cache.offset + other.m_segment_offset + i] * other.m_cache.buffer[i];
        } else {
          for (size_t i = 0; i < other.m_segment_length; i++)
            result += m_cache.buffer[other.m_segment_offset + i] * other.m_cache.buffer[i];
        }

      } else { // this is not replicated, other is replicated
        if (m_cache.io && other.m_cache.io) {
          auto l = std::min(m_cache.preferred_length, other.m_cache.preferred_length);
          for (other.m_cache.move(m_segment_offset, l), m_cache.move(0, l); m_cache.length;
               ++m_cache, ++other.m_cache) {
            for (size_t i = 0; i < m_cache.length; i++)
              result += m_cache.buffer[i] * other.m_cache.buffer[i];
          }
        } else if (m_cache.io) {
          for (m_cache.ensure(0); m_cache.length; ++m_cache) {
            for (size_t i = 0; i < m_cache.length; i++)
              result += m_cache.buffer[i] * other.m_cache.buffer[m_cache.offset + m_segment_offset + i];
          }
        } else if (other.m_cache.io) {
//      std::cout << "hello12"<<std::endl;
          for (other.m_cache.ensure(m_segment_offset);
               other.m_cache.offset < m_segment_offset + m_segment_length && other.m_cache.length; ++other.m_cache)
            for (size_t i = 0;
                 i < std::min(other.m_cache.length, m_segment_offset + m_segment_length - other.m_cache.offset); i++)
              result += m_cache.buffer[other.m_cache.offset - m_segment_offset + i] * other.m_cache.buffer[i];
        } else {
          for (size_t i = 0; i < m_segment_length; i++)
            result += m_cache.buffer[i] * other.m_cache.buffer[m_segment_offset + i];
        }
      }
    }
#ifdef HAVE_MPI_H
    //    std::cout <<m_mpi_rank<<" dot result before reduce="<<result<<std::endl;
    //if (!m_replicated || !other.m_replicated) {
      double resultLocal = result;
      double resultGlobal = result;
      MPI_Allreduce(&resultLocal, &resultGlobal, 1, MPI_DOUBLE, MPI_SUM, m_communicator);
      result = resultGlobal;
      //    std::cout <<m_mpi_rank<<" dot result after reduce="<<result<<std::endl;
    //}
#endif
    return result;
  }

  /*!
   * \brief Scalar product with a sparse vector
   * \param other The object to be contracted with this.
   * \return
   */
  scalar_type dot(const std::map<size_t, T>& other) const {
    scalar_type result = 0;
    if (m_replicated) {
      size_t seglength = seg_length();
      size_t segoffset = seg_offset();
      for (const auto& o: other)
        if (o.first >= segoffset && o.first < segoffset + seglength)
          result += o.second * (*this)[o.first];
    } else {
      for (const auto& o: other)
        if (o.first >= m_segment_offset && o.first < m_segment_offset + m_segment_length)
          result += o.second * (*this)[o.first];
    }
#ifdef HAVE_MPI_H
    //if (!m_replicated) {
      double resultLocal = result;
      MPI_Allreduce(&resultLocal,
                    &result,
                    1,
                    MPI_DOUBLE,
                    MPI_SUM,
                    m_communicator); // FIXME needs attention for non-double
    //}
#endif
    return result;
  }

  /*!
     * \brief scal Scale the object by a factor.
     * \param a The factor to scale by. If a is zero, then the current contents of the object are ignored.
     */
  void scal(scalar_type a) {
    for (m_cache.ensure(0); m_cache.length; ++m_cache) {
      if (a != 0)
        for (size_t i = 0; i < m_cache.length; i++)
          m_cache.buffer[i] *= a;
      else
        for (size_t i = 0; i < m_cache.length; i++)
          m_cache.buffer[i] = 0;
      m_cache.dirty = true;
    }
  }

  /*!
    * Find the largest values of the object.
    * @param measure A vector of the same size and matching covariancy, with which the largest contributions to the scalar
    * product with *this are selected.
    * @param maximumNumber At most this number of elements are returned.
    * @param threshold Contributions to the scalar product smaller than this are not included.
    * @return index, value pairs. value is the product of the matrix element and the corresponding element of measure.
    *
    */
  std::tuple<std::vector<size_t>, std::vector<T> > select(
      const OutOfCoreArray<T>& measure,
      const size_t maximumNumber = 1000,
      const scalar_type threshold = 0
  ) const {
    std::multimap<T, size_t, std::greater<T> > sortlist;
    const auto& measur = dynamic_cast <const OutOfCoreArray<T>&> (measure);
    if (this->m_size != m_size) throw std::logic_error("mismatching lengths");
    if (this->m_replicated != measur.m_replicated) throw std::logic_error("mismatching replication status");
    if (this == &measur) {
      for (m_cache.ensure(0); m_cache.length; ++m_cache) {
        for (size_t i = 0; i < m_cache.length; i++) {
          auto test = m_cache.buffer[i] * m_cache.buffer[i];
          if (test > threshold) {
            sortlist.insert(std::make_pair(test, i));
            if (sortlist.size() > maximumNumber) sortlist.erase(std::prev(sortlist.end()));
          }
        }
      }
    } else {
      for (m_cache.ensure(0), measur.m_cache.move(0, m_cache.length); m_cache.length; ++m_cache, ++measur.m_cache) {
        for (size_t i = 0; i < m_cache.length; i++) {
          scalar_type test = m_cache.buffer[i] * measur.m_cache.buffer[i];
          if (test < 0) test = -test;
          if (test > threshold) {
            sortlist.insert(std::make_pair(test, i));
            if (sortlist.size() > maximumNumber) sortlist.erase(std::prev(sortlist.end()));
          }
        }
      }
    }
#ifdef HAVE_MPI_H
    if (!m_replicated) {
      throw std::logic_error("incomplete MPI implementation");
    }
#endif
    while (sortlist.size() > maximumNumber) sortlist.erase(std::prev(sortlist.end()));
    std::vector<size_t> indices;
    indices.reserve(sortlist.size());
    std::vector<T> values;
    values.reserve(sortlist.size());
    for (const auto& p : sortlist) {
      indices.push_back(p.second);
      values.push_back(p.first);
    }
    return std::make_tuple(indices, values);

  };

  /*!
   * \brief Copy from one object to another, adjusting size if needed.
   * \param other The source of data.
   * \return
   */
  OutOfCoreArray& operator=(const OutOfCoreArray& other) {
    assert(m_size == other.m_size);
    if (!m_cache.io && !other.m_cache.io) { // std::cout << "both in memory"<<std::endl;
      size_t off = (m_replicated && !other.m_replicated) ? other.m_segment_offset : 0;
      size_t otheroff = (!m_replicated && other.m_replicated) ? m_segment_offset : 0;
      for (size_t i = 0; i < std::min(m_segment_length, other.m_segment_length); i++)
        m_cache.buffer[off + i] = other.m_cache.buffer[otheroff + i];
    } else if (m_cache.io && other.m_cache.io) { // std::cout << "both cached"<<std::endl;
      size_t cachelength = std::min(m_cache.preferred_length, other.m_cache.preferred_length);
      size_t off = 0, otheroff = 0;
      m_cache.move((!m_replicated || other.m_replicated) ? 0 : other.m_segment_offset, cachelength);
      other.m_cache.move((m_replicated || !other.m_replicated) ? 0 : m_segment_offset, cachelength);
      while (m_cache.length && other.m_cache.length && off < m_segment_length && otheroff < other.m_segment_length) {
        for (size_t i = 0; i < m_cache.length; i++)
          m_cache.buffer[off + i] = other.m_cache.buffer[otheroff + i];
        m_cache.dirty = true;
        if (m_cache.io) ++m_cache; else off += cachelength;
        if (other.m_cache.io) ++other.m_cache; else otheroff += cachelength;
      }
    } else if (m_cache.io) { // std::cout<< "source in memory, result cached"<<std::endl;
      size_t cachelength = m_cache.preferred_length;
      size_t off = 0;
      size_t otheroff = (m_replicated == other.m_replicated) ? 0 : m_segment_offset;
      m_cache.move((m_replicated == other.m_replicated) ? 0 : other.m_segment_offset, cachelength);
      other.m_cache.move(0, other.m_segment_length);
      while (m_cache.length && off < m_segment_length && otheroff < other.m_segment_length) {
        for (size_t i = 0; i < m_cache.length; i++) {
          m_cache.buffer[off + i] = other.m_cache.buffer[otheroff + i];
          //     if (pr) std::cout <<m_mpi_rank<<" buffer["<<off+i<<"]="<<m_cache.buffer[off+i]<<std::endl;
        }
        m_cache.dirty = true;
        ++m_cache;
        otheroff += cachelength;
      }
    } else if (other.m_cache.io) { // std::cout << "source cached, result in memory"<<std::endl;
      size_t cachelength = other.m_cache.preferred_length;
      size_t otheroff = 0;
      size_t off = (m_replicated == other.m_replicated) ? 0 : other.m_segment_offset;
      m_cache.move(0, m_segment_length);
      other.m_cache.move((m_replicated == other.m_replicated) ? 0 : m_segment_offset, cachelength);
      while (other.m_cache.length && off < m_segment_length && otheroff < other.m_segment_length) {
        for (size_t i = 0; i < other.m_cache.length; i++) {
          m_cache.buffer[off + i] = other.m_cache.buffer[otheroff + i];
        }
        off += cachelength;
        ++other.m_cache;
      }
    }
    m_cache.dirty = true;
#ifdef HAVE_MPI_H
    if (m_replicated && !other.m_replicated) { // replicated <- distributed
      size_t lenseg = ((m_size - 1) / m_mpi_size + 1);
      for (int rank = 0; rank < m_mpi_size; rank++) {
        size_t off = lenseg * rank;
        if (m_cache.io) {
          for (m_cache.ensure(off); m_cache.length && off < lenseg * (rank + 1); off += m_cache.length, ++m_cache)
            MPI_Bcast(m_cache.buffer,
                      m_cache.length,
                      MPI_DOUBLE,
                      rank,
                      m_communicator); // FIXME needs attention for non-double
        } else {
          if (lenseg > 0 && m_size > off)
            MPI_Bcast(&m_cache.buffer[off],
                      std::min(lenseg, m_size - off),
                      MPI_DOUBLE,
                      rank,
                      m_communicator); // FIXME needs attention for non-double
        }
      }
    }
#endif
    return *this;
  }

  /*!
   * @brief Test for equality of two objects
   * @param other
   * @return
   */
  bool operator==(const OutOfCoreArray& other) {
    if (this->m_size != other.m_size) throw std::logic_error("mismatching lengths");
    int diff = 0;
    if (!m_cache.io && !other.m_cache.io) { // std::cout << "both in memory"<<std::endl;
      size_t off = (m_replicated && !other.m_replicated) ? other.m_segment_offset : 0;
      size_t otheroff = (!m_replicated && other.m_replicated) ? m_segment_offset : 0;
      for (size_t i = 0; i < std::min(m_segment_length, other.m_segment_length); i++)
        diff = diff || m_cache.buffer[off + i] != other.m_cache.buffer[otheroff + i];
    } else if (m_cache.io && other.m_cache.io) { // std::cout << "both cached"<<std::endl;
      size_t cachelength = std::min(m_cache.preferred_length, other.m_cache.preferred_length);
      size_t off = 0, otheroff = 0;
      m_cache.move((!m_replicated || other.m_replicated) ? 0 : other.m_segment_offset, cachelength);
      other.m_cache.move((m_replicated || !other.m_replicated) ? 0 : m_segment_offset, cachelength);
      while (m_cache.length && other.m_cache.length && off < m_segment_length && otheroff < other.m_segment_length) {
        for (size_t i = 0; i < m_cache.length; i++)
          diff = diff || m_cache.buffer[off + i] != other.m_cache.buffer[otheroff + i];
        m_cache.dirty = true;
        if (m_cache.io) ++m_cache; else off += cachelength;
        if (other.m_cache.io) ++other.m_cache; else otheroff += cachelength;
      }
    } else if (m_cache.io) { // std::cout<< "source in memory, result cached"<<std::endl;
      size_t cachelength = m_cache.preferred_length;
      size_t off = 0;
      size_t otheroff = (m_replicated == other.m_replicated) ? 0 : m_segment_offset;
      m_cache.move((m_replicated == other.m_replicated) ? 0 : other.m_segment_offset, cachelength);
      other.m_cache.move(0, other.m_segment_length);
      while (m_cache.length && off < m_segment_length && otheroff < other.m_segment_length) {
        for (size_t i = 0; i < m_cache.length; i++)
          diff = diff || m_cache.buffer[off + i] != other.m_cache.buffer[otheroff + i];
        m_cache.dirty = true;
        ++m_cache;
        otheroff += cachelength;
      }
    } else if (other.m_cache.io) {  // std::cout << "source cached, result in memory"<<std::endl;
      size_t cachelength = other.m_cache.preferred_length;
      size_t otheroff = 0;
      size_t off = (m_replicated == other.m_replicated) ? 0 : other.m_segment_offset;
      m_cache.move(0, m_segment_length);
      other.m_cache.move((m_replicated == other.m_replicated) ? 0 : m_segment_offset, cachelength);
      while (other.m_cache.length && off < m_segment_length && otheroff < other.m_segment_length) {
        for (size_t i = 0; i < other.m_cache.length; i++)
          diff = diff || m_cache.buffer[off + i] != other.m_cache.buffer[otheroff + i];
        off += cachelength;
        ++other.m_cache;
      }
    }
#ifdef HAVE_MPI_H
    if (!m_replicated && !other.m_replicated) {
      for (int rank = 0; rank < m_mpi_size; rank++) {
        MPI_Allreduce(&diff, &diff, 1, MPI_INT, MPI_SUM, m_communicator);
      }
    }
#endif
    return diff == 0;
  }

};
template<class scalar>
inline std::ostream& operator<<(std::ostream& os, OutOfCoreArray<scalar> const& obj) { return os << obj.str(); }

}
#endif // OUTOFCOREARRAY_H
