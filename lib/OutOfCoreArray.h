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
        m_segment_length(seg_length()),
        m_buf_size(0),
        m_file(make_file()) {
  }
  /*!
   * @brief Copy constructor
   * @param source
   * @param option
   * @param mpi_communicator
   */
  OutOfCoreArray(const OutOfCoreArray& source, size_t default_buffer_size = 10485760)
      : m_size(source.m_size),
        m_communicator(source.m_communicator), m_mpi_size(mpi_size()), m_mpi_rank(mpi_rank()),
        m_data(nullptr),
        m_sync(source.m_sync),
        m_segment_offset(seg_offset()),
        m_segment_length(seg_length()),
        m_buf_size(m_segment_length > default_buffer_size ? default_buffer_size : m_segment_length),
        m_file(make_file()) {
#ifdef TIMING
    auto start=std::chrono::steady_clock::now();
#endif
    if (!m_sync) {
        source.sync();
        m_sync = true;
    }
    size_t offset = 0;
    while ((m_segment_length-offset)/m_buf_size != 0) {
        m_file.seekp(offset * sizeof(T));
        m_file.write((const char*) source.m_data + offset, m_buf_size * sizeof(T));
        offset += m_buf_size;
    }
    if (offset < m_segment_length) {
        size_t rest = m_segment_length-offset;
        m_file.seekp(offset * sizeof(T));
        m_file.write((const char*) source.m_data + offset, rest * sizeof(T));
    }
    // close file?? re-wind??
    // makes put() obsolete?
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
        m_communicator(mpi_communicator), m_mpi_size(mpi_size()), m_mpi_rank(mpi_rank()),
        m_segment_offset(seg_offset()),
        m_segment_length(seg_length()),
        m_data(buffer+m_segment_offset),
        m_sync(true),
        m_buf_size(0),
        m_file(std::fstream()) {
//    std::cout << "OutOfCoreArray map constructor "<<buffer<<" : "<<&(*this)[0]<<std::endl;
  }

  ~OutOfCoreArray() = default;

  bool synchronised() const { return m_sync; }

 private:
  std::fstream make_file() {
    std::fstream file;
    char* tmpname = strdup("tmpfileXXXXXX");
    {
      auto ifile =
          mkstemp(tmpname); // actually create the file atomically with the name to avoid race resulting in duplicates or conflicts
      if (ifile < 0) throw std::runtime_error(std::string("Cannot open cache file ") + tmpname);
      close(ifile);
    }
    file.open(tmpname, std::ios::out | std::ios::in | std::ios::binary);
    if (!file.is_open()) throw std::runtime_error(std::string("Cannot open cache file ") + tmpname);
    unlink(tmpname);
    free(tmpname);
    return file;
  }

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
  size_t m_buf_size; //!< buffer size
  mutable std::fstream m_file;

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
   * \param buffer The provided data segment
   * \param length Length of data
   * \param offset Offset in the object of buffer[0]
   */
  void put(const T* buffer, size_t length, size_t offset) {
    if (m_data != nullptr) { // data in buffer
      std::copy(buffer, buffer + length, m_data + offset);
      return;
    }
    if (offset > m_segment_offset + m_segment_length or offset + length < m_segment_offset)
      return;
    size_t buffer_offset = 0;
    if (offset < m_segment_offset) {
      buffer_offset = m_segment_offset - offset;
      offset = m_segment_offset;
      length -= m_segment_offset - offset;
    }
    if (offset + length > std::min(m_size, m_segment_offset + m_segment_length))
      length = std::min(m_size,
                        m_segment_offset + m_segment_length) - offset;
    m_file.seekp(offset * sizeof(T));
    m_file.write((const char*) buffer + buffer_offset, length * sizeof(T));
  }

  /*!
   * \brief Read a range of the object data into a provided buffer
   * @param buffer
   * @param length
   * @param offset
   */
  void get(T* buffer, size_t length, size_t offset) const {
    if (m_data != nullptr) { // data in buffer
      std::copy(m_data + offset, m_data + offset + length, buffer);
      return;
    }
    if (offset > m_segment_offset + m_segment_length or offset + length < m_segment_offset)
      return; // through exception??
    size_t buffer_offset = 0;
    if (offset < m_segment_offset) {
      buffer_offset = m_segment_offset - offset;
      offset = m_segment_offset;
      length -= m_segment_offset - buffer_offset;
    }
    if ((offset + length) > (m_segment_offset + m_segment_length))
      length = (m_segment_offset + m_segment_length) - offset;
    m_file.seekp(offset * sizeof(T));
    m_file.read((const char*) buffer + buffer_offset, length * sizeof(T));
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
    //TODO implement or delete the function
//    for (size_t k = 0; k < m_segment_length; k++)
//      os << " " << (*this)[m_segment_offset + k];
    os << std::endl;
    return os.str();
  }
// TODO below here needs implementing
  /*
   * \brief replicate the vectors over the MPI ranks
   * \return
   */
  void sync() {
#ifdef HAVE_MPI_H
    //std::cout <<m_mpi_rank<<"before broadcast this="<<*this<<std::endl;
      if (m_data != nullptr) {
          size_t alpha = m_size/m_mpi_size;
          size_t beta = m_size%m_mpi_size;
          int chunks[m_mpi_size];
          int displs[m_mpi_size];
          for (int i = 0; i < m_mpi_size; i++) {
              chunks[i] = (i < beta) ? alpha+1 : alpha;
              displs[i] = (i < beta) ? chunks[i]*i : chunks[i]*i + beta;
              //            if (m_mpi_rank == 0) std::cout<<"Chunk: "<<chunks[i]<<" Displ: "<<displs[i]<<std::endl;
          }
          MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL,&m_data[0],chunks,displs,MPI_DOUBLE,m_communicator);
          m_sync = true;
      } else {
          throw std::logic_error("Distributed data can not be out of sync");
      }
#endif
  }

  /*!
   * \brief Add a constant times another object to this object
   * \param a The factor to multiply.
   * \param other The object to be added to this.
   * \return
   */
  void axpy(scalar_type a, const OutOfCoreArray<T>& other) {
    if (this->m_size != other.m_size) throw std::logic_error("mismatching lengths"); //m_size -> other.m_size - correct?
    if (this->m_data != nullptr) {
        if (other.m_data == nullptr) {
            T buffer[other.m_buf_size];
            size_t offset = 0;
            while ((m_segment_length-offset)/other.m_buf_size != 0) {
                other.get(&buffer[0],other.m_buf_size,offset);
                for (size_t i = 0; i < other.m_buf_size; i++)
                    m_data[offset+i] += a * buffer[i]; //or use BLAS
                offset += other.m_buf_size;
            }
            if (offset < m_segment_length) {
                size_t cache_length = m_segment_length-offset;
                other.get(&buffer[0],cache_length,offset);
                for (size_t i = 0; i < cache_length; i++)
                    m_data[offset+i] += a * buffer[i]; //or use BLAS
            } // Too lengthy? Wrap in a function?
        } else { // Not expected!? Should check if the same vector => "==" operator needed??
            for (size_t i = 0; i < m_segment_length; i++) // check that of same length?
                m_data[i] += a * other.m_data[i]; //or use BLAS
        }
        m_sync = false;
    } else { // Not expected!?
        throw std::logic_error("Axpy should not be called for a vector written to disk");
    }
  }

  /*!
    * \brief Add a constant times a sparse vector to this object
    * \param a The factor to multiply.
    * \param other The object to be added to this.
    * \return
    */
  void axpy(scalar_type a, const std::map<size_t, T>& other) {
    if (this->m_data == nullptr) throw std::logic_error("Axpy should not be called for a vector written to disk");
    for (const auto& o: other)
      if (o.first >= m_segment_offset && o.first < m_segment_offset + m_segment_length)
        this->m_data[o.first-m_segment_offset] += a * o.second;
    m_sync = false;
  }

  /*!
   * \brief Scalar product of two objects.
   * \param other The object to be contracted with this.
   * \return
   */
  scalar_type dot(const OutOfCoreArray<T>& other) const {
    if (this->m_size != other.m_size) throw std::logic_error("mismatching lengths");
    scalar_type result = 0;
    if (this == &other) {
        if (this->m_data != nullptr) {
            for (size_t i = 0; i < m_segment_length; i++)
                result += m_data[i]*m_data[i]; //or use BLAS
        } else {
            T buffer[m_buf_size];
            size_t offset = 0;
            while ((m_segment_length-offset)/m_buf_size != 0) {
                get(&buffer[0],m_buf_size,offset);
                for (size_t i = 0; i < m_buf_size; i++)
                    result += buffer[i] * buffer[i]; //or use BLAS
                offset += m_buf_size;
            }
            if (offset < m_segment_length) {
                size_t cache_length = m_segment_length-offset;
                get(&buffer[0],cache_length,offset);
                for (size_t i = 0; i < cache_length; i++)
                    result += buffer[i] * buffer[i]; //or use BLAS
            }
        }
    } else {
        if (this->m_data != nullptr) {
            if (other.m_data != nullptr) {
                for (size_t i = 0; i < m_segment_length; i++) // check m_segment_length == other.m_segment_length?
                    result += m_data[i]*other.m_data[i]; //or use BLAS
            } else {
                T buffer[other.m_buf_size];
                size_t offset = 0;
                while ((m_segment_length-offset)/other.m_buf_size != 0) {
                    get(&buffer[0],other.m_buf_size,offset);
                    for (size_t i = 0; i < other.m_buf_size; i++)
                        result += m_data[offset+i] * buffer[i]; //or use BLAS
                    offset += other.m_buf_size;
                }
                if (offset < m_segment_length) {
                    size_t cache_length = m_segment_length-offset;
                    get(&buffer[0],cache_length,offset);
                    for (size_t i = 0; i < cache_length; i++)
                        result += m_data[offset+i] * buffer[i]; //or use BLAS
                }
            }
        } else {
            if (other.m_data != nullptr) {
                T buffer[m_buf_size];
                size_t offset = 0;
                while ((m_segment_length-offset)/m_buf_size != 0) {
                    get(&buffer[0],m_buf_size,offset);
                    for (size_t i = 0; i < m_buf_size; i++)
                        result += buffer[i] * other.m_data[offset+i]; //or use BLAS
                    offset += m_buf_size;
                }
                if (offset < m_segment_length) {
                    size_t cache_length = m_segment_length-offset;
                    get(&buffer[0],cache_length,offset);
                    for (size_t i = 0; i < cache_length; i++)
                        result += buffer[i] * other.m_data[offset+i]; //or use BLAS
                }
            } else {
                T lbuffer[m_buf_size], rbuffer[other.m_buf_size]; //check same length??
                size_t offset = 0;
                while ((m_segment_length-offset)/m_buf_size != 0) {
                    get(&lbuffer[0],m_buf_size,offset);
                    get(&rbuffer[0],other.m_buf_size,offset);
                    for (size_t i = 0; i < m_buf_size; i++)
                        result += lbuffer[i] * rbuffer[i]; //or use BLAS
                    offset += m_buf_size;
                }
                if (offset < m_segment_length) {
                    size_t rest = m_segment_length-offset;
                    get(&lbuffer[0],rest,offset);
                    get(&rbuffer[0],rest,offset);
                    for (size_t i = 0; i < rest; i++)
                        result += lbuffer[i] * rbuffer[i]; //or use BLAS
                }
            }
        }
    }
#ifdef HAVE_MPI_H
    //std::cout <<m_mpi_rank<<" dot result before reduce="<<result<<std::endl;
    double resultLocal = result;
    double resultGlobal = result;
    MPI_Allreduce(&resultLocal, &resultGlobal, 1, MPI_DOUBLE, MPI_SUM, m_communicator);
    result = resultGlobal;
    // std::cout <<m_mpi_rank<<" dot result after reduce="<<result<<std::endl;
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
    if (this->m_data != 0) {
        for (const auto& o: other)
            if (o.first >= m_segment_offset && o.first < m_segment_offset + m_segment_length)
                result += o.second * this->m_data[o.first-m_segment_offset];
    } else {
        T buffer[m_buf_size];
        size_t offset = 0;
        while ((m_segment_length-offset)/m_buf_size != 0) {
            get(&buffer[0],m_buf_size,offset);
            for (const auto& o: other)
                if (o.first >= m_segment_offset+offset && o.first < m_segment_offset+offset + m_buf_size)
                    result += o.second * buffer[o.first-m_segment_offset-offset]; //or use BLAS
            offset += m_buf_size;
        }
        if (offset < m_segment_length) {
            size_t rest = m_segment_length-offset;
            get(&buffer[0],rest,offset);
            for (const auto& o: other)
                if (o.first >= m_segment_offset+offset && o.first < m_segment_offset+offset + rest)
                    result += o.second * buffer[o.first-m_segment_offset-offset]; //or use BLAS
        }
    }
#ifdef HAVE_MPI_H
    double resultLocal = result;
    MPI_Allreduce(&resultLocal,
            &result,
            1,
            MPI_DOUBLE,
            MPI_SUM,
            m_communicator); // FIXME needs attention for non-double
#endif
    return result;
  }

  /*!
     * \brief scal Scale the object by a factor.
     * \param a The factor to scale by. If a is zero, then the current contents of the object are ignored.
     */
  void scal(scalar_type a) {
      if (this->m_data == nullptr) throw std::logic_error("Scal() should not be called for a vector written to disk");
      if (a != 0) {
          for (size_t i = 0; i < m_segment_length; i++)
              this->m_data[i] *= a;
      } else {
          for (size_t i = 0; i < m_segment_length; i++)
              this->m_data[i] = 0;
      }
      m_sync = false;
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
