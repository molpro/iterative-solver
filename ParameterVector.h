#ifndef PARAMETERVECTOR_H
#define PARAMETERVECTOR_H

#include <cstddef>
#include <vector>
#include <string>
#include <sstream>
#include <iostream>

namespace IterativeSolver {
  typedef double ParameterScalar;

/*!
   * \brief A class to hold expansion vectors and residual vectors for use by IterativeSolver classes.
   *
   * The base implementation provided here just stores the data internally as a one-dimensional, but an inheriting class could adopt
   * a structured layout, and could store externally and/or distributed across processes, so long as all of the public methods function correctly.
   * The operator<<(), operator[]() and size() methods provided in this base class refer to the std::vector that is used in the simple base implementation, and should not normally be used in production,
   * so that derivative classes are free to avoid defining them. They are typically used in debugging.
   */
  class ParameterVector
  {
  public:
      /*!
     * \brief Construct an object without any data.
     */
    ParameterVector(size_t length=0);
    ParameterVector(const ParameterVector &source);
    virtual ~ParameterVector();
    /*!
     * \brief Add a constant times another object to this object
     * \param a The factor to multiply.
     * \param other The object to be added to this.
     * \return
     */
    virtual void axpy(ParameterScalar a, const ParameterVector& other);
    /*!
     * \brief Scalar product of two objects.
     * \param other The object to be contracted with this.
     * \return
     */
    virtual ParameterScalar operator*(const ParameterVector& other) const;
    /*!
     * \brief Set the contents of the object to zero.
     */
    virtual void zero();
    /*!
     * \brief Copy from one object to another, adjusting size if needed.
     * \param other The source of data.
     * \return
     */
    virtual ParameterVector& operator=(const ParameterVector& other);
      /*!
     * \brief Record the co/contra-variance status of the object
     * \param variance
     * - -1: covariant vector
     * - +1: contravariant vector
     * - 0: self-dual vector
     * The class is expected to check that appropriate combinations of vectors are provided in methods that perform linear algebra functions.
     */
    void setVariance(int variance=0) {m_variance=variance;}
    int variance() {return m_variance;}
    virtual ParameterVector* clone() const;
    virtual std::string str() const { std::ostringstream os; os << "ParameterVector object:";
                                      for (size_t k=0; k<size(); k++)
                                          os <<" "<< (*this)[k];
                                      os << std::endl;
                                       return os.str();}
  protected:
  private:
    int m_variance;
    /*!
     * \brief For a simple implementation, just use an STL vector. Classes that inherit are free to do things differently.
     */
    std::vector<ParameterScalar> m_buffer;
  public:
    virtual ParameterScalar& operator[](size_t pos);
    const virtual ParameterScalar& operator[](size_t pos) const;
    virtual size_t size() const;
  };

 inline std::ostream& operator<<(std::ostream& os, ParameterVector const& pv) {
     os << pv.str();
     return os;
 }

 /*!
   * \brief A container for a collection of ParameterVector objects
   */
  class ParameterVectorSet
    {
    public:
      ParameterVectorSet() {}
      ParameterVectorSet(const ParameterVectorSet& source)
  {
    for (size_t k=0; k<source.size(); k++) {
        push_back_clone(source.pvs[k]);
        active[k] = source.active[k];
      }
  }

      ~ParameterVectorSet() { while(pvs.size()>0) pvs.pop_back();}
  	  size_t size() const { return this->pvs.size();}
      ParameterVector& operator[](size_t pos) { return *pvs[pos];}
      const ParameterVector& operator[](size_t pos) const { return *pvs[pos];}
      ParameterVector& front() { return *(pvs.front());}
      const ParameterVector& front() const { return *(pvs.front());}
      ParameterVector& back() { return *(pvs.back());}
      const ParameterVector& back() const { return *(pvs.back());}
      void push_back(ParameterVector* val)
{
          pvs.push_back(val);
          owned.push_back(false);
          active.push_back(true);
      }

      void push_back_clone(ParameterVector* val)
{
          std::cout << "push_back_clone val="<<*val<<std::endl;
          std::cout << "push_back_clone val->clone()="<<*val->clone()<<std::endl;
          pvs.push_back(val->clone());
          std::cout << "push_back_clone pvs.back()="<<*pvs.back()<<std::endl;
          owned.push_back(true);
          active.push_back(true);
      }
      void pop_back() { if (pvs.size() <=0) return; if (owned.back()) delete pvs.back(); pvs.pop_back(); active.pop_back(); owned.pop_back();}
      void resize(size_t length) { pvs.resize(length); active.resize(length);}
      ParameterVectorSet& operator=(const ParameterVectorSet& source)
      {
          while(pvs.size()>0) pvs.pop_back();
          for (size_t k=0; k<source.size(); k++) {
              push_back_clone(source.pvs[k]);
              active[k] = source.active[k];
          }
          return *this;
      }
    /*!
     * \brief Add a constant times another object to this object
     * \param a The factor to multiply.
     * \param other The object to be added to this.
     * \return
     */
    void axpy(ParameterScalar a, const ParameterVectorSet& other) {
      for (size_t k=0; k<size(); k++) this->pvs[k]->axpy(a,other[k]);
    }
    /*!
     * \brief Set the contents of the object to zero.
     */
    void zero() { for (size_t k=0; k<size(); k++) this->pvs[k]->zero(); }

      /*!
       * \brief A place to record whether each member of the set is meant to be considered active
       */
    std::vector<bool> active;
    private:
    std::vector<bool> owned; ///< whether the ParameterVector objects pointed to are owned by this container (and therefore destroyed in the destructor)
    std::vector<ParameterVector*> pvs; // contain pointers to support polymorphism
  };

  inline std::ostream& operator<<(std::ostream& os, const ParameterVectorSet& pvs) {
    for (size_t k=0; k<pvs.size(); k++) {
        os << " pvs.active[k] "<<pvs.active[k]<<std::endl;
      if (pvs.active[k])
        os << (pvs[k]);
    }
    return os;
  }

}

#endif // PARAMETERVECTOR_H
