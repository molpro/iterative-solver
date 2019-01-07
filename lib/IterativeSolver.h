#ifndef ITERATIVESOLVER_H
#define ITERATIVESOLVER_H
#ifndef _GNU_SOURCE
#define _GNU_SOURCE 1
#endif

#include <string>
#include <iostream>
#include <vector>
#include <fstream>
#include <map>
#include <algorithm>
#include <stdexcept>
#include <numeric>
#include <Eigen/Dense>
#include <cmath>
#include <complex>

#ifndef LINEARALGEBRA_OFFLINE
#define LINEARALGEBRA_OFFLINE 0x01
#endif
#ifndef LINEARALGEBRA_DISTRIBUTED
#define LINEARALGEBRA_DISTRIBUTED 0x02
#endif

#undef isnan
#undef isinf
#ifdef MOLPRO
extern std::ostream &xout;
#else
#define xout std::cout
#endif

/*!
 * @brief Contains IterativeSolver and supporting classes, including an abstract opaque vector
 */
namespace LinearAlgebra {
typedef std::map<std::string, std::string> optionMap;

/*!
* \brief A base class for iterative solvers for linear and non-linear equations, and linear eigensystems.
*
* The calling program should set up its own iterative loop, and in each iteration
* - calculate
* the action of the matrix on the current expansion vector (linear), or the actual
* residual (non-linear)
* - make a call to addVector() which takes the current and previous parameters and proposes
* an improved estimate, and the best estimate of the residual vector.
* - calculate a new solution (non-linear) or expansion vector (linear) by implementing
* appropriate preconditioning on the residual
* -  make a call to endIteration()
*
* Classes that derive from this will, in the simplest case, need to provide just the solveReducedProblem() method that governs how the parameter and residual vectors from successive iterations
* should be combined to form an optimum solution with minimal residual.
*
* The underlying vector spaces are accessed through instances of the vectorSet<scalar> class (or derivatives). These consist of a set of pointers to vector<scalar> objects, which are the vectors themselves; the vectorSet<scalar>
* object has dimension greater than one in, for example, multi-root diagonalisations where the residual is calculated simultaneously for a number of vectors.
* Two instances of vectorSet<scalar> have to be passed to iterate() or solve(),
* and these are used to construct solutions and residuals;
* this class also creates unlimited additional instances of vectorSet<scalar>,
* and in memory-sensitive environments, consideration might be given to implementing a derivative of vectorSet<scalar> where the data is resident in external storage.
* The additional instances are created with hints that they may be stored offline passed to
* the copy constructor.
* The use of pointers in vectorSet<scalar> supports polymorphism: the user is free to
* present instances of classes deriving from vector<scalar>, and thereby implement
* any special storage arrangements desired.
*/
template<class T>
class IterativeSolver {
 public:
  IterativeSolver(
  ) :
      m_Pvectors(0),
      m_verbosity(0),
      m_thresh(1e-8),
      m_actions(0),
      m_maxIterations(1000),
      m_minIterations(0),
      m_orthogonalize(true),
      m_linear(false),
      m_hermitian(false),
      m_roots(0),
      m_rspt(false),
      m_options(optionMap()),
      m_error(0),
      m_worst(0),
      m_date(0),
      m_subspaceMatrixResRes(false),
      m_residual_eigen(false),
      m_residual_rhs(false),
      m_residuals(),
      m_solutions(),
      m_others(),
      m_rhs(),
      m_dateOfBirth(),
      m_lastVectorIndex(0),
      m_updateShift(0),
      m_interpolation(),
      m_dimension(0),
      m_iterations(0),
      m_singularity_threshold(1e-16),
      m_added_vectors(0),
      m_augmented_hessian(0),
      m_svdThreshold(1e-15),
      m_maxQ(std::max(m_roots, size_t(16))) {}

  virtual ~IterativeSolver() = default;

 protected:
  using value_type = typename T::value_type; ///< The underlying type of elements of vectors
  using vectorSet = typename std::vector<T>; ///< Container of vectors
  using vectorSetP = typename std::vector<std::vector<value_type> >; ///<Container of P-space parameters
 public:
  using scalar_type = decltype(std::declval<T>().dot(std::declval<const T&>())); ///< The type of scalar products of vectors
  /*!
   * \brief Take, typically, a current solution and residual, and return new solution.
   * In the context of Lanczos-like linear methods, the input will be a current expansion vector and the result of
   * acting on it with the matrix, and the output will be a new expansion vector.
   * For non-linear equations, the input will be the current solution and residual, and the output the interpolated solution and residual.
   * \param parameters On input, the current solution or expansion vector. On exit, the interpolated solution vector.
   * \param action On input, the residual for parameters (non-linear), or action of matrix on parameters (linear). On exit, the expected (non-linear) or actual (linear) residual of the interpolated parameters.
   * \param parametersP On exit, the interpolated solution projected onto the P space.
   * \param other Optional additional vectors that should be interpolated like the residual.
   * corresponding element of other contains data to be used.
   */
  void addVector(vectorSet& parameters,
                 vectorSet& action,
                 vectorSetP& parametersP,
                 vectorSet& other) {
//   if (m_rhs.size())
//    xout << "addVector entry m_rhs.back()="<<this->m_rhs.back()<<std::endl;
    m_active.resize(parameters.size(), true);
    if (m_roots < 1) m_roots = parameters.size(); // number of roots defaults to size of parameters
    if (!m_orthogonalize && m_roots > m_maxQ) m_maxQ = m_roots;
    assert(parameters.size() == action.size());
    m_iterations++;
    m_added_vectors = 0;
    for (size_t k = 0; k < action.size(); k++) if (m_active[k]) m_added_vectors++;
    m_actions += m_added_vectors;
    m_lastVectorIndex = addVectorSet(parameters, action, other)
        - 1; // derivative classes might eventually store the vectors on top of previous ones, in which case they will need to store the position here for later calculation of iteration step
//   xout << "set lastVectorIndex=addVectorSet-1="<<m_lastVectorIndex<<std::endl;
    scalar_type weight =
        this->m_options.count("weight") ? (
            std::strtod(this->m_options.find("weight")->second.c_str(), NULL)) : 1.0;
    this->m_Weights.push_back(weight); // TODO not conformant - add style
    buildSubspace();
    solveReducedProblem();
    doInterpolation(parameters, action, parametersP, other);
  }

  void addVector(vectorSet& parameters, vectorSet& action) {
    vectorSetP parametersP;
    addVector(parameters, action, parametersP);
    return;
  }

  void
  addVector(vectorSet& parameters, vectorSet& action, vectorSetP& parametersP) {
    vectorSet other;
    addVector(parameters, action, parametersP, other);
    return;
  }

 public:
  /*!
   * \brief Specify a P-space vector as a sparse combination of parameters. The container holds a number of segments,
   * each characterised by an offset in the full space, and a vector of coefficients starting at that offset.
   */
  using Pvector = std::map<size_t, scalar_type>;

  /*!
   * \brief Add P-space vectors to the expansion set for linear methods.
   * \param Pvectors the vectors to add
   * \param PP Matrix projected onto the existing+new, new P space. It should be provided as a
   * 1-dimensional array, with the existing+new index running fastest.
   * \param parameters On exit, the interpolated solution vector.
   * \param action  On exit, the  residual of the interpolated Q parameters.
   * The contribution from the new, and any existing, P parameters is missing, and should be added in subsequently.
   * \param parametersP On exit, the interpolated solution projected onto the P space.
   * \param other On exit, interpolation of the other vectors
   */
  void addP(std::vector<Pvector> Pvectors, const scalar_type* PP,
            vectorSet& parameters,
            vectorSet& action,
            vectorSetP& parametersP,
            vectorSet& other) {
    auto oldss = m_subspaceMatrix.rows();
    m_active.resize(parameters.size(), true);
//    xout << "oldss " << oldss << ", Pvectors,size() " << Pvectors.size() << std::endl;
    m_subspaceMatrix.conservativeResize(oldss + Pvectors.size(), oldss + Pvectors.size());
    m_subspaceOverlap.conservativeResize(oldss + Pvectors.size(), oldss + Pvectors.size());
    Eigen::Index oldNP = m_PQMatrix.rows();
    Eigen::Index newNP = oldNP + Pvectors.size();
    assert(newNP + m_QQMatrix.rows() == m_subspaceOverlap.rows());
    parametersP.resize(newNP);
    m_PQMatrix.conservativeResize(newNP, m_QQMatrix.rows());
    m_PQOverlap.conservativeResize(newNP, m_QQMatrix.rows());
    m_subspaceRHS.conservativeResize(newNP, m_rhs.size());
    size_t offset = 0;
//    xout << "addP PP\n";
//    for (auto i = 0; i < Pvectors.size(); i++)
//      for (auto j = 0; j < newNP; j++)
//        xout << i << " " << j << " " << PP[offset++]<<std::endl;
//    offset = 0;
    for (size_t n = 0; n < Pvectors.size(); n++)
      m_Pvectors.push_back(Pvectors[n]);
    for (size_t n = 0; n < Pvectors.size(); n++) {
      for (Eigen::Index i = 0; i < newNP; i++) {
//        xout << "offset " << offset << std::endl;
//        xout << "PP " << PP[offset] << std::endl;
        m_subspaceMatrix(oldNP + n, i) = m_subspaceMatrix(i, oldNP + n) = PP[offset++];
        double overlap = 0;
//        xout << "n=" << n << ", i=" << i << std::endl;
        for (const auto& p : Pvectors[n]) {
//          xout << "addP Pvector=" << p.first << " : " << p.second << " " << m_Pvectors[i].count(p.first) << std::endl;
          if (m_Pvectors[i].count(p.first))
            overlap += p.second * m_Pvectors[i][p.first];
        }
//        xout << "overlap: " << overlap << std::endl;
//        xout << "oldNP+n=" << oldNP + n << ", i=" << i << ", dimensions: " << m_subspaceOverlap.rows() << ", "
//             << m_subspaceOverlap.cols() << std::endl;
        m_subspaceOverlap(oldNP + n, i) = m_subspaceOverlap(i, oldNP + n) = overlap;
//        xout << "stored " << m_subspaceOverlap(oldNP + n, i) << std::endl;
      }
    }
    size_t l = 0;
    for (size_t ll = 0; ll < m_solutions.size(); ll++) {
      for (size_t lll = 0; lll < m_solutions[ll].size(); lll++) {
        if (m_vector_active[ll][lll]) {
          for (size_t n = 0; n < Pvectors.size(); n++) {
            m_PQMatrix(oldNP + n, l) = m_residuals[ll][lll].dot(Pvectors[n]);
            m_PQOverlap(oldNP + n, l) = m_solutions[ll][lll].dot(Pvectors[n]);
          }
          l++;
        }
      }
    }
    for (size_t n = 0; n < Pvectors.size(); n++) {
      for (size_t l = 0; l < m_rhs.size(); l++) {
        m_subspaceRHS(oldNP + n, l) = m_rhs[l].dot(Pvectors[n]);
      }
    }
    buildSubspace();
    solveReducedProblem();
    doInterpolation(parameters, action, parametersP, other);
//    for (auto ll = 0; ll < parameters.size(); ll++)
//      xout << " after doInterpolation, parameters=" << parameters[ll]<< std::endl;
//      xout << " after doInterpolation, g.w=" << parameters[ll]->dot(*action[ll]) << std::endl;
  }

  void addP(std::vector<Pvector> Pvectors, const scalar_type* PP, vectorSet& parameters, vectorSet& action,
            vectorSetP& parametersP) {
    vectorSet other;
    return addP(Pvectors, PP, parameters, action, parametersP, other);
  }

  /*!
   * \brief Remove completely the whole P space
   */
  void clearP() {
    m_subspaceMatrix.conservativeResize(m_QQMatrix.rows(), m_QQMatrix.rows());
    m_subspaceOverlap.conservativeResize(m_QQMatrix.rows(), m_QQMatrix.rows());
    m_PQMatrix.conservativeResize(0, m_QQMatrix.rows());
    m_PQOverlap.conservativeResize(0, m_QQMatrix.rows());
    m_subspaceRHS.conservativeResize(0, m_rhs.size());
    m_Pvectors.clear();
  }

  /*!
     * \brief Take the updated solution vector set, and adjust it if necessary so that it becomes the vector to
     * be used in the next iteration; this is done only in the case of linear solvers where the orthogonalize option is set.
     * Also calculate the degree of convergence, and write progress to xout.
     * \param solution The current solution, after interpolation and updating with the preconditioned residual.
     * \param residual The residual after interpolation.
     * \return true if convergence reached for all roots
     */
  bool endIteration(vectorSet& solution, const vectorSet& residual) {
    calculateErrors(solution, residual);
    if (m_error >= m_thresh) adjustUpdate(solution);
    report();
    return m_error < m_thresh;
  }

  /*!
   * @brief Whether the expansion vector for a particular root is still active, ie not yet converged, and residual has meaning
   * @param root
   * @return
   */
  bool active(int root) { return m_active[root]; }
  std::vector<bool> active() { return m_active.empty() ? std::vector<bool>(1000,true) : m_active; }

  /*!
   * \brief Get the solver's suggestion of which degrees of freedom would be best
   * to add to the P-space.
   * \param solution Current solution
   * \param residual Current residual
   * \param maximumNumber Suggest no more than this number
   * \param threshold Suggest only axes for which the current residual and update
   * indicate an energy improvement in the next iteration of this amount or more.
   * \return
   */
  std::vector<size_t> suggestP(const vectorSet& solution,
                               const vectorSet& residual,
                               const size_t maximumNumber = 1000,
                               const scalar_type threshold = 0) {
    std::map<size_t, scalar_type> result;
    for (size_t kkk = 0; kkk < solution.size(); kkk++) {
//    xout << "suggestP kkk "<<kkk<<" active "<<solution.m_vector_active[kkk]<<maximumNumber<<std::endl;
      if (m_active[kkk]) {
        std::vector<size_t> indices;
        std::vector<scalar_type> values;
        std::tie(indices, values) = solution[kkk].select(residual[kkk], maximumNumber, threshold);
//     xout <<"indices.size()="<<indices.size()<<std::endl;
//     for (auto k=0; k<indices.size(); k++) xout << "select "<< indices[k] <<" : "<<values[k]<<std::endl;
        for (size_t i = 0; i < indices.size(); i++)
          if (result.count(indices[i]))
            result[indices[i]] = std::max(result[indices[i]], values[i]);
          else
            result[indices[i]] = values[i];
      }
    }
    // sort and select
//   for (const auto& kv : result) xout << "result: " << kv.first << " : " <<kv.second<<std::endl;
    std::multimap<scalar_type, size_t, std::greater<scalar_type>> inverseResult;
    for (const auto& kv : result) inverseResult.insert(std::pair<scalar_type, size_t>(kv.second, kv.first));
//   for (const auto& kv : inverseResult) xout << "inverseResult: " << kv.first << " : " <<kv.second<<std::endl;
    std::vector<size_t> indices;
//   std::vector<T> values;
    size_t k = 0;
    for (auto p = inverseResult.cbegin(); p != inverseResult.cend() && k < maximumNumber; k++) {
      indices.push_back(p->second);// values.push_back(p->first);
      ++p;
    }
//   for (auto k=0; k<indices.size(); k++) xout << "suggest P "<< indices[k] <<" : "<<values[k]<<std::endl;
    return indices;
  }

  /*!
   * \brief Set convergence threshold
   */
  void setThresholds(scalar_type thresh) { m_thresh = thresh; }

  unsigned int iterations() { return m_iterations; } //!< How many iterations have occurred

  std::vector<scalar_type> eigenvalues() ///< The calculated eigenvalues of m_subspaceMatrix
  {
    std::vector<scalar_type> result;
    for (size_t root = 0; root < (size_t) m_roots && root < (size_t) m_subspaceEigenvalues.rows(); root++)
      result.push_back(m_subspaceEigenvalues[root].real());
    return result;
  }

  std::vector<scalar_type> errors() const { return m_errors; } //!< Error at last iteration

  size_t dimensionP() const { return (size_t) m_PQMatrix.rows(); } //!< Size of P space

 protected:
  Eigen::Matrix<scalar_type, Eigen::Dynamic, Eigen::Dynamic> m_PQMatrix, m_PQOverlap; //!< The PQ block of the matrix
  std::vector<Pvector> m_Pvectors;
  std::vector<bool> m_active; ///< whether each expansion vector is currently active
 public:
  int
      m_verbosity; //!< How much to print. Zero means nothing; One results in a single progress-report line printed each iteration.
  scalar_type
      m_thresh; //!< If predicted residual . solution is less than this, converged, irrespective of cthresh and gthresh.
  unsigned int m_actions; //!< number of action vectors provided
  unsigned int m_maxIterations; //!< Maximum number of iterations in solve()
  unsigned int m_minIterations; //!< Minimum number of iterations in solve()
  bool
      m_orthogonalize; ///< Whether or not to orthogonalize the result of update() to all previous expansion vectors (appropriate only for linear methods).
  bool m_linear; ///< Whether residuals are linear functions of the corresponding expansion vectors.
  bool m_hermitian; ///< Whether residuals can be assumed to be the action of an underlying self-adjoint operator.
  size_t
      m_roots; ///< How many roots to calculate / equations to solve (defaults to size of solution and residual vectors)
  bool m_rspt;
 public:
  optionMap m_options; ///< A string of options to be interpreted by solveReducedProblem().
  ///< Possibilities include
  ///< - m_options["convergence"]=="energy" (the default), meaning that m_errors() returns the predicted eigenvalue change in the next iteration, ie the scalar product of the step and the residual
  ///< - m_options["convergence"]=="step": m_errors() returns the norm of the step in the solution
  ///< - m_options["convergence"]=="residual": m_errors() returns the norm of the residual vector

 private:
  void adjustUpdate(vectorSet& solution) {
//         xout << "m_errors[0] "<<m_errors[0]<<", m_thresh "<<m_thresh<<std::endl;
    for (size_t k = 0; k < solution.size(); k++)
      m_active[k] = (m_errors[k] >= m_thresh || m_minIterations > m_iterations);
//    xout << "active "<<active[0]<<std::endl;
//          xout << "IterativeSolverBase::adjustUpdate solution before orthogonalization: "<<solution[0]<<std::endl;
    if (m_orthogonalize) {
//          xout << "IterativeSolverBase::adjustUpdate solution before orthogonalization: "<<solution[0]<<std::endl;
      for (auto rep = 0; rep < 2; rep++)
        for (size_t kkk = 0; kkk < solution.size(); kkk++) {
          if (m_active[kkk]) {
            for (size_t i = 0; i < m_Pvectors.size(); i++) {
              const auto& p = m_Pvectors[i];
              scalar_type s = -solution[kkk].dot(p) / m_subspaceOverlap(i, i);
              solution[kkk].axpy(s, p);
            }
            for (size_t ll = 0; ll < m_solutions.size(); ll++) {
              for (size_t lll = 0; lll < m_solutions[ll].size(); lll++) {
                if (m_vector_active[ll][lll]) {
                  scalar_type s =
                      -(m_solutions[ll][lll].dot(solution[kkk])) / (m_solutions[ll][lll].dot(m_solutions[ll][lll]));
                  solution[kkk].axpy(s, m_solutions[ll][lll]);
                }
              }
            }
            for (size_t lll = 0; lll < kkk; lll++) {
              if (m_active[lll]) {
                scalar_type s = solution[lll].dot(solution[kkk]);
                solution[kkk].axpy(-s, solution[lll]);
              }
            }
            scalar_type s = solution[kkk].dot(solution[kkk]);
            if (s <= 0)
              m_active[kkk] = false;
            else
              solution[kkk].scal(1 / std::sqrt(s));
          }
        }
//          xout << "IterativeSolverBase::adjustUpdate solution after orthogonalization: "<<solution[0]<<std::endl;
    }
  }

 protected:

  virtual void solveReducedProblem() = 0;

  virtual void report() {
    if (m_verbosity > 0) {
      xout << "iteration " << iterations();
      if (this->m_roots > 1)
        xout << ", error[" << m_worst << "] = ";
      else
        xout << ", error = ";
      xout << m_error << std::endl;
    }
  }

  void buildSubspace() {
//   xout << "buildSubspace"<<std::endl;
    const size_t nP = m_Pvectors.size();
    const size_t nQ = m_QQMatrix.rows();
    const size_t n = nP + nQ;
    m_subspaceMatrix.conservativeResize(n, n);
    m_subspaceOverlap.conservativeResize(n, n);
    for (size_t i = 0; i < nQ; i++) {
      for (size_t j = 0; j < nQ; j++) {
        m_subspaceMatrix(nP + j, nP + i) = m_QQMatrix(j, i);
        m_subspaceOverlap(nP + j, nP + i) = m_QQOverlap(j, i);
      }
      for (size_t j = 0; j < nP; j++) {
        m_subspaceMatrix(j, nP + i) = m_subspaceMatrix(nP + i, j) = m_PQMatrix(j, i);
        m_subspaceOverlap(j, nP + i) = m_subspaceOverlap(nP + i, j) = m_PQOverlap(j, i);
      }
    }
    {// look for linear dependency
      if (m_verbosity > 1) xout << "Subspace matrix" << std::endl << this->m_subspaceMatrix << std::endl;
      if (m_verbosity > 1) xout << "Subspace overlap" << std::endl << this->m_subspaceOverlap << std::endl;
//    Eigen::EigenSolver<Eigen::Matrix<scalar_type, Eigen::Dynamic, Eigen::Dynamic> > ss(m_subspaceMatrix);
//    xout << "eigenvalues "<<ss.eigenvalues()<<std::endl;
      const auto& singularTester = m_residual_eigen ? m_subspaceOverlap : m_subspaceMatrix;
      Eigen::JacobiSVD<Eigen::Matrix<scalar_type, Eigen::Dynamic, Eigen::Dynamic>>
          svd(singularTester, Eigen::ComputeThinU | Eigen::ComputeThinV);
//    xout << "singular values: "<<svd.singularValues().transpose()<<std::endl;
//    xout << "U: "<<svd.matrixU()<<std::endl;
//    xout << "V: "<<svd.matrixV()<<std::endl;
//    auto tester = [](Eigen::Index i) { return std::fabs(svd.matrixV(nP+i,k))};
      const auto& k = m_subspaceMatrix.rows() - 1;
//      xout << "m_singularity_threshold "<<m_singularity_threshold<<std::endl;
//      xout << "svd.singularValues()(0) "<<svd.singularValues()(0)<<std::endl;
//      xout << "svd.singularValues()(k) "<<svd.singularValues()(k)<<std::endl;
      if (svd.singularValues()(k) < m_singularity_threshold * svd.singularValues()(0)
          || (m_linear && !m_orthogonalize
              && nQ > m_maxQ)) { // condition number assuming singular values are strictly descending
        Eigen::Index imax = 0;
        for (Eigen::Index i = 0; i < static_cast<Eigen::Index>(nQ - 1);
             i++) { // never consider the very last Q vector
//      xout << "consider " <<i<<": "<<svd.matrixV()(nP+i,k)<<std::endl;
//      if (std::fabs(svd.matrixV()(nP + i, k)) > std::fabs(svd.matrixV()(nP + imax, k))) imax = i;
          if (
              std::fabs(svd.matrixV()(nP + i, k) * singularTester(nP + i, nP + i))
                  >
                      std::fabs(svd.matrixV()(nP + imax, k) * singularTester(nP + imax, nP + imax))
              )
            imax = i;
        }
//     imax=0;
        if (m_verbosity > 0) {
          xout << " removing singular value " << svd.singularValues()(k) / svd.singularValues()(0)
               << " by deleting redundant expansion vector " << imax << ";  new subspace size " << nQ - 1;
          if (nP > 0) xout << "Q + " << nP << "P";
          xout << std::endl;
        }
        if (m_verbosity > 1) xout << "  SVD right matrix column: " << svd.matrixV().col(k).transpose() << std::endl;
        deleteVector(imax);
        return; // deleteVector calls this function to rebuild the subspace
      }
    }
    if (m_verbosity > 3 && m_PQMatrix.rows() > 0) xout << "PQ matrix" << std::endl << this->m_PQMatrix << std::endl;
    if (m_verbosity > 3 && m_PQMatrix.rows() > 0) xout << "PQ overlap" << std::endl << this->m_PQOverlap << std::endl;
    if (m_verbosity > 3 && m_PQMatrix.rows() > 0) xout << "QQ matrix" << std::endl << this->m_QQMatrix << std::endl;
    if (m_verbosity > 3 && m_PQMatrix.rows() > 0) xout << "QQ overlap" << std::endl << this->m_QQOverlap << std::endl;
    if (m_verbosity > 1) xout << "Subspace matrix" << std::endl << this->m_subspaceMatrix << std::endl;
    if (m_verbosity > 1) xout << "Subspace overlap" << std::endl << this->m_subspaceOverlap << std::endl;
  }

 protected:
  void diagonalizeSubspaceMatrix() {
    auto kept = m_subspaceMatrix.rows();

    Eigen::Matrix<scalar_type, Eigen::Dynamic, Eigen::Dynamic> H = m_subspaceMatrix.block(0, 0, kept, kept);
    Eigen::Matrix<scalar_type, Eigen::Dynamic, Eigen::Dynamic> S = m_subspaceOverlap.block(0, 0, kept, kept);
//   Eigen::GeneralizedEigenSolver<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>> s(H, S);
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(S, Eigen::ComputeThinU | Eigen::ComputeThinV);
    svd.setThreshold(m_svdThreshold);
//   xout << "singular values of overlap "<<svd.singularValues().transpose()<<std::endl;
//   auto Hbar = svd.solve(H);
    if (m_verbosity > 1 && svd.rank() < S.cols())
      xout << "SVD rank " << svd.rank() << " in subspace of dimension " << S.cols() << std::endl;
    if (m_verbosity > 2 && svd.rank() < S.cols())
      xout << "singular values " << svd.singularValues().transpose() << std::endl;
    auto svmh = svd.singularValues().head(svd.rank()).eval();
    for (auto k = 0; k < svd.rank(); k++) svmh(k) = 1 / std::sqrt(svmh(k));
    auto Hbar =
        (svmh.asDiagonal()) * (svd.matrixU().leftCols(svd.rank()).adjoint()) * H * svd.matrixV().leftCols(svd.rank())
            * (svmh.asDiagonal());
//   xout << "S\n"<<S<<std::endl;
//   xout << "S singular values"<<(Eigen::DiagonalMatrix<T, Eigen::Dynamic, Eigen::Dynamic>(svd.singularValues().head(svd.rank())))<<std::endl;
//   xout << "S inverse singular values"<<Eigen::DiagonalMatrix<T, Eigen::Dynamic>(svd.singularValues().head(svd.rank())).inverse()<<std::endl;
//   xout << "S singular values"<<sv<<std::endl;
//   xout << "H\n"<<H<<std::endl;
//   xout << "Hbar\n"<<Hbar<<std::endl;
//   xout << "symmetric Hbar? " << (Hbar-Hbar.adjoint()).norm() << std::endl;
    Eigen::EigenSolver<Eigen::Matrix<scalar_type, Eigen::Dynamic, Eigen::Dynamic>> s(Hbar);
//    xout << "s.eigenvectors()\n"<<s.eigenvectors()<<std::endl;
    m_subspaceEigenvalues = s.eigenvalues();
    if (s.eigenvalues().imag().norm() < 1e-10 and s.eigenvectors().imag().norm() < 1e-10) { // real eigenvectors
      m_subspaceEigenvectors = svd.matrixV().leftCols(svd.rank()) * svmh.asDiagonal() * s.eigenvectors().real();

    } else { // complex eigenvectors
#ifdef __INTEL_COMPILER
      throw std::runtime_error("Intel compiler does not support working with complex eigen3 entities properly");
#endif
      m_subspaceEigenvectors = svd.matrixV().leftCols(svd.rank()) * svmh.asDiagonal() * s.eigenvectors();
    }
//   xout << "unsorted eigenvalues\n"<<m_subspaceEigenvalues<<std::endl;
//   xout << "unsorted eigenvectors\n"<<m_subspaceEigenvectors<<std::endl;

    {
      // sort
      auto eigval = m_subspaceEigenvalues;
      auto eigvec = m_subspaceEigenvectors;
      std::vector<size_t> map;
      for (Eigen::Index k = 0; k < Hbar.cols(); k++) {
        Eigen::Index ll;
        for (ll = 0; std::count(map.begin(), map.end(), ll) != 0; ll++);
        for (Eigen::Index l = 0; l < Hbar.cols(); l++) {
          if (std::count(map.begin(), map.end(), l) == 0) {
            if (eigval(l).real() < eigval(ll).real())
              ll = l;
          }
        }
        map.push_back(ll);
        m_subspaceEigenvalues[k] = eigval(ll);
//    xout << "new sorted eigenvalue "<<k<<", "<<ll<<", "<<eigval(ll)<<std::endl;
//    xout << eigvec.col(ll)<<std::endl;
        m_subspaceEigenvectors.col(k) = eigvec.col(ll);
      }
    }
//   xout << "sorted eigenvalues\n"<<m_subspaceEigenvalues<<std::endl;
//   xout << "sorted eigenvectors\n"<<m_subspaceEigenvectors<<std::endl;
//   xout << m_subspaceOverlap<<std::endl;
for (auto repeat=0; repeat<1; ++repeat)
    for (Eigen::Index k = 0; k < m_subspaceEigenvectors.cols(); k++) {
      if (std::abs(m_subspaceEigenvalues(k)) < 1e-12) { // special case of zero eigenvalue -- make some real non-zero vector definitely in the null space
        m_subspaceEigenvectors.col(k).real() += double(0.3256897) * m_subspaceEigenvectors.col(k).imag();
        m_subspaceEigenvectors.col(k).imag().setZero();
      }
      for (Eigen::Index l = 0; l < k; l++) {
        auto ovl =
            (m_subspaceEigenvectors.col(l).adjoint() * m_subspaceOverlap * m_subspaceEigenvectors.col(k))( 0, 0);
        auto norm =
            (m_subspaceEigenvectors.col(l).adjoint() * m_subspaceOverlap * m_subspaceEigenvectors.col(l))(
                0,
                0);
//      xout << "k="<<k<<", l="<<l<<", ovl="<<ovl<<" norm="<<norm<<std::endl;
//      xout << m_subspaceEigenvectors.col(k).transpose()<<std::endl;
//      xout << m_subspaceEigenvectors.col(l).transpose()<<std::endl;
        m_subspaceEigenvectors.col(k) -= m_subspaceEigenvectors.col(l) * ovl / norm;
//        xout<<"immediately after projection " << k<<l<<" "<< (m_subspaceEigenvectors.col(l).adjoint() * m_subspaceOverlap * m_subspaceEigenvectors.col(k))( 0, 0)<<std::endl;
      }
//      for (Eigen::Index l = 0; l < k; l++) xout<<"after projection loop " << k<<l<<" "<< (m_subspaceEigenvectors.col(l).adjoint() * m_subspaceOverlap * m_subspaceEigenvectors.col(k))( 0, 0)<<std::endl;
//      xout << "eigenvector"<<std::endl<<m_subspaceEigenvectors.col(k).adjoint()<<std::endl;
      auto ovl =
          (m_subspaceEigenvectors.col(k).adjoint() * m_subspaceOverlap * m_subspaceEigenvectors.col(k))(0,0);
      m_subspaceEigenvectors.col(k) /= std::sqrt(ovl.real());
//      for (Eigen::Index l = 0; l < k; l++)
//      xout<<"after normalisation " << k<<l<<" "<< (m_subspaceEigenvectors.col(l).adjoint() * m_subspaceOverlap * m_subspaceEigenvectors.col(k))( 0, 0)<<std::endl;
//      xout << "eigenvector"<<std::endl<<m_subspaceEigenvectors.col(k).adjoint()<<std::endl;
      // phase
      Eigen::Index lmax = 0;
      for (Eigen::Index l = 0; l < m_subspaceEigenvectors.rows(); l++) {
        if (std::abs(m_subspaceEigenvectors(l, k)) > std::abs(m_subspaceEigenvectors(lmax, k))) lmax = l;
      }
      if (m_subspaceEigenvectors(lmax, k).real() < 0) m_subspaceEigenvectors.col(k) = -m_subspaceEigenvectors.col(k);
//      for (Eigen::Index l = 0; l < k; l++)
//      xout << k<<l<<" "<<
//                       (m_subspaceEigenvectors.col(l).adjoint() * m_subspaceOverlap * m_subspaceEigenvectors.col(k))( 0, 0)<<std::endl;
    }
//     xout << "eigenvalues"<<std::endl<<m_subspaceEigenvalues<<std::endl;
//     xout << "eigenvectors"<<std::endl<<m_subspaceEigenvectors<<std::endl;
  }

  /*!
   * @brief form the combination of P and Q vectors to give the interpolated solution and corresponding residual (and maybe other vectors).
   * On entry, m_solution contains the interpolation
   *
   * @param solution On exit, the complete current solution (P and Q parts)
   * @param residual On exit, the Q contribution to the residual. The action of the matrix on the P solution is missing,
   * and has to be evaluated by the caller.
   * @param solutionP On exit, the solution projected to the P space
   * @param other On exit, interpolation of the other vectors
   */
  void doInterpolation(vectorSet& solution, vectorSet& residual,
                       vectorSetP& solutionP,
                       vectorSet& other) {
    for (auto& s : solution) s.scal(0);
    for (auto& s : residual) s.scal(0);
    size_t nP = m_Pvectors.size();
    solutionP.resize(residual.size());
    for (auto& s : other) s.scal(0);
    for (size_t kkk = 0; kkk < residual.size() && kkk < static_cast<size_t>(m_interpolation.rows()); kkk++) {
      solutionP[kkk].resize(nP);
      for (size_t l = 0; l < nP; l++)
        solution[kkk].axpy((solutionP[kkk][l] = this->m_interpolation(l, kkk)), m_Pvectors[l]);
//      xout << "square norm of solution after P contribution " << solution[kkk]->dot(*solution[kkk]) << std::endl;
      size_t l = nP;
      for (size_t ll = 0; ll < this->m_solutions.size(); ll++) {
        for (size_t lll = 0; lll < this->m_solutions[ll].size(); lll++) {
          if (this->m_vector_active[ll][lll]) {
            if (m_verbosity > 3)
              xout << "IterativeSolver::doInterpolation kkk=" << kkk << ", ll=" << ll << ", lll=" << lll << ", l=" << l
                   << std::endl;
            if (m_verbosity > 3) xout << "Interpolation:\n" << this->m_interpolation(l, kkk) << std::endl;
            solution[kkk].axpy(this->m_interpolation(l, kkk), this->m_solutions[ll][lll]);
            residual[kkk].axpy(this->m_interpolation(l, kkk), this->m_residuals[ll][lll]);
            if (other.size() > kkk) other[kkk].axpy(this->m_interpolation(l, kkk), this->m_others[ll][lll]);
            l++;
          }
        }
      }
//    xout << "residual before axpy "<<residual[kkk]->str()<<std::endl;
//      xout << "square norm of solution after Q contribution " << solution[kkk]->dot(*solution[kkk]) << std::endl;
//      xout << "m_residual_eigen " << m_residual_eigen << std::endl;
//      xout << "g.w before axpy contribution " << solution[kkk]->dot(*residual[kkk]) << std::endl;
      if (m_residual_eigen || (m_residual_rhs && m_augmented_hessian > 0))
        residual[kkk].axpy(-this->m_subspaceEigenvalues(kkk).real(), solution[kkk]);
      if (m_residual_rhs) residual[kkk].axpy(-1, this->m_rhs[kkk]);
//    xout << "residual after axpy "<<residual[kkk]<<std::endl;
//      xout << "solution after axpy "<<solution[kkk]<<std::endl;
//      xout << "square norm of solution after axpy contribution " << solution[kkk]->dot(*solution[kkk]) << std::endl;
//      xout << "g.w after axpy contribution " << solution[kkk]->dot(*residual[kkk]) << std::endl;
    }

  }

  void calculateErrors(const vectorSet& solution, const vectorSet& residual) {
    auto errortype = 0; // step . residual
    if (this->m_options.count("convergence") && this->m_options.find("convergence")->second == "step") errortype = 1;
    if (this->m_options.count("convergence") && this->m_options.find("convergence")->second == "residual")
      errortype = 2;
    if (m_verbosity > 5) {
      xout << "IterativeSolverBase::calculateErrors m_linear" << m_linear << std::endl;
      xout << "IterativeSolverBase::calculateErrors active";
      for (size_t root = 0; root < solution.size(); root++) xout << " " << m_active[root];
      xout << std::endl;
//      xout << "IterativeSolverBase::calculateErrors solution " << solution << std::endl;
      xout << std::endl;
//      xout << "IterativeSolverBase::calculateErrors residual " << residual << std::endl;
    }
    m_errors.clear();
    if (m_linear && errortype != 1) { // we can use the extrapolated residual if the problem is linear
//      xout << "calculateErrors solution.size=" << solution.size() << std::endl;
      for (size_t k = 0; k < solution.size(); k++) {
        m_errors.push_back(m_active[k] ?
                           std::fabs(
                               residual[k].dot(errortype == 2 ? residual[k] : solution[k])
                                   / solution[k].dot(solution[k])
                           )
                                       : 0);
//        xout << "solution . solution " << solution[k]->dot(*solution[k]) << std::endl;
//        xout << "residual . solution " << residual[k]->dot(*solution[k]) << std::endl;
//        xout << "residual . residual " << residual[k]->dot(*residual[k]) << std::endl;
      }
    } else {
      vectorSet step = solution;
      for (size_t k = 0; k < solution.size(); k++) {
        step[k].axpy(-1, m_solutions[m_lastVectorIndex][k]);
        m_errors.push_back(
            m_vector_active[m_lastVectorIndex][k] ? std::fabs(
                (errortype == 1 ? step : m_residuals[m_lastVectorIndex])[k].dot(
                    (errortype == 2 ? m_residuals[m_lastVectorIndex][k] : step[k])))
                                                  : 1);
      }
    }
    for (const auto& e : m_errors)
      if (std::isnan(e))
        throw std::overflow_error("NaN detected in error measure");
    if (errortype != 0)
      for (auto& e : m_errors) e = std::sqrt(e);
    m_error = *max_element(m_errors.begin(), m_errors.end());
    m_worst = max_element(m_errors.begin(), m_errors.end()) - m_errors.begin();
    if (m_verbosity > 5) {
      xout << "IterativeSolverBase::calculateErrors m_errors";
      for (size_t root = 0; root < solution.size(); root++) xout << " " << m_errors[root];
      xout << std::endl;
    }
  }

 protected:
  static void copyvec(std::vector<vectorSet>& history, const vectorSet& newvec) {
    vectorSet newcopy;
    newcopy.reserve(newvec.size());
    for (auto& v : newvec)
      newcopy.emplace_back(v,
                           LINEARALGEBRA_DISTRIBUTED
                               | LINEARALGEBRA_OFFLINE); // TODO template-ise these options
    history.emplace_back(newcopy);
  }
 public:

  size_t
  addVectorSet(const vectorSet& solution, const vectorSet& residual, const vectorSet& other) {
//   xout << "addVectorSet entry"<<std::endl;
    //      if (residual.m_vector_active.front()==0) xout <<"warning: inactive residual"<<std::endl;
    //      if (solution.m_vector_active.front()==0) xout <<"warning: inactive solution"<<std::endl;
//         for (size_t kkk=0; kkk<solution.size(); kkk++)
//             xout << "addVectorSet solution: "<<solution[kkk]<<std::endl;
    //      for (size_t kkk=0; kkk<residual.size(); kkk++)
    //          xout << "addVectorSet residual: "<<residual[kkk]<<std::endl;
//   xout << "addVectorSet before emplace_back()"<<std::endl;
    copyvec(m_residuals, residual);
    copyvec(m_solutions, solution);
    copyvec(m_others, other);
    m_vector_active.emplace_back();
    for (size_t k = 0; k < solution.size(); k++) m_vector_active.back().push_back(m_active[k]);
//   xout << "addVectorSet after emplace_back()"<<std::endl;
    const vectorSet& residual1 = residual;
    const vectorSet& solution1 = solution;//      for (size_t kkk=0; kkk<solution.size(); kkk++)
//   for (size_t kkk=0; kkk<solution.size(); kkk++)
//             xout << "solution: "<<solution[kkk]<<std::endl;
//         for (size_t kkk=0; kkk<residual.size(); kkk++)
//             xout << "residual: "<<residual[kkk]<<std::endl;
    //      xout << "appendQQMatrix"<<std::endl;
    //      xout << "residual"<<std::endl<<residual[0]<<std::endl;
    //      xout << "solution"<<std::endl<<solution[0]<<std::endl;
    auto nP = m_PQMatrix.rows();
    auto old_size = m_QQMatrix.rows();
//    auto new_size = old_size + count(residual1.m_vector_active.begin(), residual1.m_vector_active.end(), true);
    auto new_size = old_size;
    for (size_t i = 0; i < residual1.size(); i++) if (m_active[i]) ++new_size;
    //      xout << "old_size="<<old_size<<std::endl;
    //      xout << "new_size="<<new_size<<std::endl;
    m_QQMatrix.conservativeResize(new_size, new_size);
    m_QQOverlap.conservativeResize(new_size, new_size);
    m_PQMatrix.conservativeResize(nP, new_size);
    m_PQOverlap.conservativeResize(nP, new_size);
    m_subspaceRHS.conservativeResize(new_size + nP, m_rhs.size());
    std::vector<vectorSet>* bra = m_subspaceMatrixResRes ? &m_residuals : &m_solutions;
    auto k = old_size;
    for (size_t kkk = 0; kkk < residual1.size(); kkk++) {
      if (m_active[kkk]) {
        for (auto l = 0; l < nP; l++) {
          m_PQMatrix(l, k) = residual1[kkk].dot(m_Pvectors[l]);
          m_PQOverlap(l, k) = solution1[kkk].dot(m_Pvectors[l]);
        }
        size_t l = 0;
        for (size_t ll = 0; ll < m_solutions.size(); ll++) {
          for (size_t lll = 0; lll < m_solutions[ll].size(); lll++) {
            if (m_vector_active[ll][lll]) {
//              xout << "bra"<<std::endl<<(*bra)[ll][lll]<<std::endl;
//              xout << "residual1"<<std::endl<<residual1[kkk]<<std::endl;
//              xout << "m_solutions[ll]"<<std::endl<<m_solutions[ll][lll]<<std::endl;
              m_QQMatrix(k, l) = m_QQMatrix(l, k) = (*bra)[ll][lll].dot(residual1[kkk]);
              m_QQOverlap(k, l) = m_QQOverlap(l, k) = m_solutions[ll][lll].dot(solution1[kkk]);
              l++;
            }
          }
        }
        for (size_t l = 0; l < m_rhs.size(); l++) {
//      xout << "m_rhs: "<<m_rhs[l]<<std::endl;
//      xout << "solution1: "<<solution1[kkk]<<std::endl;
          m_subspaceRHS(nP + k, l) = m_rhs[l].dot(solution1[kkk]);
//      xout << "T product: "<<m_subspaceRHS(nP+k,l)<<std::endl;
        }
        m_dateOfBirth.push_back(++m_date);
        k++;
      }
    }
    if (m_verbosity > 3) {
      xout << "QQ Matrix: " << std::endl << m_QQMatrix << std::endl;
      xout << "QQ Overlap: " << std::endl << m_QQOverlap << std::endl;
    }
//   xout << "addVectorSet exit"<<std::endl;
    return m_residuals.size();
  }

  void deleteVector(size_t index) {
    //    xout << "deleteVector "<<index<<std::endl;
    //    xout << "old m_subspaceMatrix"<<std::endl<<m_subspaceMatrix<<std::endl;
    //    xout << "old m_subspaceOverlap"<<std::endl<<m_subspaceOverlap<<std::endl;
    size_t old_size = m_QQMatrix.rows();
    if (index >= old_size) throw std::logic_error("invalid index");
    auto new_size = old_size;
    size_t l = 0;
    for (size_t ll = 0; ll < m_solutions.size(); ll++) {
      for (size_t lll = 0; lll < m_solutions[ll].size(); lll++) {
        if (m_vector_active[ll][lll]) {
          if (l == index) { // this is the one to delete
            if (m_Weights.size() > l) m_Weights.erase(m_Weights.begin() + l);
            m_dateOfBirth.erase(m_dateOfBirth.begin() + l);
            m_vector_active[ll][lll] = false;
            //                    m_others[ll].m_vector_active[lll] = false;
            for (Eigen::Index l2 = l + 1; l2 < m_QQMatrix.cols(); l2++) {
              for (Eigen::Index k = 0; k < m_QQMatrix.rows(); k++) {
                m_QQMatrix(k, l2 - 1) = m_QQMatrix(k, l2);
                m_QQOverlap(k, l2 - 1) = m_QQOverlap(k, l2);
              }
              for (Eigen::Index k = 0; k < m_PQMatrix.rows(); k++) {
                m_PQMatrix(k, l2 - 1) = m_PQMatrix(k, l2);
                m_PQOverlap(k, l2 - 1) = m_PQOverlap(k, l2);
              }
              for (size_t k = 0; k < m_rhs.size(); k++) {
                m_subspaceRHS(m_PQMatrix.rows() + l2 - 1, k) = m_subspaceRHS(m_PQMatrix.rows() + l2, k);
              }
            }
            for (Eigen::Index l2 = l + 1; l2 < m_QQMatrix.rows(); l2++) {
              for (Eigen::Index k = 0; k < m_QQMatrix.rows(); k++) {
                m_QQMatrix(l2 - 1, k) = m_QQMatrix(l2, k);
                m_QQOverlap(l2 - 1, k) = m_QQOverlap(l2, k);
              }
            }
            new_size--;
          }
          l++;
        }
        size_t siz = 0;
        for (const auto& s : m_vector_active[ll]) if (s) ++siz;
        if (siz == 0) { // can delete the whole thing
          //TODO implement, and also individual element deletions to get rid of m_vector_active
        }
      }
    }
    m_QQMatrix.conservativeResize(new_size, new_size);
    m_QQOverlap.conservativeResize(new_size, new_size);
    //    xout << "new m_subspaceMatrix"<<std::endl<<m_subspaceMatrix<<std::endl;
    //    xout << "new m_subspaceOverlap"<<std::endl<<m_subspaceOverlap<<std::endl;
    buildSubspace();
  }

  std::vector<scalar_type> m_errors; //!< Error at last iteration
  scalar_type m_error; //!< worst error at last iteration
  size_t m_worst; //!< worst-converged solution, ie m_error = m_errors[m_worst]
  int m_date;
  bool m_subspaceMatrixResRes; // whether m_subspaceMatrix is Residual.Residual (true) or Solution.Residual (false)
  bool m_residual_eigen; // whether to subtract eigenvalue*solution when constructing residual
  bool m_residual_rhs; // whether to subtract rhs when constructing residual
  // whether to use RSPT to construct solution instead of diagonalisation
  std::vector<vectorSet> m_residuals;
  std::vector<vectorSet> m_solutions;
  std::vector<vectorSet> m_others;
  std::vector<std::vector<bool>> m_vector_active;
  vectorSet m_rhs;
  std::vector<int> m_dateOfBirth;
  size_t m_lastVectorIndex;
  std::vector<scalar_type> m_updateShift;
  Eigen::Matrix<scalar_type, Eigen::Dynamic, Eigen::Dynamic>
      m_interpolation; //!< The optimum combination of subspace vectors
  Eigen::Matrix<scalar_type, Eigen::Dynamic, Eigen::Dynamic> m_subspaceMatrix;
  Eigen::Matrix<scalar_type, Eigen::Dynamic, Eigen::Dynamic> m_subspaceOverlap;
  Eigen::Matrix<scalar_type, Eigen::Dynamic, Eigen::Dynamic> m_subspaceRHS;
  Eigen::Matrix<scalar_type, Eigen::Dynamic, Eigen::Dynamic> m_QQMatrix;
  Eigen::Matrix<scalar_type, Eigen::Dynamic, Eigen::Dynamic> m_QQOverlap;
  Eigen::MatrixXcd m_subspaceSolution; // FIXME templating
  Eigen::MatrixXcd m_subspaceEigenvectors; // FIXME templating
  Eigen::VectorXcd m_subspaceEigenvalues; // FIXME templating
 public:
  size_t m_dimension;
 private:
  unsigned int m_iterations;
  scalar_type m_singularity_threshold;
 protected:
  size_t m_added_vectors; //!< number of vectors recently added to subspace
  scalar_type
      m_augmented_hessian; //!< The scale factor for augmented hessian solution of linear inhomogeneous systems. Special values:
  //!< - 0: unmodified linear equations
  //!< - 1: standard augmented hessian
 public:
  scalar_type m_svdThreshold; ///< Threshold for singular-value truncation in linear equation solver.
 protected:
  std::vector<scalar_type> m_Weights; //!< weighting of error vectors in DIIS
  size_t m_maxQ; //!< maximum size of Q space when !m_orthogonalize

};

template<class T>
typename T::scalar_type inline
operator*(const typename IterativeSolver<T>::Pvector& a, const typename IterativeSolver<T>::Pvector& b) {
  typename T::scalar_type result = 0;
  for (const auto& aa: a)
    if (b.find(aa.first))
      result += aa.second * b[aa.first];
  return result;
}
}

namespace LinearAlgebra {

/*! @example LinearEigensystemExample.cpp */
/*! @example LinearEigensystemExample-paged.cpp */
/*!
* \brief A class that finds the lowest eigensolutions of a matrix using Davidson's method, i.e. preconditioned Lanczos
*
* Example of simplest use with a simple in-memory container for eigenvectors: @include LinearEigensystemExample.cpp
*
* Example using a P-space and offline distributed storage provided by the PagedVector class: @include LinearEigensystemExample-paged.cpp
*
* \tparam scalar Type of matrix elements
*/
template<class T>
class LinearEigensystem : public IterativeSolver<T> {
 public:
  using typename IterativeSolver<T>::scalar_type;
  using typename IterativeSolver<T>::value_type;
  using IterativeSolver<T>::m_verbosity;

  /*!
   * \brief LinearEigensystem
   */
  explicit LinearEigensystem() {
    this->m_residual_rhs = false;
    this->m_residual_eigen = true;
    this->m_linear = true;
  }

 private:

  void solveReducedProblem() override {
    if (this->m_rspt) {
      throw std::logic_error("RSPT not yet implemented");
    } else {
      this->diagonalizeSubspaceMatrix();
      this->m_interpolation = this->m_subspaceEigenvectors.block(0, 0, this->m_subspaceEigenvectors.rows(),
                                                                 std::min(int(this->m_roots), int(
                                                                     this->m_subspaceEigenvectors.rows()))).real();
    }

    this->m_updateShift.resize(this->m_roots);
    for (size_t root = 0; root < (size_t) this->m_roots;
         root++)
      this->m_updateShift[root] = -(1 + std::numeric_limits<scalar_type>::epsilon()) *
          (static_cast<Eigen::Index>(root) < this->m_subspaceEigenvectors.rows()
           ? this->m_subspaceEigenvalues[root].real()
           : 0);
  }

  void report() override {
    std::vector<scalar_type> ev = this->eigenvalues();
    if (m_verbosity > 0) {
      xout << "iteration " << this->iterations() << "[" << this->m_added_vectors << "]";
      if (!this->m_Pvectors.empty())
        xout << ", P=" << this->m_Pvectors.size();
      xout << ", error[" << this->m_worst << "] = " << this->m_error
           << ", eigenvalues: ";
      for (const auto e : ev) xout << " " << e;
      xout << std::endl;
    }
  }

};

/** @example LinearEquationsExample.cpp */
/*!
* \brief A class that finds the solutions of linear equation systems using a generalisation of Davidson's method, i.e. preconditioned Lanczos
*
* Example of simplest use: @include LinearEquationsExample.cpp
* \tparam scalar Type of matrix elements
*
*/
template<class T>
class LinearEquations : public IterativeSolver<T> {
  using typename IterativeSolver<T>::scalar_type;
  using typename IterativeSolver<T>::value_type;
 public:
  using vectorSet = typename std::vector<T>; ///< Container of vectors
  using IterativeSolver<T>::m_verbosity;

  /*!
   * \brief Constructor
   * \param rhs right-hand-side vectors. More can be added subsequently using addEquations(), provided iterations have not yet started.
   * \param augmented_hessian If zero, solve the inhomogeneous equations unmodified. If 1, solve instead
   * the augmented hessian problem. Other values scale the augmented hessian damping.
   */
  explicit LinearEquations(const vectorSet& rhs, scalar_type augmented_hessian = 0)
//    : m_linear(true)
//    , IterativeSolver<T>::m_orthogonalize(true)
//    , IterativeSolver<T>::m_residual_eigen(false)
//    , IterativeSolver<T>::m_residual_rhs(true)
//    , IterativeSolver<T>::m_augmented_hessian(augmented_hessian)
  {
    this->m_linear = true;
    this->m_residual_rhs = true;
    this->m_augmented_hessian = augmented_hessian;
    addEquations(rhs);
  }

  /*!
   * \brief add one or more equations to the set to be solved, by specifying their right-hand-side vector
   * \param rhs right-hand-side vectors to be added
   */
  void addEquations(const vectorSet& rhs) {
//   for (const auto &v : rhs) this->m_rhs.push_back(v);
    this->m_rhs.clear();
    this->m_rhs.reserve(rhs.size());
    for (const auto& v : rhs)
      this->m_rhs.emplace_back(v,
                               LINEARALGEBRA_DISTRIBUTED
                                   | LINEARALGEBRA_OFFLINE); // TODO template-ise these options
//   xout << "addEquations makes m_rhs.back()="<<this->m_rhs.back()<<std::endl;
  }

 protected:
  void solveReducedProblem() override {
    const auto nP = this->m_Pvectors.size();
    const auto nQ = this->m_QQMatrix.rows();
    const Eigen::Index n = nP + nQ;
//   xout << "solveReducedProblem initial subspace matrix\n"<<this->m_subspaceMatrix<<std::endl;
//   xout << "solveReducedProblem subspaceRHS\n"<<this->m_subspaceRHS<<std::endl;
    this->m_interpolation.conservativeResize(n, this->m_rhs.size());
    for (size_t root = 0; root < this->m_rhs.size(); root++) {
      if (this->m_augmented_hessian > 0) { // Augmented hessian
        this->m_subspaceMatrix.conservativeResize(n + 1, n + 1);
        this->m_subspaceOverlap.conservativeResize(n + 1, n + 1);
        for (Eigen::Index i = 0; i < n; i++) {
          this->m_subspaceMatrix(i, n) = this->m_subspaceMatrix(n, i) =
              -this->m_augmented_hessian * this->m_subspaceRHS(i, root);
          this->m_subspaceOverlap(i, n) = this->m_subspaceOverlap(n, i) = 0;
        }
        this->m_subspaceMatrix(n, n) = 0;
        this->m_subspaceOverlap(n, n) = 1;
//     xout << "solveReducedProblem augmented subspace matrix\n"<<this->m_subspaceMatrix<<std::endl;
//     xout << "solveReducedProblem augmented subspace metric\n"<<this->m_subspaceOverlap<<std::endl;
        Eigen::GeneralizedEigenSolver<Eigen::Matrix<scalar_type, Eigen::Dynamic, Eigen::Dynamic>>
            s(this->m_subspaceMatrix, this->m_subspaceOverlap);
        auto eval = s.eigenvalues();
        auto evec = s.eigenvectors();
        Eigen::Index imax = 0;
        for (Eigen::Index i = 0; i < n + 1; i++)
          if (eval(i).real() < eval(imax).real()) imax = i;
        this->m_subspaceEigenvalues.conservativeResize(root + 1);
        this->m_subspaceEigenvalues(root) = eval(imax);
//     xout << "eigenvectors\n"<<evec.real()<<std::endl;
//     xout << "eigenvalues\n"<<eval.real()<<std::endl;
//     xout << "imax="<<imax<<std::endl;
//     xout <<evec.col(imax)<<std::endl;
//     xout <<evec.col(imax).real()<<std::endl;
//     xout <<evec.col(imax).real().head(n)<<std::endl;
//     xout <<this->m_interpolation.col(root)<<std::endl;
        this->m_interpolation.col(root) =
            evec.col(imax).real().head(n) / (this->m_augmented_hessian * evec.real()(n, imax));
      } else { // straight solution of linear equations
        this->m_interpolation = this->m_subspaceMatrix.ldlt().solve(this->m_subspaceRHS);
      }
    }
//   xout << "m_interpolation\n"<<this->m_interpolation<<std::endl;
    this->m_subspaceMatrix.conservativeResize(n, n);
    this->m_subspaceOverlap.conservativeResize(n, n);
//   xout << "solveReducedProblem final subspace matrix\n"<<this->m_subspaceMatrix<<std::endl;
//   xout << "solveReducedProblem subspaceRHS\n"<<this->m_subspaceRHS<<std::endl;
  }

};

/** @example DIISexample.cpp */
/*!
* \brief A class that encapsulates accelerated convergence of non-linear equations
* through the DIIS or related methods.
*
* Example of simplest use: @include DIISexample.cpp
*
*/
template<class T>
class DIIS : public IterativeSolver<T> {
  using IterativeSolver<T>::m_residuals;
  using IterativeSolver<T>::m_solutions;
  using IterativeSolver<T>::m_others;
 public:
  using typename IterativeSolver<T>::scalar_type;
  using typename IterativeSolver<T>::value_type;
  using IterativeSolver<T>::m_verbosity;
  using IterativeSolver<T>::m_Weights;
  enum DIISmode_type {
    disabled ///< No extrapolation is performed
    , DIISmode ///< Direct Inversion in the Iterative Subspace
    , KAINmode ///< Krylov Accelerated Inexact Newton
  };

  /*!
 * \brief DIIS
 */
  DIIS()
      : m_maxDim(6), m_LastResidualNormSq(0), m_LastAmplitudeCoeff(1) {
    this->m_residual_rhs = false;
    this->m_residual_eigen = false;
    this->m_orthogonalize = false;
    this->m_roots = 1;
    this->m_maxQ = m_maxDim;
    setMode(DIISmode);
    Reset();
  }

  /*!
 * \brief Set options for DIIS.
 * \param mode Whether to perform DIIS, KAIN, or nothing.
 */
  virtual void setMode(enum DIISmode_type mode = DIISmode) {
    m_DIISmode = mode;
    this->m_subspaceMatrixResRes = mode != KAINmode;
//     this->m_preconditionResiduals = mode==KAINmode; // FIXME

    Reset();
    if (m_verbosity > 1)
      xout << "m_DIISmode set to " << m_DIISmode << std::endl;
  }

  /*!
 * \brief discards previous iteration vectors, but does not clear records
 */
  void Reset() {
    m_LastResidualNormSq = 1e99; // so it can be tested even before extrapolation is done
    this->m_lastVectorIndex = 0;
    while (this->m_QQMatrix.rows() > 0) this->deleteVector(0);;
    this->m_residuals.clear();
    this->m_solutions.clear();
    this->m_others.clear();
    this->m_Weights.clear();
  }

 protected:
  void solveReducedProblem() override {
    //	  xout << "Enter DIIS::solveReducedProblem"<<std::endl;
    //	  xout << "residual : "<<residual<<std::endl;
    //	  xout << "solution : "<<solution<<std::endl;
    this->m_updateShift.clear();
    this->m_updateShift.push_back(-(1 + std::numeric_limits<double>::epsilon()) * this->m_subspaceMatrix(0, 0));
    if (this->m_maxDim <= 1 || this->m_DIISmode == disabled) return;

    if (this->m_roots > 1) throw std::logic_error("DIIS does not handle multiple solutions");

    //  if (m_subspaceMatrix.rows() < 9) {
    //      xout << "m_subspaceMatrix on entry to DIIS::solveReducedProblem"<<std::endl<<m_subspaceMatrix<<std::endl;
    //  }
    size_t nDim = this->m_subspaceMatrix.rows();
    this->m_LastResidualNormSq = std::fabs(this->m_subspaceMatrix(nDim - 1, nDim - 1));
    //  xout << "this->m_LastResidualNormSq "<<this->m_LastResidualNormSq<<std::endl;


    Eigen::Array<scalar_type, Eigen::Dynamic, Eigen::Dynamic>
        d = this->m_subspaceMatrix.diagonal().array().abs();
    int worst = 0, best = 0;
    for (size_t i = 0; i < nDim; i++) {
      if (d(i) > d(worst)) worst = i;
      if (d(i) < d(best)) best = i;
    }
    scalar_type fBaseScale = std::sqrt(d(worst) * d(best));

//   while (nDim > this->m_maxDim) { // prune away the worst/oldest vector. Algorithm to be done properly yet
//    size_t prune = worst;
//    if (true || prune == nDim - 1) { // the current vector is the worst, so delete the oldest
//     //          xout << "this->m_dateOfBirth: "; for (auto b=this->m_dateOfBirth.begin(); b!=this->m_dateOfBirth.end(); b++) xout <<(*b); xout<<std::endl;
//     prune = std::min_element(this->m_dateOfBirth.begin(), this->m_dateOfBirth.end()) - this->m_dateOfBirth.begin();
//    }
//    //      xout << "prune="<<prune<<std::endl;
//    //  xout << "nDim="<<nDim<<", m_Weights.size()="<<m_Weights.size()<<std::endl;
//    this->deleteVector(prune);
//    nDim--;
//    //  xout << "nDim="<<nDim<<", m_Weights.size()="<<m_Weights.size()<<std::endl;
//   }
//   if (nDim != (size_t) this->m_subspaceMatrix.rows()) throw std::logic_error("problem in pruning");
//   if (m_Weights.size() != (size_t) this->m_subspaceMatrix.rows()) {
//    xout << "nDim=" << nDim << ", m_Weights.size()=" << m_Weights.size() << std::endl;
//    throw std::logic_error("problem after pruning weights");
//   }



    //  if (m_subspaceMatrix.rows() < 9) {
    //      xout << "m_subspaceMatrix"<<std::endl<<m_subspaceMatrix<<std::endl;
    //  }

    // build actual DIIS system for the subspace used.
    Eigen::VectorXd
        Rhs(nDim + 1),
        Coeffs(nDim + 1);
    Eigen::MatrixXd
        B(nDim + 1, nDim + 1);

    // Factor out common size scales from the residual dots.
    // This is done to increase numerical stability for the case when _all_
    // residuals are very small.
    B.block(0, 0, nDim, nDim) = this->m_subspaceMatrix / fBaseScale;
    Rhs.head(nDim) = Eigen::VectorXd::Zero(nDim);

    // make Lagrange/constraint lines.
    for (size_t i = 0; i < nDim; ++i)
      B(i, nDim) = B(nDim, i) = -m_Weights[i];
    B(nDim, nDim) = 0.0;
    Rhs[nDim] = -1;
//      xout << "B:"<<std::endl<<B<<std::endl;
//      xout << "Rhs:"<<std::endl<<Rhs<<std::endl;

    // invert the system, determine extrapolation coefficients.
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(B, Eigen::ComputeThinU | Eigen::ComputeThinV);
    svd.setThreshold(this->m_svdThreshold);
//    xout << "svdThreshold "<<this->m_svdThreshold<<std::endl;
//    xout << "U\n"<<svd.matrixU()<<std::endl;
//    xout << "V\n"<<svd.matrixV()<<std::endl;
//    xout << "singularValues\n"<<svd.singularValues()<<std::endl;
    Coeffs = svd.solve(Rhs).head(nDim);
    m_LastAmplitudeCoeff = Coeffs[nDim - 1];
    if (m_verbosity > 1) xout << "Combination of iteration vectors: " << Coeffs.transpose() << std::endl;
    for (size_t k = 0; k < (size_t) Coeffs.rows(); k++)
      if (std::isnan(Coeffs(k))) {
        xout << "B:" << std::endl << B << std::endl;
        xout << "Rhs:" << std::endl << Rhs << std::endl;
        xout << "Combination of iteration vectors: " << Coeffs.transpose() << std::endl;
        throw std::overflow_error("NaN detected in DIIS submatrix solution");
      }
    this->m_interpolation = Coeffs.head(nDim);
  }

  /*!
 * \brief Return the square L2 norm of the extrapolated residual from the last call to solveReducedProblem() or iterate().
 * \return
 */
 public:
  scalar_type fLastResidual() const { return m_LastResidualNormSq; }

  /*!
 * \brief Return the coefficient of the last residual vector in the extrapolated residual from the last call to solveReducedProblem() or iterate().
 * \return
 */
  scalar_type fLastCoeff() const { return m_LastAmplitudeCoeff; }

  unsigned int nLastDim() const { return m_residuals.size(); }

  unsigned int nMaxDim() const { return m_maxDim; }

  size_t m_maxDim; ///< Maximum DIIS dimension allowed.
  static void
  randomTest(size_t sample, size_t n = 100, double alpha = 0.1, double gamma = 0.0, DIISmode_type mode = DIISmode);

 private:
  typedef unsigned int uint;
  enum DIISmode_type m_DIISmode;

  // the following variables are kept for informative/displaying purposes
  scalar_type
  // dot(R,R) of last residual vector fed into this state.
      m_LastResidualNormSq,
  // coefficient the actual new vector got in the last DIIS step
      m_LastAmplitudeCoeff;

};

//extern template
//class LinearEigensystem<double>;
//extern template
//class LinearEquations<double>;
//extern template
//class DIIS<double>;

}

// C interface
extern "C" void
IterativeSolverLinearEigensystemInitialize(size_t nQ, size_t nroot, double thresh, unsigned int maxIterations,
                                           int verbosity, int orthogonalize);

extern "C" void
IterativeSolverLinearEquationsInitialize(size_t n,
                                         size_t nroot,
                                         const double* rhs,
                                         double aughes,
                                         double thresh,
                                         unsigned int maxIterations,
                                         int verbosity,
                                         int orthogonalize);

extern "C" void
IterativeSolverDIISInitialize(size_t n, double thresh, unsigned int maxIterations, int verbosity);

extern "C" void IterativeSolverFinalize();

extern "C" void
IterativeSolverAddVector(double* parameters, double* action, double* parametersP);

extern "C" int IterativeSolverEndIteration(double* c, double* g, double* error);

extern "C" void IterativeSolverAddP(size_t nP, const size_t* offsets, const size_t* indices,
                                    const double* coefficients, const double* pp,
                                    double* parameters, double* action, double* parametersP);

extern "C" void IterativeSolverEigenvalues(double* eigenvalues);

extern "C" void IterativeSolverOption(const char* key, const char* val);

extern "C" size_t IterativeSolverSuggestP(const double* solution,
                                          const double* residual,
                                          size_t maximumNumber,
                                          double threshold,
                                          size_t* indices);

#endif // ITERATIVESOLVER_H
