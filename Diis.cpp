#include "Diis.h"

using namespace IterativeSolver;

Diis::Diis(ParameterSetTransformation updateFunction, ParameterSetTransformation residualFunction)
  : IterativeSolverBase(updateFunction, residualFunction), m_iNext(0)
{
  setOptions();
  m_LastResidualNormSq=1e99; // so it can be tested even before extrapolation is done
  Reset();
}

Diis::~Diis()
{
  for (std::vector<Storage*>::const_iterator s=store_.begin(); s!=store_.end(); s++)
    delete *s;
}

void Diis::setOptions(size_t maxDim, double threshold, DiisMode_type DiisMode)
{
   maxDim_ = maxDim;
   threshold_ = threshold;
   DiisMode_ = DiisMode;
   if (DiisMode_ == KAIN) throw std::invalid_argument("KAIN not yet supported");

   Reset();
//   std::cout << "maxDim_ set to "<<maxDim_<<" in setOptions"<<std::endl;
}

void Diis::Reset()
{

}

void Diis::LinearSolveSymSvd(Eigen::VectorXd& Out, const Eigen::MatrixXd& Mat, const Eigen::VectorXd& In, unsigned int nDim, double Thr)
{
    Eigen::VectorXd Ews(nDim);
    Eigen::VectorXd Xv(nDim); // input vectors in EV basis.
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es;
    es.compute(-Mat);
    Ews = es.eigenvalues();

    if (m_verbosity > 1) {
        std::cout << "diis::LinearSolveSymSvd"<<std::endl;
        std::cout << "Mat="<<Mat<<std::endl;
        std::cout << "Ews="<<Ews<<std::endl;
        std::cout << "In="<<In<<std::endl;
        std::cout << "es.eigenvectors()="<<es.eigenvectors()<<std::endl;
      }

    Xv = In.transpose()*es.eigenvectors();
//    std::cout << "Xv="<<Xv<<std::endl;
    for (size_t iEw = 0; iEw != nDim; ++ iEw)
        if (std::abs(Ews(iEw)) >= Thr)
            Xv(iEw) /= -Ews(iEw);
            // ^- note that this only screens by absolute value.
            // no positive semi-definiteness is assumed!
        else
            Xv(iEw) = 0.;
    if (m_verbosity > 1) std::cout << "Xv="<<Xv<<std::endl;
    Out = es.eigenvectors() * Xv;
    if (m_verbosity > 1) std::cout << "Out="<<Out<<std::endl;
    Out=Out.block(0,0,Out.size()-1,1);
//    std::cout << "Out="<<Out<<std::endl;
}



void Diis::extrapolate (ParameterVectorSet vectors, double weight)
{
  if (maxDim_ <= 1 || DiisMode_ == disabled) return;

  double fThisResidualDot = Dot(vectors[0],vectors[0],lengths_[0]);
  m_residualFunction(&fThisResidualDot,1);
  m_LastResidualNormSq = fThisResidualDot;

  if ( m_iNext == 0 && fThisResidualDot > threshold_ )
      // current vector is to be considered too wrong to be useful for DIIS
      // purposes. Don't store it.
      return;

  uint iThis = m_iNext;
  if (m_verbosity > 1) std::cout<< "iThis=m_iNext "<<m_iNext<<std::endl;
  assert(iThis < maxDim_);
  if (iThis >= m_iVectorAge.size()) m_iVectorAge.resize(iThis+1);
  if (iThis >= m_ErrorMatrix.cols()) m_ErrorMatrix.conservativeResize(iThis+1,iThis+1);
  m_ErrorMatrix(iThis,iThis)=fThisResidualDot;
  for ( uint i = 0; i < m_iVectorAge.size(); ++ i )
      m_iVectorAge[i] += 1;
  m_iVectorAge[iThis] = 0;
  if (m_verbosity>1) {
      std::cout << "iVectorAge:";
      for (std::vector<uint>::const_iterator a=m_iVectorAge.begin(); a!=m_iVectorAge.end(); a++)
        std::cout << " "<<*a;
      std::cout << std::endl;
    }

  // find set of vectors actually used in the current run and
  // find their common size scale.
  uint
      nDim;
      std::vector<uint> iUsedVecs( m_iVectorAge.size() + 1 );
      // ^- note: this is the DIIS dimension--the actual matrices and vectors have
      // dimension nDim+1 due to the Lagrange-Multipliers! ?? PJK ??
      double
      fBaseScale;
  FindUsefulVectors(&iUsedVecs[0], nDim, fBaseScale, iThis);
  if (m_verbosity>1) {
      std::cout << "iUsedVecs:";
      for (std::vector<uint>::const_iterator a=iUsedVecs.begin(); a!=iUsedVecs.end(); a++)
        std::cout << " "<<*a;
      std::cout << std::endl;
    }
  // transform iThis into a relative index.
  for ( uint i = 0; i < nDim; ++ i )
      if ( iThis == iUsedVecs[i] ) {
          iThis = i;
          break;
      }

  // write current residual and other vectors to their designated place
  if (m_verbosity>0) std::cout << "write current vectors to record "<<iUsedVecs[iThis]<<std::endl;
  for (unsigned int k=0; k<lengths_.size(); k++)
   store_[k]->write(vectors[k],lengths_[k]*sizeof(double),iUsedVecs[iThis]*lengths_[k]*sizeof(double));
  if (m_Weights.size()<=iUsedVecs[iThis]) m_Weights.resize(iUsedVecs[iThis]+1);
  m_Weights[iUsedVecs[iThis]] = weight;

  // go through previous residual vectors and form the dot products with them
  std::vector<double> ResDot = StorageDot(store_[0], vectors[0], lengths_[0], nDim);
  if (iThis >= nDim) ResDot.resize(iThis+1);
  ResDot[iThis] = fThisResidualDot;


  // update resident error matrix with new residual-dots
  for ( uint i = 0; i < nDim; ++ i ) {
      m_ErrorMatrix(iUsedVecs[i], iUsedVecs[iThis]) = ResDot[i];
      m_ErrorMatrix(iUsedVecs[iThis], iUsedVecs[i]) = ResDot[i];
  }

  // build actual DIIS system for the subspace used.
  Eigen::VectorXd
      Rhs(nDim+1),
      Coeffs(nDim+1);
  Eigen::MatrixXd
      B(nDim+1, nDim+1);

  // Factor out common size scales from the residual dots.
  // This is done to increase numerical stability for the case when _all_
  // residuals are very small.
  for ( uint nRow = 0; nRow < nDim; ++ nRow )
      for ( uint nCol = 0; nCol < nDim; ++ nCol )
          B(nRow, nCol) = m_ErrorMatrix(iUsedVecs[nRow], iUsedVecs[nCol])/fBaseScale;

  // make Lagrange/constraint lines.
  for ( uint i = 0; i < nDim; ++ i ) {
      B(i, nDim) = -m_Weights[iUsedVecs[i]];
      B(nDim, i) = -m_Weights[iUsedVecs[i]];
      Rhs[i] = 0.0;
  }
  B(nDim, nDim) = 0.0;
  Rhs[nDim] = -weight;

  // invert the system, determine extrapolation coefficients.
  LinearSolveSymSvd(Coeffs, B, Rhs, nDim+1, 1.0e-10);
  if (m_verbosity>1) std::cout << "Combination of iteration vectors: "<<Coeffs.transpose()<<std::endl;

  // Find a storage place for the vector in the next round. Either
  // an empty slot or the oldest vector.
  uint iOldestAge = m_iVectorAge[0];
  m_iNext = 0;
  if (m_iVectorAge.size() < maxDim_) {
    m_iNext = m_iVectorAge.size(); m_iVectorAge.push_back(0);
    } else
    for ( uint i = m_iVectorAge.size(); i != 0; -- i ){
      if ( iOldestAge <= m_iVectorAge[i-1] ) {
          iOldestAge = m_iVectorAge[i-1];
          m_iNext = i-1;
      }
  }

//     bool
//         PrintDiisState = true;
//     if ( PrintDiisState ) {
//         std::ostream &xout = std::cout;
//         xout << "  iUsedVecs: "; for ( uint i = 0; i < nDim; ++ i ) xout << " " << iUsedVecs[i]; xout << std::endl;
//         PrintMatrixGen( xout, m_ErrorMatrix.data(), nMaxDim(), 1, nMaxDim(), m_ErrorMatrix.nStride(), "DIIS-B (resident)" );
//         PrintMatrixGen( xout, B.data(), nDim+1, 1, nDim+1, B.nStride(), "DIIS-B/I" );
//         PrintMatrixGen( xout, Rhs.data(), 1, 1, nDim+1, 1, "DIIS-Rhs" );
//         PrintMatrixGen( xout, Coeffs.data(), 1, 1, nDim+1, 1, "DIIS-C" );
//         xout << std::endl;
//     }

  // now actually perform the extrapolation on the residuals
  // and amplitudes.
  m_LastAmplitudeCoeff = Coeffs[iThis];
//  TR.InterpolateFrom( Coeffs[iThis], Coeffs.data(), ResRecs, AmpRecs,
//      nDim, iThis, *m_Storage.pDevice, m_Memory );
  for (uint k=0; k<vectors.size(); k++) {
   StorageCombine(store_[k],vectors[k],Coeffs,lengths_[k],iUsedVecs);
}
}

void Diis::iterate(double *residual, double *solution, double weight, std::vector<double *> other)
{
  std::vector<double*> vectors;
  vectors.push_back(residual);
  vectors.push_back(solution);
  for (std::vector<double*>::const_iterator o=other.begin(); o!=other.end(); o++) vectors.push_back(*o);
  extrapolate(vectors,weight);
  update(residual,solution);
}

void Diis::FindUsefulVectors(uint *iUsedVecs, uint &nDim, double &fBaseScale, uint iThis)
{
    // remove lines from the system which correspond to vectors which are too bad
    // to be really useful for extrapolation purposes, but which might break
    // numerical stability if left in.
    double const
        fThrBadResidual = 1e12;
    double
        fBestResidualDot = m_ErrorMatrix(iThis,iThis),
        fWorstResidualDot = fBestResidualDot;
    assert(m_iVectorAge[iThis] < VecNotPresent);
    for ( uint i = 0; i < m_iVectorAge.size(); ++ i ) {
        if ( m_iVectorAge[i] >= VecNotPresent ) continue;
        fBestResidualDot = std::min( m_ErrorMatrix(i,i), fBestResidualDot );
    }
    nDim = 0;
    for ( uint i = 0; i < m_iVectorAge.size(); ++ i ){
        if ( i != iThis && m_iVectorAge[i] >= VecNotPresent )
            continue;
        if ( i != iThis && m_ErrorMatrix(i,i) > fBestResidualDot * fThrBadResidual) {
            m_iVectorAge[i] = VecNotPresent; // ignore this slot next time.
            continue;
        }
        fWorstResidualDot = std::max( m_ErrorMatrix(i,i), fWorstResidualDot );
        iUsedVecs[nDim] = i;
        ++ nDim;
    }

    fBaseScale = std::sqrt(fWorstResidualDot * fBestResidualDot);
    if ( fBaseScale <= 0. )
        fBaseScale = 1.;
}