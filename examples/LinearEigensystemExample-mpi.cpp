#include "molpro/linalg/IterativeSolver.h"
#include <fstream>
#include <iomanip>
#include <molpro/linalg/array/ArrayHandlerDistr.h>
#include <molpro/linalg/array/ArrayHandlerDistrSparse.h>
#include <molpro/linalg/array/ArrayHandlerIterable.h>
#include <molpro/linalg/array/ArrayHandlerIterableSparse.h>
#include <molpro/linalg/array/ArrayHandlerSparse.h>
#include <molpro/linalg/array/DistrArrayMPI3.h>
#include <molpro/linalg/array/util/Distribution.h>
#include <mpi.h>
#include <vector>

using molpro::linalg::array::ArrayHandler;
using molpro::linalg::array::ArrayHandlerDistr;
using molpro::linalg::array::ArrayHandlerDistrSparse;
using molpro::linalg::array::ArrayHandlerIterable;
using molpro::linalg::array::ArrayHandlerIterableSparse;
using molpro::linalg::array::ArrayHandlerSparse;
using molpro::linalg::iterativesolver::ArrayHandlers;
// Find lowest eigensolutions of a matrix obtained from an external file
using Rvector = molpro::linalg::array::DistrArrayMPI3;
using Qvector = molpro::linalg::array::DistrArrayMPI3;
using Pvector = std::map<size_t, double>;
int n; // dimension of problem
int mpi_rank;
std::vector<double> hmat;

double matrix(const size_t i, const size_t j) { return hmat[i * n + j]; }

void action(const std::vector<Rvector>& psx, std::vector<Rvector>& outputs) {
  for (size_t k = 0; k < psx.size(); k++) {
    std::vector<double> allx(n);
    psx[k].get(0, n - 1, allx.data());
    auto range = outputs[k].distribution().range(mpi_rank);
    outputs[k].fill(0);
    std::cout << "MPI rank="<<mpi_rank<<"; allx:";
    for (const auto& x : allx)
      std::cout << " " << x;
    std::cout << std::endl;
    for (size_t i = range.first; i < range.second; i++) {
      double result = 0;
      for (size_t j = 0; j < n; j++)
        result += matrix(i, j) * allx[j];
      outputs[k].set(i, result);
      std::cout << "result "<<i<<" "<<result<<std::endl;
    }
  }
}

void update(std::vector<Rvector>& psc, const std::vector<Rvector>& psg, size_t nwork,
            std::vector<double> shift = std::vector<double>()) {
  for (size_t k = 0; k < nwork; k++) {
    auto range = psg[k].distribution().range(mpi_rank);
    assert(range == psc[k].distribution().range(mpi_rank));
    std::vector<double> c_chunk(range.second - range.first);
    std::vector<double> g_chunk(range.second - range.first);
    psc[k].get(range.first, range.second - 1, c_chunk.data());
    psg[k].get(range.first, range.second - 1, g_chunk.data());
    for (size_t i = range.first; i < range.second; i++) {
      c_chunk[i - range.first] -= g_chunk[i - range.first] / (1e-12 - shift[k] + matrix(i, i));
    }
    psc[k].put(range.first, range.second - 1, c_chunk.data());
    std::cout << "MPI rank="<<mpi_rank<<"; c_chunk:";
    for (const auto& x : c_chunk)
      std::cout << " " << x;
    std::cout << std::endl;
  }
  MPI_Barrier(MPI_COMM_WORLD);
}

int main(int argc, char* argv[]) {
  MPI_Init(&argc, &argv);
  int mpi_size;
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
  for (const auto& file : std::vector<std::string>{"hf", "bh"}) {
    for (const auto& nroot : std::vector<int>{1, 2, 4}) {
      if (mpi_rank == 0) {
        std::ifstream f(std::string{"examples/"} + file + ".hamiltonian");
        f >> n;
        molpro::cout << "\n*** " << file << " (dimension " << n << "), " << nroot << " roots, mpi_size = " << mpi_size
                     << std::endl;
        hmat.resize(n * n);
        for (auto i = 0; i < n * n; i++)
          f >> hmat[i];
      }
      MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
      if (mpi_rank != 0)
        hmat.resize(n * n);
      MPI_Bcast(hmat.data(), n * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
      std::vector<double> diagonals;
      diagonals.reserve(n);
      for (auto i = 0; i < n; i++)
        diagonals.push_back(matrix(i, i));
      auto rr = std::make_shared<ArrayHandlerDistr<Rvector, Rvector>>();
      auto qq = std::make_shared<ArrayHandlerDistr<Qvector, Qvector>>();
      auto pp = std::make_shared<ArrayHandlerSparse<Pvector, Pvector>>();
      auto rq = std::make_shared<ArrayHandlerDistr<Rvector, Qvector>>();
      auto rp = std::make_shared<ArrayHandlerDistrSparse<Rvector, Pvector>>();
      auto qr = std::make_shared<ArrayHandlerDistr<Qvector, Rvector>>();
      auto qp = std::make_shared<ArrayHandlerDistrSparse<Qvector, Pvector>>();
      auto handlers = ArrayHandlers<Rvector, Rvector, Pvector>{rr, qq, pp, rq, rp, qr, qp};
      //      auto handlers = ArrayHandlers<Rvector, Qvector, Pvector>{};
      molpro::linalg::LinearEigensystem<Rvector, Qvector, Pvector> solver{handlers};
      solver.m_verbosity = 1;
      solver.m_roots = nroot;
      solver.m_thresh = 1e-9;
      std::vector<Rvector> g;
      std::vector<Rvector> x;
      for (size_t root = 0; root < solver.m_roots; root++) {
        x.emplace_back(n, MPI_COMM_WORLD);
        g.emplace_back(n, MPI_COMM_WORLD);
        x.back().allocate_buffer();
        g.back().allocate_buffer();
        x.back().fill(0);
        auto guess = std::min_element(diagonals.begin(), diagonals.end()) - diagonals.begin(); // initial guess
        if (x.back().distribution().cover(guess) < n)
          x.back().set(guess, 1);
        *std::min_element(diagonals.begin(), diagonals.end()) = 1e99;
      }
      std::vector<std::vector<double>> Pcoeff(solver.m_roots);
      int nwork = solver.m_roots;
      for (auto iter = 0; iter < 100; iter++) {
        action(x, g);
        nwork = solver.addVector(x, g, Pcoeff);
        if (mpi_rank == 0)
          solver.report();
        if (nwork == 0)
          break;
        update(x, g, nwork, solver.working_set_eigenvalues());
      }
      std::cout << "Error={ ";
      for (const auto& e : solver.errors())
        std::cout << e << " ";
      std::cout << "} after " << solver.iterations() << " iterations" << std::endl;
      for (size_t root = 0; root < solver.m_roots; root++) {
        std::cout << "Eigenvalue " << std::fixed << std::setprecision(9) << solver.eigenvalues()[root] << std::endl;
        //        solver.solution(root, x.front(), g.front(), Pcoeff.front());
        //        std::cout << "Eigenvector: (norm=" << std::sqrt(x[0].dot(x[0])) << "): ";
        //        for (size_t k = 0; k < n; k++)
        //          std::cout << " " << (x[0])[k];
        //        std::cout << std::endl;
      }
    }
  }
  MPI_Finalize();
}
