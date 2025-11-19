#include "ExampleProblem.h"
#include <iostream>
#include <molpro/linalg/itsolv/SolverFactory.h>
#include <molpro/linalg/array/Span.h>
#include <molpro/mpi.h>

int main(int argc, char* argv[]) {
  molpro::mpi::init();
  {
    auto problem = ExampleProblem<molpro::linalg::array::Span<double>>(argc > 1 ? std::stoi(argv[1]) : 20);
    using Rvector = decltype(problem)::container_t;
	// Note: Qvector needs to be allocatable, i.e. have a constructor taking in a size
	// which will allocate an appropriately sized buffer or we have to specialize
	// the molpro::linalg::array::allocate_array template function for this type to do
	// the allocation for us. The default implementation of that function will throw an
	// exception at runtime.
	using Qvector = std::vector<double>;
    auto solver = molpro::linalg::itsolv::create_LinearEigensystem<Rvector, Qvector>("Davidson");
    solver->set_n_roots(argc > 2 ? std::stoi(argv[2]) : 2);
    //  solver->set_verbosity(molpro::linalg::itsolv::Verbosity::Detailed);
    solver->set_max_iter(100);

	// This simulates having some sort of externally managed buffers that are used
	// as main working memory in iterative-solver. The buffers are passed as non-owning
	// spans, which act as a view to these buffers.
	std::vector<Rvector::value_type> cbuf(problem.n);
	std::vector<Rvector::value_type> gbuf(problem.n);
    Rvector c(cbuf.data(), cbuf.size());
    Rvector g(gbuf.data(), gbuf.size());

    if (not solver->solve(c, g, problem, true))
      std::cout << "failed" << std::endl;
    else
      std::cout << "converged in " << solver->statistics().iterations << " iterations" << std::endl;
    solver->solution(c, g);
    for (const auto& ev : solver->eigenvalues())
      std::cout << "Final eigenvalue: " << ev << std::endl;
  }
  molpro::mpi::finalize();
}

#include <molpro/linalg/itsolv/SolverFactory-implementation.h>
#include <vector>
template class molpro::linalg::itsolv::SolverFactory<molpro::linalg::array::Span<double>, std::vector<double>,
                                                     std::map<size_t, double>>;
