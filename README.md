iterative-solver
================

[![Build and test](https://github.com/molpro/iterative-solver/actions/workflows/build-and-test.yml/badge.svg)](https://github.com/molpro/iterative-solver/actions/workflows/build-and-test.yml)

[//]: # (&#40;https://github.com/molpro/iterative-solver/commits/master&#41;)
[![license](https://img.shields.io/badge/license-MIT-green.svg)](https://github.com/molpro/iterative-solver/blob/master/LICENSE)

[//]: # ([![license]&#40;https://img.shields.io/badge/documentation-blue.svg&#41;]&#40;https://molpro.gitlab.io/linearalgebra/&#41;)


## Overview

Implements iterative solvers for linear and non-linear problems and
distributed arrays for HPC. The solvers are specialised to work with specific data types, but are also templated on the
container allowing for easy integration into existing software.

List of key features:
* Implements iterative solvers for eigenvalue problem, linear equations, optimisation (L-BFGS) and non-linear equations 
(DIIS)
* Novel algorithms including P-space and D-space (see paper)
* Structured to allow easy addition of new solvers or modification of the current ones without changes to the user's 
code
* Templated on container for easy integration into existing programs
  *  User defined containers can be used without modification with the help of array handler abstraction
* Specialised for double and complex value types, so that all heavy numerical operations are only compiled once
* Provides distributed arrays in memory and on disk for HPC
* Contains Fortran and C wrappers

## Installation

CMake is used to build the library and it can integrate easily with other CMake builds.

```cmake
include(FetchContent)
FetchContent_Declare(
        iterative-solver
        GIT_REPOSITORY https://github.com/molpro/iterative-solver.git
        GIT_TAG ${COMMIT_HASH_OR_TAG_VALUE})
FetchContent_MakeAvailable(iterative-solver)
target_link_libraries(${YOUR_LIBRARY_NAME} PUBLIC molpro::iterative-solver)
```

## Usage

### Interfaces

There is a hierarchy of abstract classes defined in `molpro/linalg/itsolv/IterativeSolver.h` with `IterativeSolver` 
defining the interface for common functionality and `LinearEigensystem`, `LinearEquations`, `Optimize` and
`NonLinearEquations` defining functions that are specific to each type of solver. They are provided to reduce header 
bloat in user's code. 

Each type of solver has at least one implementation class, e.g. `LinearEigensystemDavidson` in 
`molpro/linalg/itsolv/LinearEigensystemDavidson.h`, which implements the full interface. The library is designed to make it easy
to add new iterative solvers, so there might be more than one implementation.

### Example

The simplest way to use the library is to define a `Problem` subclass that supplies
the matrix action and (optionally) diagonals, then call `solve()`. Here is an example
of using `LinearEigensystemDavidson`,

```cpp
#include <molpro/linalg/itsolv/LinearEigensystemDavidson.h>
#include <molpro/linalg/itsolv/SolverFactory.h>
// ...
using R = std::vector<double>;
class MyProblem : public molpro::linalg::itsolv::Problem<R> {
    void action(const CVecRef<R>& parameters, const VecRef<R>& actions) const override { /* ... */ }
    bool diagonals(R& d) const override { /* fill d */ return true; }
};

auto solver = molpro::linalg::itsolv::create_LinearEigensystem<R>();
R parameters(n), actions(n);
MyProblem problem;
if (!solver->solve(parameters, actions, problem, /*generate_initial_guess=*/true)) {
    // deal with the unconverged case
}
solver->solution(parameters, actions);
```

More advanced users can copy and modify the code in `solve()` to tailor it for their own use, see implementation in `molpro/linalg/itsolv/IterativeSolverTemplate.h`.

### Containers and array handlers

In many programs there are special containers for storing the vectors and operating on them. This might be for efficiency reasons,
e.g. exploiting symmetry of the problem, or for collecting metadata, e.g. memory usage and operation count. In either case,
The code is templated on container types to ease adaptation.

To avoid the code-bloat of header only libraries all of the numerically intensive work is specialised for `double` and `std::complex<double>` types.
This makes recompilation of IterativeSolver with different container types very fast.

There are 3 types of containers:

* R &mdash; working set container
  * fast access to elements
  * used sparingly to preserve memory 
* Q &mdash; slow container
  * slow access to elements
  * used for most of the subspace
  * usually on disk where there is lots of storage space
* P &mdash; sparse container
  * light-weight sparse container (e.g. `std::map<size_t, double>`)

IterativeSolver does not modify containers directly instead using ArrayHandler for all array related operations. This allows for only modest
restrictions on containers:

* must have a  move constructor

* must have conforming element type (currently `double` and `std::complex<double>`, but this can be easily extended)

ArrayHandler is an abstract class used by IterativeSolver to perform copy and linear algebra operations (`dot`, `axpy`). 
iterative-solver provides implementations for Iterable containers (e.g. `std::vector`), distributed containers (e.g. `molpro::linalg::array::DistrArray`),
and mapped containers (e.g. `std::map`). However, some users might need/want to provide their own implementations. 

## Citing

Any publications resulting from this work should cite relevant papers in CITE.txt

## List of Contributors

Peter Knowles

Marat Sibaev

Iakov Polyak

Rob Welch
