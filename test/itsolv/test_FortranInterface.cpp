// Tests of the Fortran interface (src/molpro/linalg/IterativeSolverF.F90); the tests themselves are in
// test_FortranInterfaceF.f90, and each returns non-zero on success.
#include "test.h"

#ifndef NOFORTRAN
extern "C" int test_verbosityf();
extern "C" int test_finalize_allf();
extern "C" int test_mpicommf();
extern "C" int test_initial_guess_and_max_iterf();
extern "C" int test_options_and_algorithmf();
extern "C" int test_minimizef();
extern "C" int test_problem_callbacksf();

TEST(FortranInterface, verbosity) { EXPECT_NE(test_verbosityf(), 0); }
TEST(FortranInterface, finalize_all) { EXPECT_NE(test_finalize_allf(), 0); }
TEST(FortranInterface, mpicomm) { EXPECT_NE(test_mpicommf(), 0); }
TEST(FortranInterface, initial_guess_and_max_iter) { EXPECT_NE(test_initial_guess_and_max_iterf(), 0); }
TEST(FortranInterface, options_and_algorithm) { EXPECT_NE(test_options_and_algorithmf(), 0); }
TEST(FortranInterface, minimize) { EXPECT_NE(test_minimizef(), 0); }
TEST(FortranInterface, problem_callbacks) { EXPECT_NE(test_problem_callbacksf(), 0); }
#endif
