#ifndef LINEARALGEBRA_TEST_VECTOR_TYPES_H_
#define LINEARALGEBRA_TEST_VECTOR_TYPES_H_
#include <cstddef>
#include <map>
#include <vector>
#include <deque>
#include <type_traits>
using scalar = double;
using Rvector = std::vector<scalar>;
// Ensure Rvector != Qvector while ensuring that Qvector still exposes the necessary API subset
// of std::vector - in particular operator[].
// std::deque just happens to fulfill this criterion
using Qvector = std::deque<scalar>;
static_assert(!std::is_constructible_v<Rvector, Qvector>,
        "We want to test our code works if Rvector can't directly constructed from Qvector");
static_assert(!std::is_constructible_v<Qvector, Rvector>,
        "We want to test our code works if Qvector can't directly constructed from Rvector");
using Pvector = std::map<size_t, scalar>;

#endif // LINEARALGEBRA_TEST_VECTOR_TYPES_H_
