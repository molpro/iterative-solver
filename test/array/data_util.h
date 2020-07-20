#ifndef LINEARALGEBRA_TEST_ARRAY_DATA_UTIL_H
#define LINEARALGEBRA_TEST_ARRAY_DATA_UTIL_H
namespace {
// File contents: "/dataset" where dataset is a float64 arange(0,30)
const std::string name_single_dataset{std::string(ARRAY_DATA) + "/single_dataset.hdf5"};
// File contents: "/group1/group2/dataset" where dataset is a float64 arange(0,30)
const std::string name_inner_group_dataset{std::string(ARRAY_DATA) + "/inner_group_dataset.hdf5"};
} // namespace
#endif // LINEARALGEBRA_TEST_ARRAY_DATA_UTIL_H