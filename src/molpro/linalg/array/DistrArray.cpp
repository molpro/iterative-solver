#include "DistrArray-implementation.h"

namespace molpro::linalg::array::util {
std::map<size_t, double> select_max_dot_broadcast(size_t n, std::map<size_t, double>& local_selection,
                                                  MPI_Comm communicator) {
  auto indices = std::vector<molpro::linalg::array::DistrArray<double>::index_type>();
  auto values = std::vector<double>();
  indices.reserve(n);
  values.reserve(n);
  for (const auto& el : local_selection) {
    indices.push_back(el.first);
    values.push_back(el.second);
  }
  indices.resize(n);
  values.resize(n);
  int n_dummy = static_cast<int>(n) - local_selection.size();
  MPI_Request requests[3];
  int comm_rank, comm_size;
  MPI_Comm_rank(communicator, &comm_rank);
  MPI_Comm_size(communicator, &comm_size);
  if (comm_rank == 0) {
    auto n_tot = n * comm_size;
    auto n_dummy_elements = std::vector<int>(comm_size, 0);
    MPI_Igather(&n_dummy, 1, MPI_INT, &n_dummy_elements[0], 1, MPI_INT, 0, communicator, &requests[0]);
    indices.resize(n_tot);
    values.resize(n_tot);
    MPI_Igather(MPI_IN_PLACE, n, MPI_UNSIGNED_LONG, &indices[0], n, MPI_UNSIGNED_LONG, 0, communicator, &requests[1]);
    MPI_Igather(MPI_IN_PLACE, n, MPI_DOUBLE, &values[0], n, MPI_UNSIGNED_LONG, 0, communicator, &requests[2]);
    MPI_Waitall(3, requests, MPI_STATUSES_IGNORE);
    local_selection.clear();
    using pair_t = std::pair<double, size_t>;
    auto pq = std::priority_queue<pair_t, std::vector<pair_t>, std::greater<>>();
    for (size_t i = 0; i < n; ++i)
      pq.emplace(std::numeric_limits<double>::min(),
                 std::numeric_limits<molpro::linalg::array::DistrArray<double>::index_type>::max());
    for (size_t i = 0, ii = 0; i < comm_size; ++i) {
      for (size_t j = 0; j < n - n_dummy_elements[i]; ++j, ++ii) {
        pq.emplace(values[ii], indices[ii]);
        pq.pop();
      }
      ii += n_dummy_elements[i];
    }
    indices.resize(n);
    values.resize(n);
    for (size_t i = 0; i < n; ++i) {
      values[i] = pq.top().first;
      indices[i] = pq.top().second;
      pq.pop();
    }
  } else {
    MPI_Igather(&n_dummy, 1, MPI_INT, nullptr, 1, MPI_INT, 0, communicator, &requests[0]);
    MPI_Igather(&indices[0], n, MPI_UNSIGNED_LONG, nullptr, n, MPI_UNSIGNED_LONG, 0, communicator, &requests[1]);
    MPI_Igather(&values[0], n, MPI_DOUBLE, nullptr, n, MPI_DOUBLE, 0, communicator, &requests[2]);
    MPI_Waitall(3, requests, MPI_STATUSES_IGNORE);
  }
  MPI_Ibcast(&indices[0], n, MPI_UNSIGNED_LONG, 0, communicator, &requests[0]);
  MPI_Ibcast(&values[0], n, MPI_DOUBLE, 0, communicator, &requests[1]);
  MPI_Waitall(2, requests, MPI_STATUSES_IGNORE);
  local_selection.clear();
  for (size_t i = 0; i < n; ++i)
    local_selection.emplace(indices[i], values[i]);
  return local_selection;
}
} // namespace molpro::linalg::array::util

using molpro::linalg::array::DistrArray;
// using molpro::linalg::array::util::extrema;
template std::list<std::pair<typename DistrArray<double>::index_type, double>>
molpro::linalg::array::util::extrema<std::less<double>, double>(const DistrArray<double>& x, int n);
template std::list<std::pair<typename DistrArray<double>::index_type, double>>
molpro::linalg::array::util::extrema<molpro::linalg::array::util::CompareAbs<double, std::less<>>, double>(
    const DistrArray<double>& x, int n);
template std::list<std::pair<typename DistrArray<double>::index_type, double>>
molpro::linalg::array::util::extrema<std::greater<double>, double>(const DistrArray<double>& x, int n);
template std::list<std::pair<typename DistrArray<double>::index_type, double>>
molpro::linalg::array::util::extrema<molpro::linalg::array::util::CompareAbs<double, std::greater<>>, double>(
    const DistrArray<double>& x, int n);

template class molpro::linalg::array::DistrArray<double>;
