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

template <class Compare>
std::list<std::pair<typename DistrArray<double>::index_type, double>> extrema(const DistrArray<double>& x, int n) {
  if (x.empty())
    return {};
  auto buffer = x.local_buffer();
  auto length = buffer->size();
  auto nmin = length > n ? n : length;
  auto loc_extrema = std::list<std::pair<typename DistrArray<double>::index_type, double>>();
  for (size_t i = 0; i < nmin; ++i)
    loc_extrema.emplace_back(buffer->start() + i, (*buffer)[i]);
  auto compare = Compare();
  auto compare_pair = [&compare](const auto& p1, const auto& p2) { return compare(p1.second, p2.second); };
  for (size_t i = nmin; i < length; ++i) {
    loc_extrema.emplace_back(buffer->start() + i, (*buffer)[i]);
    loc_extrema.sort(compare_pair);
    loc_extrema.pop_back();
  }
  auto indices_loc = std::vector<typename DistrArray<double>::index_type>(n, x.size() + 1);
  auto indices_glob = std::vector<typename DistrArray<double>::index_type>(n);
  auto values_loc = std::vector<double>(n);
  auto values_glob = std::vector<double>(n);
  size_t ind = 0;
  for (auto& it : loc_extrema) {
    indices_loc[ind] = it.first;
    values_loc[ind] = it.second;
    ++ind;
  }
  MPI_Request requests[3];
  int comm_rank, comm_size;
  MPI_Comm_rank(x.communicator(), &comm_rank);
  MPI_Comm_size(x.communicator(), &comm_size);
  // root collects values, does the final sort and sends the result back
  if (comm_rank == 0) {
    auto ntot = n * comm_size;
    indices_loc.resize(ntot);
    values_loc.resize(ntot);
    auto ndummy = std::vector<int>(comm_size);
    auto d = int(n - nmin);
    MPI_Igather(&d, 1, MPI_INT, ndummy.data(), 1, MPI_INT, 0, x.communicator(), &requests[0]);
    MPI_Igather(MPI_IN_PLACE, n, MPI_UNSIGNED_LONG, indices_loc.data(), n, MPI_UNSIGNED_LONG, 0, x.communicator(),
                &requests[1]);
    MPI_Igather(MPI_IN_PLACE, n, MPI_DOUBLE, values_loc.data(), n, MPI_UNSIGNED_LONG, 0, x.communicator(),
                &requests[2]);
    MPI_Waitall(3, requests, MPI_STATUSES_IGNORE);
    auto tot_dummy = std::accumulate(begin(ndummy), end(ndummy), 0);
    if (tot_dummy != 0) {
      size_t shift = 0;
      for (size_t i = 0, ind = 0; i < comm_size; ++i) {
        for (size_t j = 0; j < n - ndummy[i]; ++j, ++ind) {
          indices_loc[ind] = indices_loc[ind + shift];
          values_loc[ind] = values_loc[ind + shift];
        }
        shift += ndummy[i];
      }
      indices_loc.resize(ntot - tot_dummy);
      values_loc.resize(ntot - tot_dummy);
    }
    std::vector<unsigned int> sort_permutation(indices_loc.size());
    std::iota(begin(sort_permutation), end(sort_permutation), (unsigned int)0);
    std::sort(begin(sort_permutation), end(sort_permutation), [&values_loc, &compare](const auto& i1, const auto& i2) {
      return compare(values_loc[i1], values_loc[i2]);
    });
    for (size_t i = 0; i < n; ++i) {
      auto j = sort_permutation[i];
      indices_glob[i] = indices_loc[j];
      values_glob[i] = values_loc[j];
    }
  } else {
    auto d = int(n - nmin);
    MPI_Igather(&d, 1, MPI_INT, nullptr, 1, MPI_INT, 0, x.communicator(), &requests[0]);
    MPI_Igather(indices_loc.data(), n, MPI_UNSIGNED_LONG, nullptr, n, MPI_UNSIGNED_LONG, 0, x.communicator(),
                &requests[1]);
    MPI_Igather(values_loc.data(), n, MPI_DOUBLE, nullptr, n, MPI_DOUBLE, 0, x.communicator(), &requests[2]);
    MPI_Waitall(3, requests, MPI_STATUSES_IGNORE);
  }
  MPI_Ibcast(indices_glob.data(), n, MPI_UNSIGNED_LONG, 0, x.communicator(), &requests[0]);
  MPI_Ibcast(values_glob.data(), n, MPI_DOUBLE, 0, x.communicator(), &requests[1]);
  MPI_Waitall(2, requests, MPI_STATUSES_IGNORE);
  auto map_extrema = std::list<std::pair<typename DistrArray<double>::index_type, double>>();
  for (size_t i = 0; i < n; ++i)
    map_extrema.emplace_back(indices_glob[i], values_glob[i]);
  return map_extrema;
}
} // namespace molpro::linalg::array::util
using molpro::linalg::array::DistrArray;
using molpro::linalg::array::util::extrema;
template std::list<std::pair<typename DistrArray<double>::index_type, double>> molpro::linalg::array::util::extrema<std::less< double>>(const DistrArray<double>& x, int n);
template std::list<std::pair<typename DistrArray<double>::index_type, double>> molpro::linalg::array::util::extrema<molpro::linalg::array::util::CompareAbs<double,std::less<>>>(const DistrArray<double>& x, int n);

template std::list<std::pair<typename DistrArray<double>::index_type, double>> molpro::linalg::array::util::extrema<std::greater< double>>(const DistrArray<double>& x, int n);
template std::list<std::pair<typename DistrArray<double>::index_type, double>> molpro::linalg::array::util::extrema<molpro::linalg::array::util::CompareAbs<double,std::greater<>>>(const DistrArray<double>& x, int n);


// TODO sort out template problems so that the following place-holders aren't needed
template <>
std::list<std::pair<DistrArray<double>::index_type, double> > molpro::linalg::array::util::extrema<molpro::linalg::array::util::CompareAbs<double, std::less<void> >, double>(molpro::linalg::array::DistrArray<double> const&, int) {throw std::logic_error("unimplemented");}
template <>
std::__1::list<std::__1::pair<molpro::linalg::array::DistrArray<double>::index_type, molpro::linalg::array::DistrArray<double>::value_type>, std::__1::allocator<std::__1::pair<molpro::linalg::array::DistrArray<double>::index_type, molpro::linalg::array::DistrArray<double>::value_type> > > molpro::linalg::array::util::extrema<std::__1::less<double>, double>(molpro::linalg::array::DistrArray<double> const&, int) {throw std::logic_error("unimplemented");}

template <>
std::__1::list<std::__1::pair<molpro::linalg::array::DistrArray<double>::index_type, molpro::linalg::array::DistrArray<double>::value_type>, std::__1::allocator<std::__1::pair<molpro::linalg::array::DistrArray<double>::index_type, molpro::linalg::array::DistrArray<double>::value_type> > > molpro::linalg::array::util::extrema<molpro::linalg::array::util::CompareAbs<double, std::__1::greater<void> >, double>(molpro::linalg::array::DistrArray<double> const&, int) {throw std::logic_error("unimplemented");}
template <>
std::__1::list<std::__1::pair<molpro::linalg::array::DistrArray<double>::index_type, molpro::linalg::array::DistrArray<double>::value_type>, std::__1::allocator<std::__1::pair<molpro::linalg::array::DistrArray<double>::index_type, molpro::linalg::array::DistrArray<double>::value_type> > > molpro::linalg::array::util::extrema<std::__1::greater<double>, double>(molpro::linalg::array::DistrArray<double> const&, int) {throw std::logic_error("unimplemented");}

template class molpro::linalg::array::DistrArray<double>;
