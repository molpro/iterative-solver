#include <unistd.h>

#include <molpro/linalg/array/DistrArrayFile.h>
#include <molpro/linalg/array/util/Distribution.h>
#include <molpro/linalg/array/util/temp_file.h>
#include <utility>

#include <memory>
#include <molpro/Profiler.h>

namespace molpro::linalg::array {
namespace {
int mpi_size(MPI_Comm comm) {
  if (comm == molpro::mpi::comm_global())
    return molpro::mpi::size_global();
#ifdef HAVE_MPI_H
  int size;
  MPI_Comm_size(comm, &size);
  return size;
#endif
  throw std::logic_error("Attempt to access MPI communicator in serial mode");
}

int mpi_rank(MPI_Comm comm) {
  if (comm == molpro::mpi::comm_global())
    return molpro::mpi::rank_global();
#ifdef HAVE_MPI_H
  int rank;
  MPI_Comm_rank(comm, &rank);
  return rank;
#endif
  throw std::logic_error("Attempt to access MPI communicator in serial mode");
}
} // namespace

std::unique_ptr<util::FileAttributes> DistrArrayFile::file = nullptr;

std::tuple<DistrArrayFile::index_type, DistrArrayFile::index_type, DistrArrayFile::index_type>
DistrArrayFile::local_bounds() const {
  index_type lo_loc, hi_loc, size_loc;
  std::tie(lo_loc, hi_loc) = m_distribution->range(mpi_rank(m_communicator));
  size_loc = hi_loc - lo_loc;
  return {lo_loc, hi_loc, size_loc};
}

void DistrArrayFile::update_records() {
  if (std::get<2>(local_bounds()) == 0)
    return;
  if (file->registry.empty()) {
    m_frecs = {0, std::get<2>(local_bounds()) - 1};
    file->registry.emplace(m_frecs);
  } else {
    std::set<std::pair<size_t, size_t>>::reverse_iterator rec;
    for (rec = file->registry.rbegin(); rec != file->registry.rend(); ++rec) {
      if ((*rec).first != 0 && (*rec).first - (*std::next(rec)).second > 1) {
        rec = std::next(rec);
        break;
      };
      if ((*rec).first == 0) {
        rec = file->registry.rbegin();
        break;
      }
    }
    size_t first_rec = (*rec).second + 1;
    m_frecs = {first_rec, first_rec + std::get<2>(local_bounds()) - 1};
    file->registry.emplace(m_frecs);
  }
  m_lrec = true;
}

DistrArrayFile::DistrArrayFile(DistrArrayFile&& source) noexcept
    : DistrArrayDisk(std::move(source)), m_frecs(std::move(source.m_frecs)) {
  if (source.m_lrec) {
    source.m_lrec = false;
    m_lrec = true;
  }
}

DistrArrayFile::DistrArrayFile(size_t dimension, MPI_Comm comm, const std::string& directory)
    : DistrArrayFile(std::make_unique<Distribution>(
                         util::make_distribution_spread_remainder<index_type>(dimension, mpi_size(comm))),
                     comm, directory) {}

DistrArrayFile::DistrArrayFile(std::unique_ptr<Distribution> distribution, MPI_Comm comm, const std::string& directory)
: DistrArrayDisk(std::move(distribution), comm) {
  auto prof = molpro::Profiler::single()->push("DistrArrayFile()");
  if (m_distribution->border().first != 0)
    DistrArray::error("Distribution of array must start from 0");
  if (!file) {
    file = std::make_unique<util::FileAttributes>(make_file(fs::absolute(fs::path(directory))));
  }
  update_records();
}

DistrArrayFile::DistrArrayFile(const DistrArrayFile& source)
    : DistrArrayFile(std::make_unique<Distribution>(source.distribution()), source.communicator()) {
  DistrArrayFile::copy(source);
}

DistrArrayFile::DistrArrayFile(const DistrArray& source)
    : DistrArrayFile(std::make_unique<Distribution>(source.distribution()), source.communicator()) {
  DistrArrayFile::copy(source);
}

DistrArrayFile& DistrArrayFile::operator=(DistrArrayFile&& source) noexcept {
  DistrArrayFile t{std::move(source)};
  swap(*this, t);
  return *this;
}

DistrArrayFile DistrArrayFile::CreateTempCopy(const DistrArray& source, const std::string& directory) {
  DistrArrayFile t(std::make_unique<Distribution>(source.distribution()), source.communicator(), directory);
  t.copy(source);
  return t;
}

void swap(DistrArrayFile& x, DistrArrayFile& y) noexcept {
  using std::swap;
  swap(x.m_dimension, y.m_dimension);
  swap(x.m_communicator, y.m_communicator);
  swap(x.m_allocated, y.m_allocated);
  swap(x.m_distribution, y.m_distribution);
  swap(x.m_frecs, y.m_frecs);
  swap(x.m_lrec, y.m_lrec);
}

DistrArrayFile::~DistrArrayFile() {
  if (file) {
    if (m_lrec) {
      file->registry.erase(m_frecs);
      if (file->registry.empty())
        file.reset();
    }
  }
}

bool DistrArrayFile::compatible(const DistrArrayFile& source) const {
  auto res = DistrArray::compatible(source);
  if (m_distribution && source.m_distribution)
    res &= m_distribution->compatible(*source.m_distribution);
  else
    res &= !m_distribution && !source.m_distribution;
  return res;
}

std::fstream DistrArrayFile::make_file(const fs::path& dir) {
  auto prof = molpro::Profiler::single()->push("DistrArrayFile::make_file()");
  std::fstream tfile;
  std::string file_name = util::temp_file_name(dir.string() + "/", "");
  tfile.open(file_name.c_str(), std::ios::out | std::ios::binary);
  tfile.close();
  tfile.open(file_name.c_str(), std::ios::out | std::ios::in | std::ios::binary);
  unlink(file_name.c_str());
  return tfile;
}

DistrArray::value_type DistrArrayFile::at(DistrArray::index_type ind) const {
  value_type val;
  get(ind, ind + 1, &val);
  return val;
}

void DistrArrayFile::set(DistrArray::index_type ind, DistrArray::value_type val) { put(ind, ind + 1, &val); }

void DistrArrayFile::get(DistrArray::index_type lo, DistrArray::index_type hi, DistrArray::value_type* buf) const {
  if (lo >= hi)
    return;
  auto prof = molpro::Profiler::single()->push("DistrArrayFile::get()");
  prof += hi - lo;
  DistrArray::index_type length = hi - lo;
  DistrArray::index_type lo_loc, hi_loc;
  auto bounds_loc = local_bounds();
  std::tie(lo_loc, hi_loc) = {std::get<0>(bounds_loc), std::get<1>(bounds_loc)};
  if (lo < lo_loc || hi > hi_loc) {
    error("Only local array indices can be accessed via DistrArrayFile.get() function");
  }
  DistrArray::index_type offset = m_frecs.first + lo - lo_loc;
  file->object.seekg(0, std::ios::end);
  int current = file->object.tellg();
  if (current < (offset + length) * sizeof(DistrArray::value_type)) {
    return;
  }
  file->object.seekg(offset * sizeof(DistrArray::value_type));
  file->object.read((char*)buf, length * sizeof(DistrArray::value_type));
}

std::vector<DistrArrayFile::value_type> DistrArrayFile::get(DistrArray::index_type lo,
                                                            DistrArray::index_type hi) const {
  if (lo >= hi)
    return {};
  auto buf = std::vector<DistrArray::value_type>(hi - lo);
  get(lo, hi, &buf[0]);
  return buf;
}

void DistrArrayFile::put(DistrArray::index_type lo, DistrArray::index_type hi, const DistrArray::value_type* data) {
  if (lo >= hi)
    return;
  auto prof = molpro::Profiler::single()->push("DistrArrayFile::put()");
  prof += hi - lo;
  DistrArray::index_type lo_loc, hi_loc;
  auto bounds_loc = local_bounds();
  std::tie(lo_loc, hi_loc) = {std::get<0>(bounds_loc), std::get<1>(bounds_loc)};
  if (lo < lo_loc || hi > hi_loc) {
    error("Only values at local array indices can be written via DistrArrayFile.put() function");
  }
  DistrArray::index_type offset = m_frecs.first + lo - lo_loc;
  DistrArray::index_type length = hi - lo;
  file->object.seekp(offset * sizeof(DistrArray::value_type));
  file->object.write((const char*)data, length * sizeof(DistrArray::value_type));
}

void DistrArrayFile::acc(DistrArray::index_type lo, DistrArray::index_type hi, const DistrArray::value_type* data) {
  if (lo >= hi)
    return;
  auto prof = molpro::Profiler::single()->push("DistrArrayFile::acc()");
  auto disk_copy = get(lo, hi);
  std::transform(disk_copy.begin(), disk_copy.end(), data, disk_copy.begin(), [](auto& l, auto& r) { return l + r; });
  put(lo, hi, &disk_copy[0]);
}

std::vector<DistrArrayFile::value_type> DistrArrayFile::gather(const std::vector<index_type>& indices) const {
  std::vector<value_type> data;
  data.reserve(indices.size());
  auto minmax = std::minmax_element(indices.begin(), indices.end());
  DistrArray::index_type lo_loc, hi_loc;
  auto bounds_loc = local_bounds();
  std::tie(lo_loc, hi_loc) = {std::get<0>(bounds_loc), std::get<1>(bounds_loc)};
  if (*minmax.first < lo_loc || *minmax.second > hi_loc) {
    error("Only local array indices can be accessed via DistrArrayFile.gather() function");
  }
  for (auto i : indices) {
    data.push_back(at(i));
  }
  return data;
}

void DistrArrayFile::scatter(const std::vector<index_type>& indices, const std::vector<value_type>& data) {
  if (indices.size() != data.size()) {
    error("Length of the indices and data vectors should be the same: DistrArray::scatter()");
  }
  auto prof = molpro::Profiler::single()->push("DistrArrayFile::scatter()");
  auto minmax = std::minmax_element(indices.begin(), indices.end());
  DistrArray::index_type lo_loc, hi_loc;
  auto bounds_loc = local_bounds();
  std::tie(lo_loc, hi_loc) = {std::get<0>(bounds_loc), std::get<1>(bounds_loc)};
  if (*minmax.first < lo_loc || *minmax.second > hi_loc) {
    error("Only local array indices can be accessed via DistrArrayFile.gather() function");
  }
  for (auto i : indices) {
    set(i, data[i - *minmax.first]); // TODO: check it shouldn't be data[*minmax.first++]??
  }
}

void DistrArrayFile::scatter_acc(std::vector<index_type>& indices, const std::vector<value_type>& data) {
  auto disk_copy = gather(indices);
  std::transform(data.begin(), data.end(), disk_copy.begin(), disk_copy.begin(),
                 [](auto& l, auto& r) { return l + r; });
  scatter(indices, disk_copy);
}

std::vector<DistrArrayFile::value_type> DistrArrayFile::vec() const {
  DistrArray::index_type lo_loc, hi_loc;
  auto bounds_loc = local_bounds();
  std::tie(lo_loc, hi_loc) = {std::get<0>(bounds_loc), std::get<1>(bounds_loc)};
  return get(lo_loc, hi_loc);
}

} // namespace molpro::linalg::array
