#include "DistrArrayDisk.h"
#include "util.h"
#include "util/Distribution.h"
#include "util/gemm.h"
#include <future>
#include <iostream>

namespace molpro::linalg::array {
using util::Task;
namespace {

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

DistrArrayDisk::DistrArrayDisk(std::unique_ptr<Distribution> distr, MPI_Comm commun)
    : DistrArray(distr->border().second, commun), m_distribution(std::move(distr)) {}

DistrArrayDisk::DistrArrayDisk() = default;

DistrArrayDisk::DistrArrayDisk(const DistrArray& source)
    : DistrArray(source), m_distribution(std::make_unique<Distribution>(source.distribution())) {
  copy(source);
}

DistrArrayDisk::DistrArrayDisk(DistrArrayDisk&& source) noexcept
    : DistrArray(source), m_distribution(std::move(source.m_distribution)) {
  using std::swap;
  if (source.m_allocated) {
    swap(m_allocated, source.m_allocated);
  }
}

DistrArrayDisk::~DistrArrayDisk() = default;

void DistrArrayDisk::copy(const DistrArray& y) {
  auto name = std::string{"Array::copy"};
  if (!compatible(y))
    error(name + " incompatible arrays");
  auto loc_y = y.local_buffer();
  auto range = m_distribution->range(mpi_rank(m_communicator));
  put(range.first, range.second, loc_y->data());
}

DistrArrayDisk::LocalBufferDisk::LocalBufferDisk(DistrArrayDisk& source) : m_source{source} {
  //  std::cout << "DistrArrayDisk::LocalBufferDisk::LocalBufferDisk(DistrArrayDisk& source)  " << this << std::endl;
  int rank = mpi_rank(source.communicator());
  index_type hi;
  std::tie(m_start, hi) = source.distribution().range(rank);
  m_size = hi - m_start;
  //  std::cout << "LocalBufferDisk " << this << " resizes from  " << m_snapshot_buffer.size() << " to " << m_size
  //            << std::endl;
  m_snapshot_buffer.resize(m_size);
  m_buffer = &m_snapshot_buffer[0];
  source.get(start(), start() + size(), m_buffer);
}

DistrArrayDisk::LocalBufferDisk::LocalBufferDisk(DistrArrayDisk& source, const Span<value_type>& buffer)
    : m_source{source} {
  int rank = mpi_rank(source.communicator());
  index_type hi;
  std::tie(m_start, hi) = source.distribution().range(rank);
  m_size = hi - m_start;
  if (m_size > buffer.size())
    source.error("LocalBufferDisk(): attempting to construct from a buffer that is too small");
  m_buffer = buffer.empty() ? nullptr : &buffer[0];
  source.get(start(), start() + size(), m_buffer);
}

DistrArrayDisk::LocalBufferDisk::~LocalBufferDisk() {
  //  std::cout << "LocalBufferDisk destructor " << this << ", size = " << m_snapshot_buffer.size() << std::endl;
  if (do_dump)
    m_source.put(start(), start() + size(), m_buffer);
}

const DistrArray::Distribution& DistrArrayDisk::distribution() const {
  if (!m_distribution)
    error("allocate buffer before asking for distribution");
  return *m_distribution;
}

std::unique_ptr<DistrArray::LocalBuffer> DistrArrayDisk::local_buffer() {
  // TODO implement this, and the three following, when all code that calls this is removed
  //  throw std::logic_error("local_buffer created (this should never happen)");
  return std::make_unique<LocalBufferDisk>(*this);
}

std::unique_ptr<const DistrArray::LocalBuffer> DistrArrayDisk::local_buffer() const {
  //  throw std::logic_error("local_buffer created (this should never happen)");
  auto l = std::make_unique<LocalBufferDisk>(*const_cast<DistrArrayDisk*>(this));
  l->do_dump = false;
  return l;
}

std::unique_ptr<DistrArray::LocalBuffer> DistrArrayDisk::local_buffer(const Span<value_type>& buffer) {
  //  throw std::logic_error("local_buffer created (this should never happen)");
  return std::make_unique<LocalBufferDisk>(*this, buffer);
}
std::unique_ptr<const DistrArray::LocalBuffer> DistrArrayDisk::local_buffer(const Span<value_type>& buffer) const {
  //  throw std::logic_error("local_buffer created (this should never happen)");
  auto l = std::make_unique<LocalBufferDisk>(*const_cast<DistrArrayDisk*>(this), buffer);
  l->do_dump = false;
  return l;
}

DistrArray::value_type DistrArrayDisk::dot(const DistrArrayDisk& y) const {
  if (&y == this) {
    throw std::invalid_argument("Cannot dot a DistrArrayDisk with itself");
  }
  return DistrArray::dot(y); // TODO: implement buffering in both DistrArrays
}

DistrArray::value_type DistrArrayDisk::dot(const DistrArray& y) const {
  auto y_cvec = molpro::linalg::itsolv::CVecRef<DistrArray>{{y}};
  auto this_cvec = molpro::linalg::itsolv::CVecRef<DistrArray>{{*this}};
  auto result = molpro::linalg::array::util::gemm_inner_distr_distr(y_cvec, this_cvec)(0, 0);
  return result;
}

DistrArray::value_type DistrArrayDisk::dot(const DistrArray::SparseArray& y) const { return DistrArray::dot(y); }

// BufferManager class functions

BufferManager::BufferManager(const DistrArrayDisk& distr_array_disk, Span<DistrArray::value_type> buffer,
                             BufferManager::buffertype number_of_buffers, bool aggregated_async)
    : chunk_size(std::move(buffer.size() / number_of_buffers)), distr_array_disk(distr_array_disk),
      next_chunk_futures(number_of_buffers), range(distr_array_disk.distribution().range(molpro::mpi::rank_global())), m_aggregated_async(aggregated_async) {
  auto prof = molpro::Profiler::single()->push("BufferManager()");
  for (size_t buffer_count = 0; buffer_count < number_of_buffers; ++buffer_count)
    this->chunks.emplace_back(buffer.data() + (chunk_size * buffer_count));
}

BufferManager::BufferManager(const DistrArrayDisk& distr_array_disk, size_t chunk_size,
                             BufferManager::buffertype number_of_buffers, bool aggregated_async)
    : chunk_size(std::move(chunk_size)), distr_array_disk(distr_array_disk), next_chunk_futures(number_of_buffers),
      range(distr_array_disk.distribution().range(molpro::mpi::rank_global())), m_aggregated_async(aggregated_async) {
  own_buffer.reserve(chunk_size * number_of_buffers);
  for (size_t buffer_count = 0; buffer_count < number_of_buffers; ++buffer_count)
    this->chunks.emplace_back(own_buffer.data() + (chunk_size * buffer_count) / number_of_buffers);
}
template <class T>
inline void DistrArrayGet(const T& obj, size_t lo, size_t hi, typename T::value_type* data) {
  obj.get(lo, hi, data);
}
Span<BufferManager::value_type> BufferManager::next(bool initial) {
  auto prof = molpro::Profiler::single()->push(std::string{"BufferManager::next("} + (initial ? "T" : "F") + ")");
  if (initial)
    curr_chunk = 0;
  m_read_parameters[0].m_buffer = chunks[curr_chunk%chunks.size()];
   m_read_parameters[0].m_offset = range.first + curr_chunk * this->chunk_size;
   m_read_parameters[0].m_high = std::min(m_read_parameters[0].m_offset + this->chunk_size, range.second);
   m_read_parameters[1].m_offset = range.first + (curr_chunk + 1) * this->chunk_size;
   m_read_parameters[1].m_high = std::min(m_read_parameters[1].m_offset + this->chunk_size, range.second);
   m_read_parameters[1].m_buffer = chunks[(curr_chunk + 1) % chunks.size()];
    const auto buffer_id = curr_chunk % chunks.size();
  DistrArray::value_type* buffer = chunks[buffer_id];


  {
    if (m_read_parameters[0].m_offset >= range.second)
      return Span<BufferManager::value_type>(nullptr, 0);
    if (chunks.size() == 1 or m_read_parameters[0].m_offset == range.first) {
//        this->distr_array_disk.get(m_read_parameters[0].m_offset, m_read_parameters[0].m_high, buffer);
        read(0);
    } else {
      this->next_chunk_futures[buffer_id].wait();
    }
  }

    if (chunks.size() > 1 and m_read_parameters[1].m_offset < range.second) {
        if (m_aggregated_async) {
  } else
    {
        this->next_chunk_futures.at((curr_chunk + 1) % chunks.size()) =
            std::async(std::launch::async, [this]() {
//              this->distr_array_disk.get(m_read_parameters[1].m_offset, m_read_parameters[1].m_high, m_read_parameters[1].m_buffer);
              read(1);
            });
      }
    }

  ++curr_chunk;
  return Span<value_type>(buffer, m_read_parameters[0].m_high-m_read_parameters[0].m_offset);
}

    void BufferManager::read(int buffer) {
        this->distr_array_disk.get(m_read_parameters[buffer].m_offset, m_read_parameters[buffer].m_high, m_read_parameters[buffer].m_buffer);
    }

} // namespace molpro::linalg::array
