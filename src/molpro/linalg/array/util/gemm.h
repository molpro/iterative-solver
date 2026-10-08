#ifndef LINEARALGEBRA_SRC_MOLPRO_LINALG_ARRAY_UTIL_GEMM_H
#define LINEARALGEBRA_SRC_MOLPRO_LINALG_ARRAY_UTIL_GEMM_H
#ifdef HAVE_MPI_H
#include <mpi.h>
#endif
#include "BufferManager.h"
#include <future>
#include <iostream>
#include <molpro/Options.h>
#include <molpro/Profiler.h>
#include <molpro/cblas.h>
#include <molpro/linalg/array/DistrArrayDisk.h>
#include <molpro/linalg/array/type_traits.h>
#include <molpro/linalg/itsolv/subspace/Matrix.h>
#include <molpro/linalg/itsolv/wrap.h>
#include <molpro/linalg/itsolv/wrap_util.h>
#include <molpro/linalg/options.h>
#include <numeric>
#include <vector>

namespace molpro::linalg::array::util {

using molpro::linalg::itsolv::CVecRef;
using molpro::linalg::itsolv::VecRef;
using molpro::linalg::itsolv::subspace::Matrix;

enum gemm_type { inner, outer };

//! Whether arrays of type A are held on disk, and so must be read through buffers rather than local_buffer()
template <class A>
inline constexpr bool is_disk_array_v = std::is_base_of_v<DistrArrayDisk, std::decay_t<A>>;

//! The section [lo, hi) of a distributed array held by this process
template <class A>
auto local_range(const A& a) {
  return a.distribution().range(BufferManager<std::decay_t<A>>::rank_in(a));
}

// Buffered: the disk arrays xx are read in chunks through a BufferManager

template <class AL, class AD, std::enable_if_t<is_disk_array_v<AD>, int> = 0>
Matrix<typename array::mapped_or_value_type_t<AL>> gemm_inner_distr_distr(const CVecRef<AL>& yy,
                                                                          const CVecRef<AD>& xx) {
  auto prof = molpro::Profiler::single()->push("gemm_inner_distr_distr (buffered)");
  if (not yy.empty())
    prof += xx.size() * yy.size() * (local_range(yy[0].get()).second - local_range(yy[0].get()).first) * 2;
  using value_type = typename array::mapped_or_value_type_t<AL>;
  auto alphas = Matrix<value_type>({yy.size(), xx.size()});
  alphas.fill(0);
  auto non_const_yy = molpro::linalg::itsolv::const_cast_wrap(yy);
  auto alphadata = const_cast<value_type*>(alphas.data().data());
  gemm_distr_distr(alphadata, xx, non_const_yy, gemm_type::inner);
#ifdef HAVE_MPI_H
  // reduce over the arrays' own communicator, which need not be the global one
  if (alphas.size() > 0)
    MPI_Allreduce(MPI_IN_PLACE, alphadata, alphas.size(), MPI_DOUBLE, MPI_SUM, xx.front().get().communicator());
#endif
  return alphas;
}

template <class AD, class AL, std::enable_if_t<is_disk_array_v<AD> and not is_disk_array_v<AL>, int> = 0>
Matrix<typename array::mapped_or_value_type_t<AL>> gemm_inner_distr_distr(const CVecRef<AD>& xx,
                                                                          const CVecRef<AL>& yy) {
  // gemm_inner is the hermitian inner product, so the two orders are conjugate transposes
  auto result_transpose = gemm_inner_distr_distr(yy, xx);
  Matrix<typename array::mapped_or_value_type_t<AL>> result({result_transpose.cols(), result_transpose.rows()});
  molpro::linalg::itsolv::subspace::conjugate_transpose_copy(result, result_transpose);
  return result;
}

template <class AL, class AD, std::enable_if_t<is_disk_array_v<AD>, int> = 0>
void gemm_outer_distr_distr(const Matrix<typename array::mapped_or_value_type_t<AL>> alphas, const CVecRef<AD>& xx,
                            const VecRef<AL>& yy) {
  if (yy.empty() or xx.empty())
    return;
  auto prof = molpro::Profiler::single()->push("gemm_outer_distr_distr (buffered)");
  if (not yy.empty())
    prof += xx.size() * yy.size() * (local_range(yy[0].get()).second - local_range(yy[0].get()).first) * 2;
  if (alphas.rows() != xx.size())
    throw std::out_of_range(std::string{"gemm_outer_distr_distr: dimensions of xx and alphas are different: "} +
                            std::to_string(alphas.rows()) + " " + std::to_string(xx.size()));
  if (alphas.cols() != yy.size())
    throw std::out_of_range(std::string{"gemm_outer_distr_distr: dimensions of yy and alphas are different: "} +
                            std::to_string(alphas.cols()) + " " + std::to_string(yy.size()));
  gemm_distr_distr(const_cast<typename array::mapped_or_value_type_t<AL>*>(alphas.data().data()), xx, yy,
                   gemm_type::outer);
}

template <class AL, class AD>
void gemm_distr_distr(array::mapped_or_value_type_t<AL>* alphadata, const CVecRef<AD>& xx, const VecRef<AL>& yy,
                      gemm_type gemm_type) {

  auto prof = molpro::Profiler::single()->push("gemm_distr_distr");
  if (xx.size() == 0 || yy.size() == 0) {
    return;
  }

  // yy may be held in memory, and addressed in place, or on disk, when each chunk of it is read into yy_chunk (and for
  // gemm_outer written back afterwards)
  using value_type = array::mapped_or_value_type_t<AL>;
  const bool yy_on_disk = yy.front().get().disk_page_size() > 0;
  const auto [lo, hi] = local_range(xx.front().get());
  std::vector<value_type> yy_chunk;
  bool yy_constant_stride = true;
  int yy_stride = hi - lo;
  if (not yy_on_disk) {
    int previous_stride = 0;
    for (size_t j = 0; j < std::max((size_t)1, yy.size()) - 1; ++j) {
      auto unique_ptr_j = yy.at(j).get().local_buffer()->data();
      auto unique_ptr_jp1 = yy.at(j + 1).get().local_buffer()->data();
      yy_stride = unique_ptr_jp1 - unique_ptr_j;
      //        std::cout << "j="<<j<<" yy_stride="<<yy_stride<<std::endl;
      if (j > 0)
        yy_constant_stride = yy_constant_stride && (yy_stride == previous_stride);
      previous_stride = yy_stride;
    }
    yy_constant_stride = yy_constant_stride && (yy_stride > 0);
  }

  auto options = molpro::linalg::options();
  auto number_of_buffers = options->parameter("GEMM_BUFFERS", 2);
  // BufferManager's buffer_size parameter is the size of ONE chunk (it already multiplies
  // by number_of_buffers internally to size the total pool) -- do not multiply by
  // number_of_buffers again here, or each chunk silently grows number_of_buffers-fold.
  const int buf_size = std::min(int(hi - lo), options->parameter("GEMM_PAGESIZE", 8192));
//      std::cout << "buf_size=" << buf_size << " number_of_buffers=" << number_of_buffers << std::endl;

  molpro::Profiler::single()->start("gemm: buffer setup");
  BufferManager buffer(xx, buf_size, number_of_buffers);
  molpro::Profiler::single()->stop("gemm: buffer setup");
  for (auto buffer_iterator = buffer.begin(); buffer_iterator != buffer.end(); ++buffer_iterator) {
    auto container_offset = buffer.buffer_offset();
    int current_buf_size = buffer.buffer_size();
//    std::cout << "container_offset="<<container_offset<<", current_buf_size="<<current_buf_size<<std::endl;
    value_type* yy_data;
    if (yy_on_disk) {
      yy_chunk.resize(yy.size() * current_buf_size);
      for (size_t j = 0; j < yy.size(); ++j)
        yy[j].get().get(lo + container_offset, lo + container_offset + current_buf_size,
                        yy_chunk.data() + j * current_buf_size);
      yy_data = yy_chunk.data();
      yy_stride = current_buf_size;
    } else
      yy_data = yy[0].get().local_buffer()->data() + container_offset;
    if (gemm_type == gemm_type::outer) {
      if (yy_constant_stride and not yy.empty()) {
        auto prof =
            molpro::Profiler::single()->push("gemm_outer: cblas_dgemm dimensions " + std::to_string(xx.size()) + ", " +
                                             std::to_string(yy.size()) + ", " + std::to_string(current_buf_size));
//        std::cout << "outer dgemm container_offset="<<container_offset<<std::endl;
        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans, current_buf_size, yy.size(), xx.size(), 1,
                    buffer_iterator->data(), buffer.buffer_stride(), alphadata, yy.size(), 1, yy_data, yy_stride);
      } else { // non-uniform stride:
        auto prof =
            molpro::Profiler::single()->push("gemm_outer: cblas_dgemv dimensions " + std::to_string(xx.size()) + ", " +
                                             std::to_string(yy.size()) + ", " + std::to_string(current_buf_size));
//        std::cout << "outer dgemv"<<std::endl;
        for (size_t i = 0; i < yy.size(); ++i) {
          cblas_dgemv(CblasColMajor, CblasNoTrans, current_buf_size, xx.size(), 1, buffer_iterator->data(), buffer.buffer_stride(),
                      alphadata + i, yy.size(), 1, yy[i].get().local_buffer()->data() + container_offset, 1);
        }
      }
    } else if (gemm_type == gemm_type::inner) {
      if (yy_constant_stride and not yy.empty()) {
        auto prof =
            molpro::Profiler::single()->push("gemm_inner: cblas_dgemm dimensions " + std::to_string(xx.size()) + ", " +
                                             std::to_string(yy.size()) + ", " + std::to_string(current_buf_size));
//        std::cout << "inner dgemm"<<std::endl;
        cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, xx.size(), yy.size(), current_buf_size, 1,
                    buffer_iterator->data(), buffer.buffer_stride(), yy_data, yy_stride, 1, alphadata, xx.size());
      } else { // non-uniform stride:
        auto prof =
            molpro::Profiler::single()->push("gemm_inner: cblas_dgemv dimensions " + std::to_string(xx.size()) + ", " +
                                             std::to_string(yy.size()) + ", " + std::to_string(current_buf_size));
//        std::cout << "inner dgemv"<<std::endl;
        for (size_t k = 0; k < yy.size(); ++k) {
          cblas_dgemv(CblasColMajor, CblasTrans, current_buf_size, xx.size(), 1, buffer_iterator->data(), buffer.buffer_stride(),
                      yy[k].get().local_buffer()->data() + container_offset, 1, 1, alphadata + k * xx.size(), 1);
        }
      }
    }
    if (yy_on_disk and gemm_type == gemm_type::outer)
      for (size_t j = 0; j < yy.size(); ++j)
        yy[j].get().put(lo + container_offset, lo + container_offset + current_buf_size,
                        yy_chunk.data() + j * current_buf_size);
  }
}

// Without buffers, for arrays held in memory

template <class AL, class AR = AL, std::enable_if_t<not is_disk_array_v<AL> and not is_disk_array_v<AR>, int> = 0>
Matrix<typename array::mapped_or_value_type_t<AL>> gemm_inner_distr_distr(const CVecRef<AL>& xx,
                                                                          const CVecRef<AR>& yy) {
  // const size_t spacing = 1;
  using value_type = typename array::mapped_or_value_type_t<AL>;
  auto mat = Matrix<value_type>({xx.size(), yy.size()});
  if (xx.size() == 0 || yy.size() == 0)
    return mat;
  auto prof = molpro::Profiler::single()->push("gemm_inner_distr_distr (unbuffered)");
  if (not xx.empty())
    prof += mat.cols() * mat.rows() * xx.at(0).get().local_buffer()->size() * 2;
  for (size_t j = 0; j < mat.cols(); ++j) {
    auto loc_y = yy.at(j).get().local_buffer();
    for (size_t i = 0; i < mat.rows(); ++i) {
      auto loc_x = xx.at(i).get().local_buffer();
      mat(i, j) = std::inner_product(begin(*loc_x), end(*loc_x), begin(*loc_y), (value_type)0);
      // mat(i,j) += cblas_ddot(end(*loc_x) - begin(*loc_x), begin(*loc_x), spacing, begin(*loc_y), spacing);
    }
  }
#ifdef HAVE_MPI_H
  MPI_Allreduce(MPI_IN_PLACE, const_cast<value_type*>(mat.data().data()), mat.size(), MPI_DOUBLE, MPI_SUM,
                xx.at(0).get().communicator());
#endif
  return mat;
}

template <class AL, class AR = AL, std::enable_if_t<not is_disk_array_v<AR>, int> = 0>
void gemm_outer_distr_distr(const Matrix<typename array::mapped_or_value_type_t<AL>> alphas, const CVecRef<AR>& xx,
                            const VecRef<AL>& yy) {
  if (is_disk_array_v<AL>) {
    throw std::runtime_error("gemm_outer_distr_distr (unbuffered) called to update disk arrays (should never happen!)");
  }
  auto prof = molpro::Profiler::single()->push("gemm_outer_distr_distr (unbuffered)");
  if (not yy.empty())
    prof += alphas.rows() * alphas.cols() * yy[0].get().local_buffer()->size() * 2;
  for (size_t ii = 0; ii < alphas.rows(); ++ii) {
    auto loc_x = xx.at(ii).get().local_buffer();
    for (size_t jj = 0; jj < alphas.cols(); ++jj) {
      auto loc_y = yy[jj].get().local_buffer();
      for (size_t i = 0; i < loc_y->size(); ++i)
        (*loc_y)[i] += alphas(ii, jj) * (*loc_x)[i];
    }
  }
}

// Sparse

template <class AL, class AR = AL>
void gemm_outer_distr_sparse(const Matrix<typename array::mapped_or_value_type_t<AL>> alphas, const CVecRef<AR>& xx,
                             const VecRef<AL>& yy) {
  // combine the sparse contributions to each yy, so that it is updated in one pass even when it is held on disk
  for (size_t ii = 0; ii < alphas.cols(); ++ii) {
    auto combined = std::decay_t<AR>{};
    for (size_t jj = 0; jj < alphas.rows(); ++jj)
      for (const auto& [i, v] : xx.at(jj).get())
        combined[i] += alphas(jj, ii) * v;
    yy[ii].get().axpy(1, combined);
  }
}

template <class AL, class AR = AL>
Matrix<typename array::mapped_or_value_type_t<AL>> gemm_inner_distr_sparse(const CVecRef<AL>& xx,
                                                                           const CVecRef<AR>& yy) {
  using value_type = typename array::mapped_or_value_type_t<AL>;
  auto mat = Matrix<value_type>({xx.size(), yy.size()});
  if (xx.size() == 0 || yy.size() == 0)
    return mat;
  const auto [lo, hi] = local_range(xx.at(0).get());
  for (size_t i = 0; i < mat.rows(); ++i) {
    const auto& x = xx.at(i).get();
    // an array in memory is read in place; one on disk has only the elements needed read
    const bool on_disk = x.disk_page_size() > 0;
    decltype(x.local_buffer()) loc_x;
    if (not on_disk)
      loc_x = x.local_buffer();
    for (size_t j = 0; j < mat.cols(); ++j) {
      mat(i, j) = 0;
      for (auto it = yy.at(j).get().lower_bound(lo); it != yy.at(j).get().end() and it->first < hi; ++it) {
        value_type xk;
        if (on_disk)
          x.get(it->first, it->first + 1, &xk);
        else
          xk = (*loc_x)[it->first - lo];
        mat(i, j) += xk * it->second;
      }
    }
  }
#ifdef HAVE_MPI_H
  MPI_Allreduce(MPI_IN_PLACE, const_cast<value_type*>(mat.data().data()), mat.size(), MPI_DOUBLE, MPI_SUM,
                xx.at(0).get().communicator());
#endif
  return mat;
}

// Handlers

template <class Handler, class AL, class AR = AL>
void gemm_outer_default(Handler& handler, const Matrix<typename Handler::value_type> alphas, const CVecRef<AR>& xx,
                        const VecRef<AL>& yy) {
  for (size_t ii = 0; ii < alphas.rows(); ++ii) {
    for (size_t jj = 0; jj < alphas.cols(); ++jj) {
      handler.axpy(alphas(ii, jj), xx.at(ii).get(), yy[jj].get());
    }
  }
}

template <class Handler, class AL, class AR = AL>
Matrix<typename Handler::value_type> gemm_inner_default(Handler& handler, const CVecRef<AL>& xx,
                                                        const CVecRef<AR>& yy) {
  auto mat = Matrix<typename Handler::value_type>({xx.size(), yy.size()});
  if (xx.size() == 0 || yy.size() == 0)
    return mat;
  for (size_t ii = 0; ii < mat.rows(); ++ii) {
    for (size_t jj = 0; jj < mat.cols(); ++jj) {
      mat(ii, jj) = handler.dot(xx.at(ii).get(), yy.at(jj).get());
    }
  }
  return mat;
}

} // namespace molpro::linalg::array::util

#endif // LINEARALGEBRA_SRC_MOLPRO_LINALG_ARRAY_UTIL_GEMM_H
