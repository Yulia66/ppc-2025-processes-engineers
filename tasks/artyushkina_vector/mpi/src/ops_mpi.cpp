#include "artyushkina_vector/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cstddef>
#include <utility>
#include <vector>

#ifdef __GNUC__
#  pragma GCC diagnostic push
#  pragma GCC diagnostic ignored "-Wnull-dereference"
#endif

namespace artyushkina_vector {

VerticalStripMatVecMPI::VerticalStripMatVecMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = OutType{};
}

bool VerticalStripMatVecMPI::ValidationImpl() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if (rank != 0) {
    return true;
  }

  const auto &[matrix, vector] = GetInput();

  if (matrix.empty() || vector.empty()) {
    return false;
  }

  size_t cols = matrix[0].size();

  for (size_t i = 1; i < matrix.size(); ++i) {
    if (matrix[i].size() != cols) {
      return false;
    }
  }

  return vector.size() == cols;
}

bool VerticalStripMatVecMPI::PreProcessingImpl() {
  return true;
}

namespace {

void BroadcastDimensions(int &rows, int &cols) {
  MPI_Bcast(&rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&cols, 1, MPI_INT, 0, MPI_COMM_WORLD);
}

std::pair<int, int> GetProcessParams(int proc, int base, int rem) {
  int proc_start = proc * base;
  if (proc < rem) {
    proc_start += proc;
  } else {
    proc_start += rem;
  }

  int proc_width = base;
  if (proc < rem) {
    proc_width += 1;
  }

  return {proc_start, proc_width};
}

void DistributeVectorStripes(int world_size, const Vector &vector, int base, int rem, std::vector<double> &local_vector,
                             int rank, int my_width) {
  constexpr int tag_vector = 101;

  if (rank == 0) {
    for (int proc = 0; proc < world_size; ++proc) {
      auto [proc_start, proc_width] = GetProcessParams(proc, base, rem);

      if (proc_width <= 0) {
        continue;
      }

      std::vector<double> sendbuf(static_cast<size_t>(proc_width));
      for (int j = 0; j < proc_width; ++j) {
        size_t j_idx = static_cast<size_t>(j);
        size_t src_idx = static_cast<size_t>(proc_start) + j_idx;
        sendbuf[j_idx] = vector[src_idx];
      }

      if (proc == 0) {
        local_vector = sendbuf;
      } else {
        MPI_Send(sendbuf.data(), proc_width, MPI_DOUBLE, proc, tag_vector, MPI_COMM_WORLD);
      }
    }
  } else if (my_width > 0) {
    MPI_Recv(local_vector.data(), my_width, MPI_DOUBLE, 0, tag_vector, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  }
}

void MultiplyStrip(const std::vector<double> &matrix_flat, const std::vector<double> &local_vector,
                   std::vector<double> &local_result, int rows, int cols, int my_width, int my_start) {
  for (int i = 0; i < rows; ++i) {
    size_t i_idx = static_cast<size_t>(i);
    for (int j = 0; j < my_width; ++j) {
      size_t j_idx = static_cast<size_t>(j);
      int global_j = my_start + j;
      size_t global_j_idx = static_cast<size_t>(global_j);
      size_t matrix_idx = static_cast<size_t>(i * cols) + global_j_idx;
      local_result[i_idx] += matrix_flat[matrix_idx] * local_vector[j_idx];
    }
  }
}

void GatherResultsInRoot(int world_size, int rows, const std::vector<double> &local_result,
                         std::vector<double> &final_result) {
  constexpr int tag_result = 102;

  for (int i = 0; i < rows; ++i) {
    size_t i_idx = static_cast<size_t>(i);
    final_result[i_idx] = local_result[i_idx];
  }

  for (int proc = 1; proc < world_size; ++proc) {
    std::vector<double> recv_buf(static_cast<size_t>(rows));
    MPI_Recv(recv_buf.data(), rows, MPI_DOUBLE, proc, tag_result, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    for (int i = 0; i < rows; ++i) {
      size_t i_idx = static_cast<size_t>(i);
      final_result[i_idx] += recv_buf[i_idx];
    }
  }
}

void BroadcastFinalResult(int rank, int world_size, const std::vector<double> &final_result,
                          std::vector<double> &local_final_result) {
  constexpr int tag_broadcast = 103;

  if (rank == 0) {
    for (int proc = 1; proc < world_size; ++proc) {
      MPI_Send(final_result.data(), static_cast<int>(final_result.size()), MPI_DOUBLE, proc, tag_broadcast,
               MPI_COMM_WORLD);
    }
  } else {
    MPI_Recv(local_final_result.data(), static_cast<int>(local_final_result.size()), MPI_DOUBLE, 0, tag_broadcast,
             MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  }
}

bool HandleWorldSizeGreaterThanCols(int world_size, int rank, int rows, int cols, std::vector<double> &result,
                                    const InType &input) {
  if (rows > 0) {
    result.resize(static_cast<size_t>(rows), 0.0);
  }

  if (rank == 0) {
    const auto &[matrix, vector] = input;
    for (int i = 0; i < rows; ++i) {
      size_t i_idx = static_cast<size_t>(i);
      for (int j = 0; j < cols; ++j) {
        size_t j_idx = static_cast<size_t>(j);
        result[i_idx] += matrix[i_idx][j_idx] * vector[j_idx];
      }
    }

    for (int proc = 1; proc < world_size; ++proc) {
      MPI_Send(result.data(), rows, MPI_DOUBLE, proc, 100, MPI_COMM_WORLD);
    }
  } else if (rows > 0) {
    MPI_Recv(result.data(), rows, MPI_DOUBLE, 0, 100, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  }

  return true;
}

void PrepareMatrixFlat(int rank, int rows, int cols, std::vector<double> &matrix_flat, const InType &input) {
  if (rank == 0 && rows > 0 && cols > 0) {
    const auto &[matrix, vector] = input;
    for (int i = 0; i < rows; ++i) {
      size_t i_idx = static_cast<size_t>(i);
      for (int j = 0; j < cols; ++j) {
        size_t j_idx = static_cast<size_t>(j);
        size_t matrix_idx = static_cast<size_t>(i * cols) + j_idx;
        matrix_flat[matrix_idx] = matrix[i_idx][j_idx];
      }
    }
  }
}

std::pair<int, int> CalculateDimensions(int rank, const InType &input) {
  int rows = 0;
  int cols = 0;

  if (rank == 0) {
    const auto &[matrix, vector] = input;
    rows = static_cast<int>(matrix.size());
    if (rows > 0) {
      cols = static_cast<int>(matrix[0].size());
    }
  }

  return {rows, cols};
}

}  // namespace

bool VerticalStripMatVecMPI::RunImpl() {
  int world_size = 0;
  int rank = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  const auto &input_data = GetInput();
  auto [rows, cols] = CalculateDimensions(rank, input_data);
  BroadcastDimensions(rows, cols);

  if (rows <= 0 || cols <= 0) {
    GetOutput() = Vector{};
    return true;
  }

  if (world_size > cols) {
    Vector result;
    if (HandleWorldSizeGreaterThanCols(world_size, rank, rows, cols, result, input_data)) {
      GetOutput() = result;
      return true;
    }
    return false;
  }

  int base = cols / world_size;
  int rem = cols % world_size;

  auto [my_start, my_width] = GetProcessParams(rank, base, rem);

  std::vector<double> matrix_flat;
  if (rows > 0 && cols > 0) {
    matrix_flat.resize(static_cast<size_t>(rows) * static_cast<size_t>(cols), 0.0);
  }

  std::vector<double> local_vector;
  if (my_width > 0) {
    local_vector.resize(static_cast<size_t>(my_width), 0.0);
  }

  std::vector<double> local_result;
  if (rows > 0) {
    local_result.resize(static_cast<size_t>(rows), 0.0);
  }

  std::vector<double> final_result;
  if (rows > 0) {
    final_result.resize(static_cast<size_t>(rows), 0.0);
  }

  PrepareMatrixFlat(rank, rows, cols, matrix_flat, input_data);

  if (rows > 0 && cols > 0) {
    MPI_Bcast(matrix_flat.data(), rows * cols, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  }

  DistributeVectorStripes(world_size, rank == 0 ? input_data.second : Vector{}, base, rem, local_vector, rank,
                          my_width);

  if (my_width > 0 && rows > 0 && cols > 0) {
    MultiplyStrip(matrix_flat, local_vector, local_result, rows, cols, my_width, my_start);
  }

  if (rank == 0) {
    GatherResultsInRoot(world_size, rows, local_result, final_result);
  } else if (rows > 0) {
    MPI_Send(local_result.data(), rows, MPI_DOUBLE, 0, 102, MPI_COMM_WORLD);
  }

  std::vector<double> local_final_result;
  if (rows > 0) {
    local_final_result.resize(static_cast<size_t>(rows), 0.0);
  }

  if (rank == 0) {
    BroadcastFinalResult(rank, world_size, final_result, local_final_result);
    GetOutput() = final_result;
  } else {
    BroadcastFinalResult(rank, world_size, final_result, local_final_result);
    GetOutput() = local_final_result;
  }

  return true;
}

bool VerticalStripMatVecMPI::PostProcessingImpl() {
  return true;
}

}  // namespace artyushkina_vector

#ifdef __GNUC__
#  pragma GCC diagnostic pop
#endif
