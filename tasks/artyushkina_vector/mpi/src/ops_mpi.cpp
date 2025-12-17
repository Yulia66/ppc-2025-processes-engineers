#include "artyushkina_vector/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cstddef>
#include <utility>
#include <vector>

#include "artyushkina_vector/common/include/common.hpp"

#ifdef __GNUC__
#  pragma GCC diagnostic push
#  pragma GCC diagnostic ignored "-Wnull-dereference"
#endif

namespace artyushkina_vector {

VerticalStripMatVecMPI::VerticalStripMatVecMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = Vector{};
}

bool VerticalStripMatVecMPI::ValidationImpl() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if (rank != 0) {
    return true;
  }

  const auto &[matrix, vector] = GetInput();

  if (matrix.empty()) {
    return vector.empty();
  }

  if (vector.empty()) {
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

constexpr int kTagVector = 101;
constexpr int kTagResult = 102;
constexpr int kTagBroadcast = 103;
constexpr int kTagSimple = 100;
constexpr int kTagEmpty = 104;

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

void PrepareAndSendVectorParts(int world_size, const Vector &vector, int base, int rem,
                               std::vector<double> &local_vector) {
  for (int proc = 0; proc < world_size; ++proc) {
    auto [proc_start, proc_width] = GetProcessParams(proc, base, rem);

    if (proc_width <= 0) {
      int empty_signal = -1;
      if (proc != 0) {
        MPI_Send(&empty_signal, 1, MPI_INT, proc, kTagEmpty, MPI_COMM_WORLD);
      }
      continue;
    }

    std::vector<double> sendbuf(static_cast<size_t>(proc_width));
    for (int j = 0; j < proc_width; ++j) {
      auto j_idx = static_cast<size_t>(j);
      auto src_idx = static_cast<size_t>(proc_start) + j_idx;
      sendbuf[j_idx] = vector[src_idx];
    }

    if (proc == 0) {
      local_vector = std::move(sendbuf);
    } else {
      MPI_Send(sendbuf.data(), proc_width, MPI_DOUBLE, proc, kTagVector, MPI_COMM_WORLD);
    }
  }
}

void ReceiveVectorPart(int my_width, std::vector<double> &local_vector) {
  if (my_width > 0) {
    local_vector.resize(static_cast<size_t>(my_width));
    MPI_Recv(local_vector.data(), my_width, MPI_DOUBLE, 0, kTagVector, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  } else {
    int empty_signal = 0;
    MPI_Recv(&empty_signal, 1, MPI_INT, 0, kTagEmpty, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  }
}

void DistributeVectorStripes(int world_size, const Vector &vector, int base, int rem, std::vector<double> &local_vector,
                             int rank, int my_width) {
  if (rank == 0) {
    PrepareAndSendVectorParts(world_size, vector, base, rem, local_vector);
  } else {
    ReceiveVectorPart(my_width, local_vector);
  }
}

void MultiplyStrip(const std::vector<double> &matrix_flat, const std::vector<double> &local_vector,
                   std::vector<double> &local_result, int rows, int cols, int my_width, int my_start) {
  for (int i = 0; i < rows; ++i) {
    auto i_idx = static_cast<size_t>(i);
    for (int j = 0; j < my_width; ++j) {
      auto j_idx = static_cast<size_t>(j);
      auto global_j = my_start + j;
      auto global_j_idx = static_cast<size_t>(global_j);
      auto matrix_idx = static_cast<size_t>(i * cols) + global_j_idx;
      local_result[i_idx] += matrix_flat[matrix_idx] * local_vector[j_idx];
    }
  }
}

void GatherResultsInRoot(int world_size, int rows, const std::vector<double> &local_result,
                         std::vector<double> &final_result) {
  if (rows > 0) {
    std::ranges::copy(local_result, final_result.begin());
  }

  for (int proc = 1; proc < world_size; ++proc) {
    if (rows > 0) {
      std::vector<double> recv_buf(static_cast<size_t>(rows));
      MPI_Recv(recv_buf.data(), rows, MPI_DOUBLE, proc, kTagResult, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

      for (int i = 0; i < rows; ++i) {
        auto i_idx = static_cast<size_t>(i);
        final_result[i_idx] += recv_buf[i_idx];
      }
    } else {
      int dummy = 0;
      MPI_Recv(&dummy, 1, MPI_INT, proc, kTagResult, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
  }
}

void BroadcastFinalResult(int rank, int world_size, const std::vector<double> &final_result,
                          std::vector<double> &local_final_result) {
  if (rank == 0) {
    for (int proc = 1; proc < world_size; ++proc) {
      if (!final_result.empty()) {
        MPI_Send(final_result.data(), static_cast<int>(final_result.size()), MPI_DOUBLE, proc, kTagBroadcast,
                 MPI_COMM_WORLD);
      } else {
        int empty_signal = -1;
        MPI_Send(&empty_signal, 1, MPI_INT, proc, kTagEmpty, MPI_COMM_WORLD);
      }
    }
    local_final_result = final_result;
  } else {
    MPI_Status status;
    MPI_Probe(0, kTagBroadcast, MPI_COMM_WORLD, &status);

    int count = 0;
    MPI_Get_count(&status, MPI_DOUBLE, &count);

    if (count > 0) {
      local_final_result.resize(static_cast<size_t>(count));
      MPI_Recv(local_final_result.data(), count, MPI_DOUBLE, 0, kTagBroadcast, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    } else {
      int empty_signal = 0;
      MPI_Recv(&empty_signal, 1, MPI_INT, 0, kTagEmpty, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      local_final_result.clear();
    }
  }
}

bool HandleWorldSizeGreaterThanCols(int world_size, int rank, int rows, int cols, std::vector<double> &result,
                                    const InType &input) {
  if (rows <= 0) {
    result.clear();
    MPI_Barrier(MPI_COMM_WORLD);
    return true;
  }

  result.resize(static_cast<size_t>(rows), 0.0);

  if (rank == 0) {
    const auto &[matrix, vector] = input;
    for (int i = 0; i < rows; ++i) {
      auto i_idx = static_cast<size_t>(i);
      for (int j = 0; j < cols; ++j) {
        auto j_idx = static_cast<size_t>(j);
        result[i_idx] += matrix[i_idx][j_idx] * vector[j_idx];
      }
    }

    for (int proc = 1; proc < world_size; ++proc) {
      MPI_Send(result.data(), rows, MPI_DOUBLE, proc, kTagSimple, MPI_COMM_WORLD);
    }
  } else {
    MPI_Recv(result.data(), rows, MPI_DOUBLE, 0, kTagSimple, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  }

  return true;
}

void PrepareMatrixFlat(int rank, int rows, int cols, std::vector<double> &matrix_flat, const InType &input) {
  if (rank == 0 && rows > 0 && cols > 0) {
    const auto &[matrix, vector] = input;
    size_t total_size = static_cast<size_t>(rows) * static_cast<size_t>(cols);
    matrix_flat.resize(total_size, 0.0);

    for (int i = 0; i < rows; ++i) {
      auto i_idx = static_cast<size_t>(i);
      for (int j = 0; j < cols; ++j) {
        auto j_idx = static_cast<size_t>(j);
        auto matrix_idx = static_cast<size_t>(i * cols) + j_idx;
        matrix_flat[matrix_idx] = matrix[i_idx][j_idx];
      }
    }
  } else if (rows > 0 && cols > 0) {
    size_t total_size = static_cast<size_t>(rows) * static_cast<size_t>(cols);
    matrix_flat.resize(total_size, 0.0);
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

bool ProcessDimensions(int world_size, int rank, int rows, int cols, const InType &input_data, Vector &result) {
  if (rows <= 0 || cols <= 0) {
    result = Vector{};
    MPI_Barrier(MPI_COMM_WORLD);
    return true;
  }

  if (world_size > cols) {
    std::vector<double> temp_result;
    bool success = HandleWorldSizeGreaterThanCols(world_size, rank, rows, cols, temp_result, input_data);
    if (success) {
      if (!temp_result.empty()) {
        result = Vector(temp_result.begin(), temp_result.end());
      } else {
        result = Vector{};
      }
    }
    return true;
  }

  return false;
}

bool PrepareLocalData(int rows, int cols, int my_width, std::vector<double> &matrix_flat,
                      std::vector<double> &local_vector, std::vector<double> &local_result,
                      std::vector<double> &final_result) {
  if (rows > 0 && cols > 0) {
    size_t matrix_size = static_cast<size_t>(rows) * static_cast<size_t>(cols);
    matrix_flat.resize(matrix_size, 0.0);
  } else {
    matrix_flat.clear();
  }

  if (my_width > 0) {
    local_vector.resize(static_cast<size_t>(my_width), 0.0);
  } else {
    local_vector.clear();
  }

  if (rows > 0) {
    local_result.resize(static_cast<size_t>(rows), 0.0);
    final_result.resize(static_cast<size_t>(rows), 0.0);
  } else {
    local_result.clear();
    final_result.clear();
  }

  return true;
}

bool PerformLocalComputation(int rank, int world_size, int rows, int cols, int my_start, int my_width,
                             const InType &input_data, std::vector<double> &matrix_flat,
                             std::vector<double> &local_vector, std::vector<double> &local_result) {
  PrepareMatrixFlat(rank, rows, cols, matrix_flat, input_data);

  if (rows > 0 && cols > 0) {
    MPI_Bcast(matrix_flat.data(), rows * cols, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  }

  DistributeVectorStripes(world_size, rank == 0 ? input_data.second : Vector{}, cols / world_size, cols % world_size,
                          local_vector, rank, my_width);

  if (my_width > 0 && rows > 0 && cols > 0) {
    MultiplyStrip(matrix_flat, local_vector, local_result, rows, cols, my_width, my_start);
  }

  return true;
}

bool CollectResults(int rank, int world_size, int rows, const std::vector<double> &local_result,
                    std::vector<double> &final_result) {
  if (rank == 0) {
    GatherResultsInRoot(world_size, rows, local_result, final_result);
  } else if (rows > 0) {
    MPI_Send(local_result.data(), rows, MPI_DOUBLE, 0, kTagResult, MPI_COMM_WORLD);
  } else {
    int dummy = 0;
    MPI_Send(&dummy, 1, MPI_INT, 0, kTagResult, MPI_COMM_WORLD);
  }

  return true;
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

  Vector result;
  if (ProcessDimensions(world_size, rank, rows, cols, input_data, result)) {
    GetOutput() = result;
    return true;
  }

  int base = cols / world_size;
  int rem = cols % world_size;
  auto [my_start, my_width] = GetProcessParams(rank, base, rem);

  std::vector<double> matrix_flat;
  std::vector<double> local_vector;
  std::vector<double> local_result;
  std::vector<double> final_result;

  PrepareLocalData(rows, cols, my_width, matrix_flat, local_vector, local_result, final_result);

  PerformLocalComputation(rank, world_size, rows, cols, my_start, my_width, input_data, matrix_flat, local_vector,
                          local_result);

  CollectResults(rank, world_size, rows, local_result, final_result);

  std::vector<double> local_final_result;
  if (rows > 0) {
    local_final_result.resize(static_cast<size_t>(rows), 0.0);
  }

  BroadcastFinalResult(rank, world_size, final_result, local_final_result);

  GetOutput() = Vector(local_final_result.begin(), local_final_result.end());

  return true;
}

bool VerticalStripMatVecMPI::PostProcessingImpl() {
  return true;
}

}  // namespace artyushkina_vector

#ifdef __GNUC__
#  pragma GCC diagnostic pop
#endif
