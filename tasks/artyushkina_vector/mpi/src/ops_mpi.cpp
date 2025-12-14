#include "artyushkina_vector/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

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

  size_t matrix_rows = matrix.size();
  size_t matrix_cols = matrix[0].size();
  size_t vector_size = vector.size();

  for (size_t i = 1; i < matrix_rows; ++i) {
    if (matrix[i].size() != matrix_cols) {
      return false;
    }
  }

  return vector_size == matrix_cols;
}

bool VerticalStripMatVecMPI::PreProcessingImpl() {
  return true;
}

bool VerticalStripMatVecMPI::RunImpl() {
  int world_size, rank;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  int rows = 0, cols = 0;

  if (rank == 0) {
    const auto &[matrix, vector] = GetInput();
    rows = static_cast<int>(matrix.size());
    cols = static_cast<int>(matrix[0].size());
  }

  MPI_Bcast(&rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&cols, 1, MPI_INT, 0, MPI_COMM_WORLD);

  if (rows <= 0 || cols <= 0) {
    GetOutput() = Vector();
    return true;
  }

  if (cols < world_size) {
    world_size = cols;
    if (rank >= world_size) {
      GetOutput() = Vector(static_cast<size_t>(rows), 0.0);
      return true;
    }
  }

  int base = cols / world_size;
  int rem = cols % world_size;
  int local_start = rank * base + std::min(rank, rem);
  int local_width = base + (rank < rem ? 1 : 0);

  std::vector<double> result(static_cast<size_t>(rows), 0.0);

  if (rank == 0) {
    const auto &[matrix, vector] = GetInput();

    for (int proc = 1; proc < world_size; ++proc) {
      int proc_start = proc * base + std::min(proc, rem);
      int proc_width = base + (proc < rem ? 1 : 0);

      if (proc_width > 0) {
        std::vector<double> send_vec(static_cast<size_t>(proc_width));
        for (int j = 0; j < proc_width; ++j) {
          send_vec[j] = vector[proc_start + j];
        }
        MPI_Send(send_vec.data(), proc_width, MPI_DOUBLE, proc, 1, MPI_COMM_WORLD);
      }
    }

    std::vector<double> local_vector(static_cast<size_t>(local_width));
    for (int j = 0; j < local_width; ++j) {
      local_vector[j] = vector[local_start + j];
    }

    for (int i = 0; i < rows; ++i) {
      double sum = 0.0;
      for (int j = 0; j < local_width; ++j) {
        sum += matrix[i][local_start + j] * local_vector[j];
      }
      result[i] = sum;
    }

    for (int proc = 1; proc < world_size; ++proc) {
      int proc_width = base + (proc < rem ? 1 : 0);
      if (proc_width > 0) {
        std::vector<double> partial_result(static_cast<size_t>(rows));
        MPI_Recv(partial_result.data(), rows, MPI_DOUBLE, proc, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        for (int i = 0; i < rows; ++i) {
          result[i] += partial_result[i];
        }
      }
    }

    GetOutput() = result;

  } else if (local_width > 0) {
    std::vector<double> local_vector(static_cast<size_t>(local_width));
    MPI_Recv(local_vector.data(), local_width, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    std::vector<double> matrix_flat(static_cast<size_t>(rows) * static_cast<size_t>(cols));
    MPI_Bcast(matrix_flat.data(), rows * cols, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    std::vector<double> partial_result(static_cast<size_t>(rows), 0.0);
    for (int i = 0; i < rows; ++i) {
      double sum = 0.0;
      for (int j = 0; j < local_width; ++j) {
        sum += matrix_flat[i * cols + local_start + j] * local_vector[j];
      }
      partial_result[i] = sum;
    }

    MPI_Send(partial_result.data(), rows, MPI_DOUBLE, 0, 2, MPI_COMM_WORLD);

    MPI_Recv(result.data(), rows, MPI_DOUBLE, 0, 3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    GetOutput() = result;

  } else {
    GetOutput() = Vector(static_cast<size_t>(rows), 0.0);
  }

  if (rank == 0) {
    for (int proc = 1; proc < world_size; ++proc) {
      MPI_Send(result.data(), rows, MPI_DOUBLE, proc, 3, MPI_COMM_WORLD);
    }
  }

  return true;
}

bool VerticalStripMatVecMPI::PostProcessingImpl() {
  return true;
}

}  // namespace artyushkina_vector
