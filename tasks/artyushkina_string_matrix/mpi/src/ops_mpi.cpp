#include "artyushkina_string_matrix/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <climits>
#include <utility>

namespace artyushkina_string_matrix {

ArtyushkinaStringMatrixMPI::ArtyushkinaStringMatrixMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());

  if (!in.empty()) {
    GetInput() = in;
  } else {
    GetInput() = std::vector<std::vector<int>>();
  }
  GetOutput() = std::vector<int>();
}

bool ArtyushkinaStringMatrixMPI::ValidationImpl() {
  const auto& input = GetInput();
  if (input.empty()) {
    return false;
  }

  const size_t cols = input[0].size();
  if (cols == 0) {
    return false;
  }

  for (const auto &row : input) {
    if (row.size() != cols) {
      return false;
    }
  }

  return true;
}

bool ArtyushkinaStringMatrixMPI::PreProcessingImpl() {
  GetOutput().clear();
  return true;
}

std::vector<int> ArtyushkinaStringMatrixMPI::FlattenMatrix(
    const std::vector<std::vector<int>> &matrix) {
  if (matrix.empty()) {
    return {};
  }

  const size_t rows = matrix.size();
  const size_t cols = matrix[0].size();
  
  std::vector<int> flat;
  flat.reserve(rows * cols);
  
  for (const auto &row : matrix) {
    flat.insert(flat.end(), row.begin(), row.end());
  }

  return flat;
}

bool ArtyushkinaStringMatrixMPI::RunImpl() {
  int rank = 0;
  int size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  const auto &matrix = GetInput();


  int total_rows = static_cast<int>(matrix.size());
  int total_cols = (total_rows > 0) ? static_cast<int>(matrix[0].size()) : 0;

  if (rank == 0) {
    if (total_rows < size) {
      size = total_rows;
    }
  }

  int dimensions[2] = {total_rows, total_cols};
  MPI_Bcast(dimensions, 2, MPI_INT, 0, MPI_COMM_WORLD);

  total_rows = dimensions[0];
  total_cols = dimensions[1];

  const int rows_per_process = total_rows / size;
  const int remainder = total_rows % size;

  const int my_rows = rows_per_process + ((rank < remainder) ? 1 : 0);


  int offset = 0;
  for (int i = 0; i < rank; ++i) {
    const int rows_for_i = rows_per_process + ((i < remainder) ? 1 : 0);
    offset += rows_for_i;
  }
 

  std::vector<int> local_data;

  if (rank == 0) {
    std::vector<int> flat_matrix = FlattenMatrix(matrix);

    std::vector<int> send_counts(size);
    std::vector<int> displacements(size);

    offset = 0;
    for (int i = 0; i < size; ++i) {
      const int rows_for_i = rows_per_process + ((i < remainder) ? 1 : 0);
      send_counts[i] = rows_for_i * total_cols;
      displacements[i] = offset * total_cols;
      offset += rows_for_i;
    }

    local_data.resize(static_cast<size_t>(my_rows) * static_cast<size_t>(total_cols));

    MPI_Scatterv(flat_matrix.data(), send_counts.data(), displacements.data(), 
                 MPI_INT, local_data.data(), my_rows * total_cols, MPI_INT, 
                 0, MPI_COMM_WORLD);
  } else {
    local_data.resize(static_cast<size_t>(my_rows) * static_cast<size_t>(total_cols));
    
    MPI_Scatterv(nullptr, nullptr, nullptr, MPI_INT, 
                 local_data.data(), my_rows * total_cols, MPI_INT, 
                 0, MPI_COMM_WORLD);
  }

  std::vector<int> local_minima(static_cast<size_t>(my_rows), INT_MAX);
  for (int i = 0; i < my_rows; ++i) {
    for (int j = 0; j < total_cols; ++j) {
      const int val = local_data[i * total_cols + j];
      if (val < local_minima[static_cast<size_t>(i)]) {
        local_minima[static_cast<size_t>(i)] = val;
      }
    }
  }

  std::vector<int> global_minima;
  if (rank == 0) {
    global_minima.resize(static_cast<size_t>(total_rows));
  }

  std::vector<int> recv_counts(size);
  std::vector<int> displacements_recv(size);

  int rows_so_far = 0;
  for (int i = 0; i < size; ++i) {
    const int rows_for_i = rows_per_process + ((i < remainder) ? 1 : 0);
    recv_counts[i] = rows_for_i;
    displacements_recv[i] = rows_so_far;
    rows_so_far += rows_for_i;
  }

  MPI_Gatherv(local_minima.data(), my_rows, MPI_INT, 
              global_minima.data(), recv_counts.data(),
              displacements_recv.data(), MPI_INT, 0, MPI_COMM_WORLD);

  if (rank == 0) {
    GetOutput() = std::move(global_minima);
  } else {
    GetOutput().clear();
  }

  return true;
}

bool ArtyushkinaStringMatrixMPI::PostProcessingImpl() {
  return true;
}

}  // namespace artyushkina_string_matrix