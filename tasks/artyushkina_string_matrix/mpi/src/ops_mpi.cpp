#include "artyushkina_string_matrix/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <array>
#include <climits>
#include <cstddef>
#include <ranges>  // IWYU pragma: keep
#include <utility>
#include <vector>

#include "artyushkina_string_matrix/common/include/common.hpp"

namespace artyushkina_string_matrix {

ArtyushkinaStringMatrixMPI::ArtyushkinaStringMatrixMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());

  if (!in.empty()) {
    GetInput() = in;
  } else {
    GetInput() = InType{};
  }

  GetOutput() = OutType{};
}

bool ArtyushkinaStringMatrixMPI::ValidationImpl() {
  const auto &input = GetInput();
  if (input.empty()) {
    return false;
  }

  const std::size_t cols = input[0].size();
  if (cols == 0) {
    return false;
  }

  return std::ranges::all_of(input, [cols](const auto &row) { return row.size() == cols; });
}

bool ArtyushkinaStringMatrixMPI::PreProcessingImpl() {
  GetOutput().clear();
  return true;
}

std::vector<int> ArtyushkinaStringMatrixMPI::FlattenMatrix(const std::vector<std::vector<int>> &matrix) {
  if (matrix.empty() || matrix[0].empty()) {
    return {};
  }

  const std::size_t rows = matrix.size();
  const std::size_t cols = matrix[0].size();

  std::vector<int> flat;
  flat.reserve(rows * cols);

  for (const auto &row : matrix) {
    flat.insert(flat.end(), row.begin(), row.end());
  }

  return flat;
}

std::pair<int, int> ArtyushkinaStringMatrixMPI::PrepareDimensions(const std::vector<std::vector<int>> &matrix, int rank,
                                                                  int &size) {
  int total_rows = static_cast<int>(matrix.size());
  int total_cols = (total_rows > 0) ? static_cast<int>(matrix[0].size()) : 0;

  if (rank == 0 && total_rows > 0 && total_rows < size) {
    size = std::min(total_rows, size);
  }

  if (size == 0) {
    size = 1;
  }

  std::array<int, 2> dimensions = {total_rows, total_cols};
  MPI_Bcast(dimensions.data(), 2, MPI_INT, 0, MPI_COMM_WORLD);

  return {dimensions[0], dimensions[1]};
}

std::pair<int, int> ArtyushkinaStringMatrixMPI::CalculateProcessInfo(int total_rows, int size, int rank) {
  if (size <= 0) {
    size = 1;
  }

  const int rows_per_process = total_rows / size;
  const int remainder = total_rows % size;

  const int my_rows = rows_per_process + ((rank < remainder) ? 1 : 0);

  int offset = 0;
  for (int i = 0; i < rank; ++i) {
    const int rows_for_i = rows_per_process + ((i < remainder) ? 1 : 0);
    offset += rows_for_i;
  }

  return {my_rows, offset};
}

std::vector<int> ArtyushkinaStringMatrixMPI::ScatterData(const std::vector<std::vector<int>> &matrix, int total_rows,
                                                         int total_cols, int size, int rank, int my_rows) {
  if (size <= 0) {
    size = 1;
  }

  std::vector<int> local_data(static_cast<std::size_t>(my_rows) * static_cast<std::size_t>(total_cols));

  if (rank == 0) {
    std::vector<int> flat_matrix = FlattenMatrix(matrix);

    const int rows_per_process = total_rows / size;
    const int remainder = total_rows % size;

    std::vector<int> send_counts(size);
    std::vector<int> displacements(size);

    int offset = 0;
    for (int i = 0; i < size; ++i) {
      const int rows_for_i = rows_per_process + ((i < remainder) ? 1 : 0);
      send_counts[i] = rows_for_i * total_cols;
      displacements[i] = offset * total_cols;
      offset += rows_for_i;
    }

    MPI_Scatterv(flat_matrix.data(), send_counts.data(), displacements.data(), MPI_INT, local_data.data(),
                 my_rows * total_cols, MPI_INT, 0, MPI_COMM_WORLD);
  } else {
    MPI_Scatterv(nullptr, nullptr, nullptr, MPI_INT, local_data.data(), my_rows * total_cols, MPI_INT, 0,
                 MPI_COMM_WORLD);
  }

  return local_data;
}

std::vector<int> ArtyushkinaStringMatrixMPI::ComputeLocalMinima(const std::vector<int> &local_data, int my_rows,
                                                                int total_cols) {
  std::vector<int> local_minima(static_cast<std::size_t>(my_rows), INT_MAX);

  for (int i = 0; i < my_rows; ++i) {
    for (int j = 0; j < total_cols; ++j) {
      const int index = (i * total_cols) + j;
      const int val = local_data[static_cast<std::size_t>(index)];
      local_minima[static_cast<std::size_t>(i)] = std::min(val, local_minima[static_cast<std::size_t>(i)]);
    }
  }

  return local_minima;
}

std::vector<int> ArtyushkinaStringMatrixMPI::GatherResults(const std::vector<int> &local_minima, int total_rows,
                                                           int size, int rank, int my_rows) {
  std::vector<int> global_minima;

  if (rank == 0) {
    global_minima.resize(static_cast<std::size_t>(total_rows));
  }

  if (size <= 0) {
    size = 1;
  }

  const int rows_per_process = total_rows / size;
  const int remainder = total_rows % size;

  std::vector<int> recv_counts(size);
  std::vector<int> displacements_recv(size);

  int rows_so_far = 0;
  for (int i = 0; i < size; ++i) {
    const int rows_for_i = rows_per_process + ((i < remainder) ? 1 : 0);
    recv_counts[i] = rows_for_i;
    displacements_recv[i] = rows_so_far;
    rows_so_far += rows_for_i;
  }

  MPI_Gatherv(local_minima.data(), my_rows, MPI_INT, global_minima.data(), recv_counts.data(),
              displacements_recv.data(), MPI_INT, 0, MPI_COMM_WORLD);

  return global_minima;
}

bool ArtyushkinaStringMatrixMPI::RunImpl() {
  int rank = 0;
  int size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  const auto &matrix = GetInput();

  if (matrix.empty()) {
    if (rank == 0) {
      GetOutput().clear();
    }
    return true;
  }

  auto [total_rows, total_cols] = PrepareDimensions(matrix, rank, size);
  auto [my_rows, offset] = CalculateProcessInfo(total_rows, size, rank);

  std::vector<int> local_data = ScatterData(matrix, total_rows, total_cols, size, rank, my_rows);

  std::vector<int> local_minima = ComputeLocalMinima(local_data, my_rows, total_cols);

  std::vector<int> global_minima = GatherResults(local_minima, total_rows, size, rank, my_rows);

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
