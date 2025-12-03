#include "artyushkina_string_matrix/mpi/include/ops_mpi.hpp"

#include <algorithm>
#include <climits>

#include "C:/Program Files (x86)/Microsoft SDKs/MPI/Include/mpi.h"

namespace artyushkina_string_matrix {

ArtyushkinaStringMatrixMPI::ArtyushkinaStringMatrixMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = {};
}

bool ArtyushkinaStringMatrixMPI::ValidationImpl() {
  if (GetInput().empty()) {
    return false;
  }

  size_t cols = GetInput()[0].size();
  if (cols == 0) {
    return false;
  }

  for (const auto &row : GetInput()) {
    if (row.size() != cols) {
      return false;
    }
  }

  return true;
}

bool ArtyushkinaStringMatrixMPI::PreProcessingImpl() {
  GetOutput() = std::vector<int>();
  return true;
}

std::vector<int> ArtyushkinaStringMatrixMPI::FlattenMatrix(const std::vector<std::vector<int>> &matrix) {
  std::vector<int> flat;
  size_t rows = matrix.size();
  size_t cols = rows > 0 ? matrix[0].size() : 0;

  flat.reserve(rows * cols);
  for (const auto &row : matrix) {
    flat.insert(flat.end(), row.begin(), row.end());
  }

  return flat;
}

bool ArtyushkinaStringMatrixMPI::RunImpl() {
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  const auto &matrix = GetInput();

  // Размеры матрицы
  int total_rows = static_cast<int>(matrix.size());
  int total_cols = total_rows > 0 ? static_cast<int>(matrix[0].size()) : 0;

  if (rank == 0) {
    if (total_rows < size) {
      // Если строк меньше, чем процессов, используем только нужное количество процессов
      size = total_rows;
    }
  }

  // Рассылаем размеры матрицы всем процессам
  int dimensions[2] = {total_rows, total_cols};
  MPI_Bcast(dimensions, 2, MPI_INT, 0, MPI_COMM_WORLD);

  total_rows = dimensions[0];
  total_cols = dimensions[1];

  int rows_per_process = total_rows / size;
  int remainder = total_rows % size;

  int my_rows = rows_per_process + (rank < remainder ? 1 : 0);
  // int my_offset = 0;

  int offset = 0;
  for (int i = 0; i < rank; ++i) {
    int rows_for_i = rows_per_process + (i < remainder ? 1 : 0);
    offset += rows_for_i;
  }
  // my_offset = offset;

  // Буфер для данных
  std::vector<int> local_data;

  if (rank == 0) {
    // Преобразуем матрицу в одномерный массив
    std::vector<int> flat_matrix = FlattenMatrix(matrix);

    // Рассылаем данные всем процессам
    std::vector<int> send_counts(size);
    std::vector<int> displacements(size);

    offset = 0;
    for (int i = 0; i < size; ++i) {
      int rows_for_i = rows_per_process + (i < remainder ? 1 : 0);
      send_counts[i] = rows_for_i * total_cols;
      displacements[i] = offset * total_cols;
      offset += rows_for_i;
    }

    // Выделяем память для локальных данных
    local_data.resize(my_rows * total_cols);

    // Рассылаем данные
    MPI_Scatterv(flat_matrix.data(), send_counts.data(), displacements.data(), MPI_INT, local_data.data(),
                 my_rows * total_cols, MPI_INT, 0, MPI_COMM_WORLD);
  } else {
    // Выделяем память для локальных данных
    local_data.resize(my_rows * total_cols);

    // Получаем данные от главного процесса
    MPI_Scatterv(nullptr, nullptr, nullptr, MPI_INT, local_data.data(), my_rows * total_cols, MPI_INT, 0,
                 MPI_COMM_WORLD);
  }

  // Локальный поиск минимумов
  std::vector<int> local_minima(my_rows, INT_MAX);
  for (int i = 0; i < my_rows; ++i) {
    for (int j = 0; j < total_cols; ++j) {
      int val = local_data[i * total_cols + j];
      if (val < local_minima[i]) {
        local_minima[i] = val;
      }
    }
  }

  // Собираем результаты
  std::vector<int> global_minima;
  if (rank == 0) {
    global_minima.resize(total_rows);
  }

  // Подготавливаем данные для Gatherv
  std::vector<int> recv_counts(size);
  std::vector<int> displacements_recv(size);

  int rows_so_far = 0;
  for (int i = 0; i < size; ++i) {
    int rows_for_i = rows_per_process + (i < remainder ? 1 : 0);
    recv_counts[i] = rows_for_i;
    displacements_recv[i] = rows_so_far;
    rows_so_far += rows_for_i;
  }

  // Собираем все минимумы на главном процессе
  MPI_Gatherv(local_minima.data(), my_rows, MPI_INT, global_minima.data(), recv_counts.data(),
              displacements_recv.data(), MPI_INT, 0, MPI_COMM_WORLD);

  // Сохраняем результат на главном процессе
  if (rank == 0) {
    GetOutput() = global_minima;
  } else {
    GetOutput() = std::vector<int>();
  }

  return true;
}

bool ArtyushkinaStringMatrixMPI::PostProcessingImpl() {
  return true;
}

}  // namespace artyushkina_string_matrix
