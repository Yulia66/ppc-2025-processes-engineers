#include "artyushkina_vector/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <vector>

namespace artyushkina_vector {

VerticalStripMatVecMPI::VerticalStripMatVecMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  // Используем move семантику
  GetInput().first = std::move(in.first);
  GetInput().second = std::move(in.second);
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

bool VerticalStripMatVecMPI::RunImpl() {
  int world_size, rank;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  int rows = 0, cols = 0;

  // Процесс 0 получает размеры
  if (rank == 0) {
    const auto &[matrix, vector] = GetInput();
    rows = static_cast<int>(matrix.size());
    cols = static_cast<int>(matrix[0].size());
  }

  // Распространяем размеры
  MPI_Bcast(&rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&cols, 1, MPI_INT, 0, MPI_COMM_WORLD);

  if (rows <= 0 || cols <= 0) {
    GetOutput() = std::vector<double>();
    return true;
  }

  // Если процессоров больше чем столбцов - используем только первый процесс
  if (world_size > cols) {
    std::vector<double> result(rows, 0.0);

    if (rank == 0) {
      const auto &[matrix, vector] = GetInput();
      for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
          result[i] += matrix[i][j] * vector[j];
        }
      }

      // Распространяем результат
      for (int proc = 1; proc < world_size; ++proc) {
        MPI_Send(result.data(), rows, MPI_DOUBLE, proc, 0, MPI_COMM_WORLD);
      }
    } else {
      MPI_Recv(result.data(), rows, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    GetOutput() = result;
    return true;
  }

  // Нормальный случай: распределяем столбцы

  // Распределение столбцов
  int base = cols / world_size;
  int rem = cols % world_size;
  int start_col = rank * base + std::min(rank, rem);
  int end_col = start_col + base + (rank < rem ? 1 : 0);
  int local_cols = end_col - start_col;

  // Буферы
  std::vector<double> matrix_flat(rows * cols);
  std::vector<double> local_vector(local_cols);
  std::vector<double> local_result(rows, 0.0);
  std::vector<double> final_result(rows, 0.0);

  // Процесс 0 инициализирует данные
  if (rank == 0) {
    const auto &[matrix, vector] = GetInput();

    // Заполняем матрицу
    for (int i = 0; i < rows; ++i) {
      for (int j = 0; j < cols; ++j) {
        matrix_flat[i * cols + j] = matrix[i][j];
      }
    }

    // Своя часть вектора
    for (int j = 0; j < local_cols; ++j) {
      local_vector[j] = vector[start_col + j];
    }

    // Отправляем другим процессам их части
    for (int proc = 1; proc < world_size; ++proc) {
      int proc_start = proc * base + std::min(proc, rem);
      int proc_end = proc_start + base + (proc < rem ? 1 : 0);
      int proc_cols = proc_end - proc_start;

      std::vector<double> send_buf(proc_cols);
      for (int j = 0; j < proc_cols; ++j) {
        send_buf[j] = vector[proc_start + j];
      }

      MPI_Send(send_buf.data(), proc_cols, MPI_DOUBLE, proc, 1, MPI_COMM_WORLD);
    }
  } else {
    // Получаем свою часть вектора
    MPI_Recv(local_vector.data(), local_cols, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  }

  // Распространяем матрицу
  MPI_Bcast(matrix_flat.data(), rows * cols, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  // Локальные вычисления
  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < local_cols; ++j) {
      local_result[i] += matrix_flat[i * cols + start_col + j] * local_vector[j];
    }
  }

  // Сбор результатов
  if (rank == 0) {
    // Копируем свою часть
    final_result = local_result;

    // Получаем от других
    for (int proc = 1; proc < world_size; ++proc) {
      std::vector<double> recv_buf(rows);
      MPI_Recv(recv_buf.data(), rows, MPI_DOUBLE, proc, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

      for (int i = 0; i < rows; ++i) {
        final_result[i] += recv_buf[i];
      }
    }

    GetOutput() = final_result;

    // Отправляем результат всем
    for (int proc = 1; proc < world_size; ++proc) {
      MPI_Send(final_result.data(), rows, MPI_DOUBLE, proc, 3, MPI_COMM_WORLD);
    }
  } else {
    // Отправляем свой результат
    MPI_Send(local_result.data(), rows, MPI_DOUBLE, 0, 2, MPI_COMM_WORLD);

    // Получаем финальный результат
    MPI_Recv(final_result.data(), rows, MPI_DOUBLE, 0, 3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    GetOutput() = final_result;
  }

  return true;
}

bool VerticalStripMatVecMPI::PostProcessingImpl() {
  return true;
}

}  // namespace artyushkina_vector
