#include "artyushkina_vector/mpi/include/ops_mpi.hpp"

#include <mpi.h>

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

  // Создаем вектор результата
  std::vector<double> result(rows, 0.0);

  // Если процессоров больше чем столбцов - используем только первый процесс
  if (world_size > cols) {
    if (rank == 0) {
      const auto &[matrix, vector] = GetInput();
      for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
          result[i] += matrix[i][j] * vector[j];
        }
      }
    }

    // Распространяем результат
    if (rank == 0) {
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

  // Каждый процесс вычисляет свою часть
  int base = cols / world_size;
  int rem = cols % world_size;
  int start_col = rank * base + std::min(rank, rem);
  int end_col = start_col + base + (rank < rem ? 1 : 0);

  // Создаем локальные буферы
  std::vector<double> matrix_flat(rows * cols);
  std::vector<double> local_vector(end_col - start_col);
  std::vector<double> local_result(rows, 0.0);

  // Процесс 0 заполняет матрицу и распределяет вектор
  if (rank == 0) {
    const auto &[matrix, vector] = GetInput();

    // Заполняем плоскую матрицу
    for (int i = 0; i < rows; ++i) {
      for (int j = 0; j < cols; ++j) {
        matrix_flat[i * cols + j] = matrix[i][j];
      }
    }

    // Заполняем свою часть вектора
    for (int j = start_col; j < end_col; ++j) {
      local_vector[j - start_col] = vector[j];
    }

    // Отправляем части вектора другим процессам
    for (int proc = 1; proc < world_size; ++proc) {
      int proc_start = proc * base + std::min(proc, rem);
      int proc_end = proc_start + base + (proc < rem ? 1 : 0);
      int proc_width = proc_end - proc_start;

      std::vector<double> send_buf(proc_width);
      for (int j = 0; j < proc_width; ++j) {
        send_buf[j] = vector[proc_start + j];
      }

      MPI_Send(send_buf.data(), proc_width, MPI_DOUBLE, proc, 1, MPI_COMM_WORLD);
    }
  } else {
    // Получаем свою часть вектора
    int local_width = end_col - start_col;
    MPI_Recv(local_vector.data(), local_width, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  }

  // Все получают матрицу
  MPI_Bcast(matrix_flat.data(), rows * cols, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  // Локальные вычисления
  for (int i = 0; i < rows; ++i) {
    for (int j = start_col; j < end_col; ++j) {
      int local_j = j - start_col;
      local_result[i] += matrix_flat[i * cols + j] * local_vector[local_j];
    }
  }

  // Сбор результатов на процесс 0
  if (rank == 0) {
    // Копируем свою часть
    result = local_result;

    // Получаем от других
    for (int proc = 1; proc < world_size; ++proc) {
      std::vector<double> recv_buf(rows);
      MPI_Recv(recv_buf.data(), rows, MPI_DOUBLE, proc, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

      for (int i = 0; i < rows; ++i) {
        result[i] += recv_buf[i];
      }
    }

    GetOutput() = result;

    // Отправляем финальный результат всем
    for (int proc = 1; proc < world_size; ++proc) {
      MPI_Send(result.data(), rows, MPI_DOUBLE, proc, 3, MPI_COMM_WORLD);
    }
  } else {
    // Отправляем свой результат
    MPI_Send(local_result.data(), rows, MPI_DOUBLE, 0, 2, MPI_COMM_WORLD);

    // Получаем финальный результат
    MPI_Recv(result.data(), rows, MPI_DOUBLE, 0, 3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    GetOutput() = result;
  }

  return true;
}

bool VerticalStripMatVecMPI::PostProcessingImpl() {
  return true;
}

}  // namespace artyushkina_vector
