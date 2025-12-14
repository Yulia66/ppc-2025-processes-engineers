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

  // Проверка прямоугольности матрицы
  for (size_t i = 1; i < matrix_rows; ++i) {
    if (matrix[i].size() != matrix_cols) {
      return false;
    }
  }

  // Размер вектора должен совпадать с количеством столбцов матрицы
  if (vector_size != matrix_cols) {
    return false;
  }

  return true;
}

bool VerticalStripMatVecMPI::PreProcessingImpl() {
  return true;
}

void VerticalStripMatVecMPI::DistributeVectorColumns(
    int world_size, int base, int rem,
    std::vector<double> &local_vector,
    int rank, int /* matrix_cols */, int local_width) {  // Добавил local_width
  
  const int vector_tag = 201;

  if (rank == 0) {
    const auto &[matrix, full_vector] = GetInput();
    
    for (int proc = 0; proc < world_size; ++proc) {
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

      if (proc_width <= 0) continue;

      std::vector<double> send_buf(proc_width);
      for (int j = 0; j < proc_width; ++j) {
        int global_col = proc_start + j;
        send_buf[j] = full_vector[global_col];
      }

      if (proc == 0) {
        local_vector = std::move(send_buf);
      } else {
        MPI_Send(send_buf.data(), proc_width, MPI_DOUBLE, 
                 proc, vector_tag, MPI_COMM_WORLD);
      }
    }
  } else if (local_width > 0) {
    // Обеспечиваем что вектор имеет правильный размер
    if (local_vector.size() != static_cast<size_t>(local_width)) {
      local_vector.resize(local_width);
    }
    if (!local_vector.empty()) {
      MPI_Recv(local_vector.data(), local_width, MPI_DOUBLE,
               0, vector_tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
  }
}

void VerticalStripMatVecMPI::ComputeLocalStrip(
    const std::vector<double> &matrix_flat,
    const std::vector<double> &local_vector,
    std::vector<double> &partial_result,
    int rows, int cols, int local_width, int local_start) {
  
  // Добавляем проверки на пустые векторы
  if (matrix_flat.empty() || local_vector.empty() || partial_result.empty()) {
    return;
  }
  
  if (local_width <= 0) {
    return;
  }

  // Проверяем что индексы в пределах
  if (local_start + local_width > cols) {
    return;
  }

  // Для каждого элемента результата
  for (int i = 0; i < rows; ++i) {
    double sum = 0.0;
    
    // Для каждого локального столбца
    for (int local_j = 0; local_j < local_width; ++local_j) {
      int global_j = local_start + local_j;
      // Дополнительная проверка индексов
      if (global_j >= cols) break;
      
      size_t matrix_idx = static_cast<size_t>(i) * cols + global_j;
      if (matrix_idx >= matrix_flat.size()) break;
      
      double matrix_val = matrix_flat[matrix_idx];
      double vector_val = local_vector[local_j];
      sum += matrix_val * vector_val;
    }
    
    partial_result[i] = sum;
  }
}

void VerticalStripMatVecMPI::CollectResults(
    int world_size, int rank, int rows,
    int base, int rem,
    const std::vector<double> &partial_result,
    int local_width, int /* local_start */,
    std::vector<double> &final_result) {
  
  const int result_tag = 202;

  if (rank == 0) {
    // Сначала сохраняем свой частичный результат
    if (!partial_result.empty()) {
      for (int i = 0; i < rows && i < static_cast<int>(partial_result.size()); ++i) {
        final_result[i] = partial_result[i];
      }
    }

    // Получаем результаты от других процессов
    for (int proc = 1; proc < world_size; ++proc) {
      int proc_width = base;
      if (proc < rem) {
        proc_width += 1;
      }

      if (proc_width <= 0) continue;

      std::vector<double> recv_buf(rows);
      MPI_Recv(recv_buf.data(), rows, MPI_DOUBLE,
               proc, result_tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

      // Суммируем результаты
      for (int i = 0; i < rows; ++i) {
        final_result[i] += recv_buf[i];
      }
    }
  } else if (local_width > 0 && !partial_result.empty()) {
    // Отправляем свой частичный результат процессу 0
    MPI_Send(partial_result.data(), rows, MPI_DOUBLE,
             0, result_tag, MPI_COMM_WORLD);
  }
}

bool VerticalStripMatVecMPI::RunImpl() {
  int world_size, rank;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  int rows = 0, cols = 0;

  // Процесс 0 определяет размеры
  if (rank == 0) {
    const auto &[matrix, vector] = GetInput();
    rows = static_cast<int>(matrix.size());
    cols = static_cast<int>(matrix[0].size());
  }

  // Распространение размеров на все процессы
  MPI_Bcast(&rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&cols, 1, MPI_INT, 0, MPI_COMM_WORLD);

  // Проверка валидности размеров
  if (rows <= 0 || cols <= 0) {
    GetOutput() = Vector(rows, 0.0);
    return true;
  }

  // Разбиваем вектор на полосы
  int base = cols / world_size;
  int rem = cols % world_size;
  int local_start = rank * base + std::min(rank, rem);
  int local_width = base + (rank < rem ? 1 : 0);

  // Подготовка локальных данных
  std::vector<double> matrix_flat;
  std::vector<double> local_vector;
  std::vector<double> partial_result;
  std::vector<double> final_result;

  // Инициализируем векторы с правильными размерами
  if (local_width > 0) {
    local_vector.resize(local_width);
    partial_result.resize(rows, 0.0);
  }
  
  matrix_flat.resize(static_cast<size_t>(rows) * static_cast<size_t>(cols));

  // Процесс 0 инициализирует матрицу
  if (rank == 0) {
    const auto &[matrix, vector] = GetInput();
    for (int i = 0; i < rows; ++i) {
      for (int j = 0; j < cols; ++j) {
        matrix_flat[i * cols + j] = matrix[i][j];
      }
    }
  }

  // Распространение матрицы
  if (!matrix_flat.empty()) {
    MPI_Bcast(matrix_flat.data(), rows * cols, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  }

  // Распределение частей вектора
  DistributeVectorColumns(world_size, base, rem, local_vector, rank, cols, local_width);

  // Локальное вычисление
  if (local_width > 0 && !matrix_flat.empty() && !local_vector.empty() && !partial_result.empty()) {
    ComputeLocalStrip(matrix_flat, local_vector, partial_result, 
                      rows, cols, local_width, local_start);
  }

  // Сбор результатов
  if (rank == 0) {
    final_result.resize(rows, 0.0);
  }

  CollectResults(world_size, rank, rows, base, rem, 
                 partial_result, local_width, local_start, final_result);

  // Распространение финального результата
  if (rank == 0) {
    GetOutput() = final_result;
    
    // Отправляем результат другим процессам
    for (int proc = 1; proc < world_size; ++proc) {
      if (!final_result.empty()) {
        MPI_Send(final_result.data(), rows, MPI_DOUBLE, 
                 proc, 203, MPI_COMM_WORLD);
      }
    }
  } else {
    Vector result(rows);
    if (rows > 0) {
      MPI_Recv(result.data(), rows, MPI_DOUBLE, 
               0, 203, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    GetOutput() = result;
  }

  return true;
}

bool VerticalStripMatVecMPI::PostProcessingImpl() {
  return true;
}

}  // namespace artyushkina_vector