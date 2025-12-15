#include "artyushkina_vector/mpi/include/ops_mpi.hpp"

// Отключить предупреждение C4100 (неиспользуемые параметры) для MSVC
#ifdef _MSC_VER
#  pragma warning(push)
#  pragma warning(disable : 4100)  // unreferenced formal parameter
#endif

#include <mpi.h>

#include <algorithm>
#include <cstddef>
#include <vector>

#ifdef _MSC_VER
#  pragma warning(pop)
#endif

namespace artyushkina_vector {

VerticalStripMatVecMPI::VerticalStripMatVecMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());

  // Простая инициализация без декомпозиции
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

  // Проверяем, что матрица прямоугольная
  for (size_t i = 1; i < matrix.size(); ++i) {
    if (matrix[i].size() != cols) {
      return false;
    }
  }

  // Количество столбцов матрицы должно совпадать с размером вектора
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

void GetProcessParams(int proc, int base, int rem, int &proc_start, int &proc_width) {
  proc_start = proc * base;
  if (proc < rem) {
    proc_start += proc;
  } else {
    proc_start += rem;
  }

  proc_width = base;
  if (proc < rem) {
    proc_width += 1;
  }
}

void DistributeVectorStripes(int world_size, const Vector &vector, int base, int rem, Vector &local_vector, int rank,
                             int my_width) {
  const int tag_vector = 101;

  // Подавляем предупреждение о неиспользуемых параметрах
  (void)base;
  (void)rem;

  if (rank == 0) {
    for (int proc = 0; proc < world_size; ++proc) {
      int proc_start = 0;
      int proc_width = 0;
      GetProcessParams(proc, base, rem, proc_start, proc_width);

      if (proc_width <= 0) {
        continue;
      }

      // Подготовка полосы вектора для процесса
      std::vector<double> sendbuf(proc_width);
      for (int j = 0; j < proc_width; ++j) {
        sendbuf[j] = vector[proc_start + j];
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

void MultiplyStrip(const std::vector<double> &matrix_flat, const Vector &local_vector, Vector &local_result, int rows,
                   int cols, int my_width, int my_start) {
  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < my_width; ++j) {
      int global_j = my_start + j;
      local_result[i] += matrix_flat[i * cols + global_j] * local_vector[j];
    }
  }
}

void GatherResultsInRoot(int world_size, int rows, int base, int rem, const Vector &local_result,
                         Vector &final_result) {
  const int tag_result = 102;

  // Подавляем предупреждение о неиспользуемых параметрах
  (void)base;
  (void)rem;

  // Копируем свою часть
  for (int i = 0; i < rows; ++i) {
    final_result[i] = local_result[i];
  }

  // Получаем от других процессов
  for (int proc = 1; proc < world_size; ++proc) {
    std::vector<double> recv_buf(rows);
    MPI_Recv(recv_buf.data(), rows, MPI_DOUBLE, proc, tag_result, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    for (int i = 0; i < rows; ++i) {
      final_result[i] += recv_buf[i];
    }
  }
}

void BroadcastFinalResult(int rank, int world_size, const Vector &final_result, Vector &local_final_result) {
  const int tag_broadcast = 103;

  // Подавляем предупреждение о неиспользуемых параметрах
  (void)world_size;

  if (rank == 0) {
    for (int proc = 1; proc < world_size; ++proc) {
      MPI_Send(final_result.data(), final_result.size(), MPI_DOUBLE, proc, tag_broadcast, MPI_COMM_WORLD);
    }
  } else {
    MPI_Recv(local_final_result.data(), local_final_result.size(), MPI_DOUBLE, 0, tag_broadcast, MPI_COMM_WORLD,
             MPI_STATUS_IGNORE);
  }
}

}  // namespace

bool VerticalStripMatVecMPI::RunImpl() {
  int world_size, rank;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  int rows = 0, cols = 0;

  // Процесс 0 получает размеры
  if (rank == 0) {
    const auto &[matrix, vector] = GetInput();
    rows = static_cast<int>(matrix.size());
    if (rows > 0) {
      cols = static_cast<int>(matrix[0].size());
    }
  }

  // Распространяем размеры
  BroadcastDimensions(rows, cols);

  if (rows <= 0 || cols <= 0) {
    GetOutput() = Vector{};
    return true;
  }

  // Если процессоров больше чем столбцов
  if (world_size > cols) {
    Vector result(rows, 0.0);

    if (rank == 0) {
      const auto &[matrix, vector] = GetInput();
      // Последовательное умножение
      for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
          result[i] += matrix[i][j] * vector[j];
        }
      }

      // Отправляем результат всем
      for (int proc = 1; proc < world_size; ++proc) {
        MPI_Send(result.data(), rows, MPI_DOUBLE, proc, 100, MPI_COMM_WORLD);
      }
    } else {
      MPI_Recv(result.data(), rows, MPI_DOUBLE, 0, 100, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    GetOutput() = result;
    return true;
  }

  // Нормальное распределение - вертикальные полосы
  int base = cols / world_size;
  int rem = cols % world_size;
  int my_start = 0, my_width = 0;
  GetProcessParams(rank, base, rem, my_start, my_width);

  // Подготовка данных
  std::vector<double> matrix_flat(rows * cols, 0.0);
  Vector local_vector(my_width, 0.0);
  Vector local_result(rows, 0.0);
  Vector final_result(rows, 0.0);

  // Процесс 0 инициализирует матрицу и распределяет вектор
  if (rank == 0) {
    const auto &[matrix, vector] = GetInput();

    // Преобразуем матрицу в плоский массив
    for (int i = 0; i < rows; ++i) {
      for (int j = 0; j < cols; ++j) {
        matrix_flat[i * cols + j] = matrix[i][j];
      }
    }
  }

  // Распространяем матрицу
  MPI_Bcast(matrix_flat.data(), rows * cols, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  // Распределяем полосы вектора
  DistributeVectorStripes(world_size, rank == 0 ? GetInput().second : Vector{}, base, rem, local_vector, rank,
                          my_width);

  // Локальные вычисления
  if (my_width > 0) {
    MultiplyStrip(matrix_flat, local_vector, local_result, rows, cols, my_width, my_start);
  }

  // Сбор и распространение результатов
  if (rank == 0) {
    GatherResultsInRoot(world_size, rows, base, rem, local_result, final_result);
  } else {
    MPI_Send(local_result.data(), rows, MPI_DOUBLE, 0, 102, MPI_COMM_WORLD);
  }

  // Распространяем финальный результат
  Vector local_final_result(rows, 0.0);
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
