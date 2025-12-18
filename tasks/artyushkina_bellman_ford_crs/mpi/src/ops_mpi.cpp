#include "artyushkina_bellman_ford_crs/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

namespace artyushkina_bellman_ford_crs {

// Глобальный флаг инициализации MPI
namespace {
bool mpi_initialized_globally = false;
bool mpi_finalize_on_exit = false;
}  // namespace

BellmanFordCRSMPI::BellmanFordCRSMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = OutType{};

  // Проверяем инициализацию MPI при создании объекта
  int initialized = 0;
  MPI_Initialized(&initialized);
  if (!initialized) {
    mpi_initialized_globally = true;
  }
}

BellmanFordCRSMPI::~BellmanFordCRSMPI() {
  // Не финализируем MPI здесь, чтобы избежать проблем с Valgrind
  // MPI должен финализироваться автоматически при выходе из программы
}

bool BellmanFordCRSMPI::ValidationImpl() {
  int mpi_initialized = 0;
  MPI_Initialized(&mpi_initialized);

  int rank = 0;
  if (mpi_initialized) {
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank != 0) {
      return true;
    }
  }

  const auto &graph = GetInput();

  if (graph.num_vertices < 0) {
    return false;
  }

  if (graph.num_vertices == 0) {
    if (graph.source_vertex != 0) {
      return false;
    }
    if (graph.row_ptr.size() != 1 || graph.row_ptr[0] != 0) {
      return false;
    }
    if (graph.num_edges != 0) {
      return false;
    }
    if (!graph.col_idx.empty() || !graph.values.empty()) {
      return false;
    }
    return true;
  }

  if (graph.source_vertex < 0 || graph.source_vertex >= graph.num_vertices) {
    return false;
  }

  if (graph.row_ptr.size() != static_cast<size_t>(graph.num_vertices + 1)) {
    return false;
  }

  if (graph.col_idx.size() != static_cast<size_t>(graph.num_edges) ||
      graph.values.size() != static_cast<size_t>(graph.num_edges)) {
    return false;
  }

  if (graph.row_ptr[0] != 0) {
    return false;
  }

  if (graph.row_ptr[static_cast<size_t>(graph.num_vertices)] != graph.num_edges) {
    return false;
  }

  for (size_t i = 1; i < graph.row_ptr.size(); ++i) {
    if (graph.row_ptr[i] < graph.row_ptr[i - 1]) {
      return false;
    }
  }

  for (size_t i = 0; i < graph.col_idx.size(); ++i) {
    if (graph.col_idx[i] < 0 || graph.col_idx[i] >= graph.num_vertices) {
      return false;
    }
  }

  return true;
}

bool BellmanFordCRSMPI::PreProcessingImpl() {
  GetOutput().clear();
  GetOutput().shrink_to_fit();

  // Инициализация MPI только если нужно
  int mpi_initialized = 0;
  MPI_Initialized(&mpi_initialized);
  if (!mpi_initialized) {
    // Используем MPI_Init вместо MPI_Init_thread для простоты
    MPI_Init(nullptr, nullptr);
    mpi_finalize_on_exit = true;
  }

  return true;
}

bool BellmanFordCRSMPI::RunImpl() {
  int mpi_initialized = 0;
  MPI_Initialized(&mpi_initialized);

  // Если MPI не инициализирован, запускаем последовательную версию
  if (!mpi_initialized) {
    const auto &graph = GetInput();

    if (graph.num_vertices <= 0) {
      GetOutput() = std::vector<double>{};
      return true;
    }

    std::vector<double> distances(static_cast<size_t>(graph.num_vertices), std::numeric_limits<double>::infinity());

    if (graph.source_vertex >= 0 && graph.source_vertex < graph.num_vertices) {
      distances[static_cast<size_t>(graph.source_vertex)] = 0.0;
    }

    for (int32_t i = 0; i < graph.num_vertices - 1; ++i) {
      bool updated = false;

      for (int32_t u = 0; u < graph.num_vertices; ++u) {
        size_t u_idx = static_cast<size_t>(u);
        if (distances[u_idx] == std::numeric_limits<double>::infinity()) {
          continue;
        }

        if (u_idx + 1 >= graph.row_ptr.size()) {
          continue;
        }

        int32_t start = graph.row_ptr[u_idx];
        int32_t end = graph.row_ptr[u_idx + 1];

        if (start < 0 || end < start || end > graph.num_edges) {
          continue;
        }

        for (int32_t j = start; j < end; ++j) {
          size_t j_idx = static_cast<size_t>(j);
          if (j_idx >= graph.col_idx.size() || j_idx >= graph.values.size()) {
            continue;
          }

          int32_t v = graph.col_idx[j_idx];
          double weight = graph.values[j_idx];

          if (v < 0 || v >= graph.num_vertices) {
            continue;
          }

          size_t v_idx = static_cast<size_t>(v);
          double new_dist = distances[u_idx] + weight;

          if (new_dist < distances[v_idx]) {
            distances[v_idx] = new_dist;
            updated = true;
          }
        }
      }

      if (!updated) {
        break;
      }
    }

    GetOutput() = distances;
    return true;
  }

  // MPI версия
  int world_size = 0;
  int rank = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  // Получаем данные
  int32_t num_vertices = 0;
  int32_t source_vertex = 0;
  int32_t num_edges = 0;

  std::vector<int32_t> row_ptr;
  std::vector<int32_t> col_idx;
  std::vector<double> values;

  if (rank == 0) {
    const auto &graph = GetInput();
    num_vertices = graph.num_vertices;
    source_vertex = graph.source_vertex;
    num_edges = graph.num_edges;

    row_ptr = graph.row_ptr;
    col_idx = graph.col_idx;
    values = graph.values;
  }

  // Распространяем данные
  MPI_Bcast(&num_vertices, 1, MPI_INT32_T, 0, MPI_COMM_WORLD);
  MPI_Bcast(&source_vertex, 1, MPI_INT32_T, 0, MPI_COMM_WORLD);
  MPI_Bcast(&num_edges, 1, MPI_INT32_T, 0, MPI_COMM_WORLD);

  if (rank != 0) {
    row_ptr.resize(static_cast<size_t>(num_vertices + 1));
    col_idx.resize(static_cast<size_t>(num_edges));
    values.resize(static_cast<size_t>(num_edges));
  }

  if (num_vertices > 0) {
    MPI_Bcast(row_ptr.data(), num_vertices + 1, MPI_INT32_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(col_idx.data(), num_edges, MPI_INT32_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(values.data(), num_edges, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  }

  // Выполняем алгоритм
  std::vector<double> distances;

  if (num_vertices > 0) {
    distances.resize(static_cast<size_t>(num_vertices), std::numeric_limits<double>::infinity());

    if (source_vertex >= 0 && source_vertex < num_vertices) {
      distances[static_cast<size_t>(source_vertex)] = 0.0;
    }

    for (int32_t iter = 0; iter < num_vertices - 1; ++iter) {
      bool updated = false;

      // Каждый процесс обрабатывает свою часть вершин
      for (int32_t u = rank; u < num_vertices; u += world_size) {
        size_t u_idx = static_cast<size_t>(u);

        if (distances[u_idx] == std::numeric_limits<double>::infinity()) {
          continue;
        }

        if (u_idx + 1 >= row_ptr.size()) {
          continue;
        }

        int32_t start = row_ptr[u_idx];
        int32_t end = row_ptr[u_idx + 1];

        for (int32_t j = start; j < end; ++j) {
          int32_t v = col_idx[static_cast<size_t>(j)];
          double weight = values[static_cast<size_t>(j)];

          if (v < 0 || v >= num_vertices) {
            continue;
          }

          double new_dist = distances[u_idx] + weight;
          size_t v_idx = static_cast<size_t>(v);

          if (new_dist < distances[v_idx]) {
            distances[v_idx] = new_dist;
            updated = true;
          }
        }
      }

      // Синхронизируем расстояния
      std::vector<double> global_distances(distances.size());
      MPI_Allreduce(distances.data(), global_distances.data(), num_vertices, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
      distances.swap(global_distances);

      // Проверяем, нужно ли продолжать
      int global_updated = updated ? 1 : 0;
      MPI_Allreduce(MPI_IN_PLACE, &global_updated, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);

      if (!global_updated) {
        break;
      }
    }
  } else {
    distances = std::vector<double>{};
  }

  GetOutput() = distances;
  return true;
}

bool BellmanFordCRSMPI::PostProcessingImpl() {
  if (GetOutput().capacity() > GetOutput().size() * 2) {
    GetOutput().shrink_to_fit();
  }

  // Финализируем MPI только если мы его инициализировали в PreProcessing
  if (mpi_finalize_on_exit) {
    int finalized = 0;
    MPI_Finalized(&finalized);
    if (!finalized) {
      MPI_Finalize();
    }
    mpi_finalize_on_exit = false;
  }

  return true;
}

}  // namespace artyushkina_bellman_ford_crs
