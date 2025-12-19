#include "artyushkina_bellman_ford_crs/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

namespace artyushkina_bellman_ford_crs {

BellmanFordCRSMPI::BellmanFordCRSMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = OutType{};
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
  return true;
}

bool ProcessVertex(int32_t u, const std::vector<int32_t> &row_ptr, const std::vector<int32_t> &col_idx,
                   const std::vector<double> &values, std::vector<double> &distances, int32_t num_vertices,
                   int32_t num_edges) {
  size_t u_idx = static_cast<size_t>(u);

  if (distances[u_idx] == std::numeric_limits<double>::infinity()) {
    return false;
  }

  if (u_idx + 1 >= row_ptr.size()) {
    return false;
  }

  int32_t start = row_ptr[u_idx];
  int32_t end = row_ptr[u_idx + 1];

  if (start < 0 || end < start || end > num_edges) {
    return false;
  }

  bool updated = false;
  for (int32_t j = start; j < end; ++j) {
    size_t j_idx = static_cast<size_t>(j);
    if (j_idx >= col_idx.size() || j_idx >= values.size()) {
      continue;
    }

    int32_t v = col_idx[j_idx];
    double weight = values[j_idx];

    if (v < 0 || v >= num_vertices) {
      continue;
    }

    size_t v_idx = static_cast<size_t>(v);
    double new_dist = distances[u_idx] + weight;

    if (new_dist < distances[v_idx]) {
      distances[v_idx] = new_dist;
      updated = true;
    }
  }

  return updated;
}

std::vector<double> RunSequentialVersion(const CRSGraph &graph) {
  if (graph.num_vertices <= 0) {
    return std::vector<double>{};
  }

  std::vector<double> distances(static_cast<size_t>(graph.num_vertices), std::numeric_limits<double>::infinity());

  if (graph.source_vertex >= 0 && graph.source_vertex < graph.num_vertices) {
    distances[static_cast<size_t>(graph.source_vertex)] = 0.0;
  }

  for (int32_t i = 0; i < graph.num_vertices - 1; ++i) {
    bool updated = false;

    for (int32_t u = 0; u < graph.num_vertices; ++u) {
      if (ProcessVertex(u, graph.row_ptr, graph.col_idx, graph.values, distances, graph.num_vertices,
                        graph.num_edges)) {
        updated = true;
      }
    }

    if (!updated) {
      break;
    }
  }

  return distances;
}

bool BellmanFordCRSMPI::RunImpl() {
  int mpi_initialized = 0;
  MPI_Initialized(&mpi_initialized);

  if (!mpi_initialized) {
    GetOutput() = RunSequentialVersion(GetInput());
    return true;
  }

  int world_size = 0;
  int rank = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

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

  MPI_Bcast(&num_vertices, 1, MPI_INT32_T, 0, MPI_COMM_WORLD);
  MPI_Bcast(&source_vertex, 1, MPI_INT32_T, 0, MPI_COMM_WORLD);
  MPI_Bcast(&num_edges, 1, MPI_INT32_T, 0, MPI_COMM_WORLD);

  if (rank != 0) {
    const size_t row_ptr_size = (num_vertices > 0) ? static_cast<size_t>(num_vertices + 1) : 1;
    const size_t data_size = static_cast<size_t>(num_edges);

    row_ptr.resize(row_ptr_size, 0);
    col_idx.resize(data_size, 0);
    values.resize(data_size, 0.0);
  }

  const int row_ptr_bcast_size = (num_vertices > 0) ? (num_vertices + 1) : 1;
  MPI_Bcast(row_ptr.data(), row_ptr_bcast_size, MPI_INT32_T, 0, MPI_COMM_WORLD);

  if (num_edges > 0) {
    MPI_Bcast(col_idx.data(), num_edges, MPI_INT32_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(values.data(), num_edges, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  }

  std::vector<double> distances;

  if (num_vertices <= 0) {
    distances = std::vector<double>{};
  } else {
    distances.resize(static_cast<size_t>(num_vertices), std::numeric_limits<double>::infinity());

    const bool valid_source = (source_vertex >= 0) && (source_vertex < num_vertices);
    if (valid_source) {
      distances[static_cast<size_t>(source_vertex)] = 0.0;
    }

    for (int32_t iter = 0; iter < num_vertices - 1; ++iter) {
      bool updated = false;

      for (int32_t u = rank; u < num_vertices; u += world_size) {
        if (ProcessVertex(u, row_ptr, col_idx, values, distances, num_vertices, num_edges)) {
          updated = true;
        }
      }

      MPI_Allreduce(MPI_IN_PLACE, distances.data(), num_vertices, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);

      int global_updated = updated ? 1 : 0;
      MPI_Allreduce(MPI_IN_PLACE, &global_updated, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);

      if (global_updated == 0) {
        break;
      }
    }
  }

  GetOutput() = distances;
  return true;
}

bool BellmanFordCRSMPI::PostProcessingImpl() {
  if (GetOutput().capacity() > GetOutput().size() * 2) {
    GetOutput().shrink_to_fit();
  }
  return true;
}

}  // namespace artyushkina_bellman_ford_crs
