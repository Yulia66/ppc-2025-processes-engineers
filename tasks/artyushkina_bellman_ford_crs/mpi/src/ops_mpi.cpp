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

  if (!mpi_initialized) {
    return false;
  }

  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if (rank != 0) {
    return true;
  }

  const auto &graph = GetInput();

  if (graph.num_vertices < 0) {
    return false;
  }

  if (graph.source_vertex < 0 || (graph.num_vertices > 0 && graph.source_vertex >= graph.num_vertices)) {
    return false;
  }

  if (graph.row_ptr.size() != static_cast<size_t>(graph.num_vertices + 1)) {
    return false;
  }

  if (graph.col_idx.size() != static_cast<size_t>(graph.num_edges) ||
      graph.values.size() != static_cast<size_t>(graph.num_edges)) {
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

bool BellmanFordCRSMPI::RunImpl() {
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
    if (num_vertices > 0) {
      row_ptr.resize(static_cast<size_t>(num_vertices + 1));
    } else {
      row_ptr.resize(1);
    }

    if (num_edges > 0) {
      col_idx.resize(static_cast<size_t>(num_edges));
      values.resize(static_cast<size_t>(num_edges));
    }
  }

  if (num_vertices >= 0) {
    int row_ptr_size = (num_vertices > 0) ? (num_vertices + 1) : 1;
    MPI_Bcast(row_ptr.data(), row_ptr_size, MPI_INT32_T, 0, MPI_COMM_WORLD);
  }

  if (num_edges > 0) {
    MPI_Bcast(col_idx.data(), num_edges, MPI_INT32_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(values.data(), num_edges, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  }

  std::vector<double> distances;

  if (num_vertices > 0) {
    distances.resize(static_cast<size_t>(num_vertices), std::numeric_limits<double>::infinity());

    if (source_vertex >= 0 && source_vertex < num_vertices) {
      distances[static_cast<size_t>(source_vertex)] = 0.0;
    }

    for (int32_t iter = 0; iter < num_vertices - 1; ++iter) {
      bool updated = false;

      for (int32_t u = rank; u < num_vertices; u += world_size) {
        size_t u_idx = static_cast<size_t>(u);

        if (distances[u_idx] == std::numeric_limits<double>::infinity()) {
          continue;
        }

        int32_t start = row_ptr[u_idx];
        int32_t end = row_ptr[u_idx + 1];

        for (int32_t j = start; j < end; ++j) {
          size_t j_idx = static_cast<size_t>(j);
          int32_t v = col_idx[j_idx];
          double weight = values[j_idx];

          size_t v_idx = static_cast<size_t>(v);
          double new_dist = distances[u_idx] + weight;

          if (new_dist < distances[v_idx]) {
            distances[v_idx] = new_dist;
            updated = true;
          }
        }
      }

      MPI_Allreduce(MPI_IN_PLACE, distances.data(), num_vertices, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);

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

  row_ptr.clear();
  row_ptr.shrink_to_fit();
  col_idx.clear();
  col_idx.shrink_to_fit();
  values.clear();
  values.shrink_to_fit();

  return true;
}

bool BellmanFordCRSMPI::PostProcessingImpl() {
  if (GetOutput().capacity() > GetOutput().size() * 2) {
    GetOutput().shrink_to_fit();
  }
  return true;
}

}  // namespace artyushkina_bellman_ford_crs
