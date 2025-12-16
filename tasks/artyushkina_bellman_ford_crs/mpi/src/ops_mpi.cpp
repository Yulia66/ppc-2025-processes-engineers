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
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if (rank != 0) {
    return true;
  }

  const auto &graph = GetInput();

  if (graph.num_vertices <= 0) {
    return false;
  }

  if (graph.source_vertex < 0 || graph.source_vertex >= graph.num_vertices) {
    return false;
  }

  if (graph.row_ptr.size() != static_cast<size_t>(graph.num_vertices + 1)) {
    return false;
  }

  if (graph.col_idx.size() != static_cast<size_t>(graph.num_edges)) {
    return false;
  }

  return true;
}

bool BellmanFordCRSMPI::PreProcessingImpl() {
  return true;
}

bool BellmanFordCRSMPI::RunImpl() {
  int world_size = 0;
  int rank = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  CRSGraph graph;
  if (rank == 0) {
    graph = GetInput();
  }

  int32_t num_vertices = 0;
  int32_t source_vertex = 0;
  int32_t num_edges = 0;

  if (rank == 0) {
    num_vertices = graph.num_vertices;
    source_vertex = graph.source_vertex;
    num_edges = graph.num_edges;
  }

  MPI_Bcast(&num_vertices, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&source_vertex, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&num_edges, 1, MPI_INT, 0, MPI_COMM_WORLD);

  if (num_vertices <= 0) {
    GetOutput() = std::vector<double>{};
    return true;
  }

  std::vector<int32_t> row_ptr(static_cast<size_t>(num_vertices + 1));
  std::vector<int32_t> col_idx(static_cast<size_t>(num_edges));
  std::vector<double> values(static_cast<size_t>(num_edges));

  if (rank == 0) {
    row_ptr = graph.row_ptr;
    col_idx = graph.col_idx;
    values = graph.values;
  }

  MPI_Bcast(row_ptr.data(), num_vertices + 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(col_idx.data(), num_edges, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(values.data(), num_edges, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  std::vector<double> distances(static_cast<size_t>(num_vertices), std::numeric_limits<double>::infinity());
  distances[static_cast<size_t>(source_vertex)] = 0.0;

  for (int32_t i = 0; i < num_vertices - 1; ++i) {
    bool updated = false;

    for (int32_t u = 0; u < num_vertices; ++u) {
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

    if (!updated) {
      break;
    }
  }

  GetOutput() = distances;
  return true;
}

bool BellmanFordCRSMPI::PostProcessingImpl() {
  return true;
}

}  // namespace artyushkina_bellman_ford_crs
