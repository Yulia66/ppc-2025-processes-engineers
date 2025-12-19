#include <mpi.h>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

#include "artyushkina_bellman_ford_crs/mpi/include/ops_mpi.hpp"

namespace artyushkina_bellman_ford_crs {

namespace {

bool ProcessVertex(int32_t vertex, const std::vector<int32_t> &row_ptr, const std::vector<int32_t> &col_idx,
                   const std::vector<double> &values, std::vector<double> &distances, int32_t num_vertices,
                   int32_t num_edges) {
  const auto vertex_idx = static_cast<size_t>(vertex);

  if (distances[vertex_idx] == std::numeric_limits<double>::infinity()) {
    return false;
  }

  if (vertex_idx + 1 >= row_ptr.size()) {
    return false;
  }

  const int32_t start = row_ptr[vertex_idx];
  const int32_t end = row_ptr[vertex_idx + 1];

  if (start < 0 || end < start || end > num_edges) {
    return false;
  }

  bool updated = false;
  for (int32_t j = start; j < end; ++j) {
    const auto j_idx = static_cast<size_t>(j);
    if (j_idx >= col_idx.size() || j_idx >= values.size()) {
      continue;
    }

    const int32_t v = col_idx[j_idx];
    const double weight = values[j_idx];

    if (v < 0 || v >= num_vertices) {
      continue;
    }

    const auto v_idx = static_cast<size_t>(v);
    const double new_dist = distances[vertex_idx] + weight;

    if (new_dist < distances[v_idx]) {
      distances[v_idx] = new_dist;
      updated = true;
    }
  }

  return updated;
}

std::vector<double> RunSequentialVersion(const InType &graph) {
  if (graph.num_vertices <= 0) {
    return std::vector<double>{};
  }

  std::vector<double> distances(static_cast<size_t>(graph.num_vertices), std::numeric_limits<double>::infinity());

  if (graph.source_vertex >= 0 && graph.source_vertex < graph.num_vertices) {
    distances[static_cast<size_t>(graph.source_vertex)] = 0.0;
  }

  for (int32_t i = 0; i < graph.num_vertices - 1; ++i) {
    bool updated = false;

    for (int32_t vertex = 0; vertex < graph.num_vertices; ++vertex) {
      if (ProcessVertex(vertex, graph.row_ptr, graph.col_idx, graph.values, distances, graph.num_vertices,
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

bool ValidateEmptyGraph(const CRSGraph &graph) {
  return graph.source_vertex == 0 && graph.row_ptr.size() == 1 && graph.row_ptr[0] == 0 && graph.num_edges == 0 &&
         graph.col_idx.empty() && graph.values.empty();
}

bool ValidateRowPtr(const CRSGraph &graph) {
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

  return true;
}

bool ValidateIndicesAndValues(const CRSGraph &graph) {
  if (graph.col_idx.size() != static_cast<size_t>(graph.num_edges) ||
      graph.values.size() != static_cast<size_t>(graph.num_edges)) {
    return false;
  }

  for (size_t i = 0; i < graph.col_idx.size(); ++i) {
    if (graph.col_idx[i] < 0 || graph.col_idx[i] >= graph.num_vertices) {
      return false;
    }
  }

  return true;
}

bool ValidateCRSGraph(const CRSGraph &graph) {
  if (graph.num_vertices < 0) {
    return false;
  }

  if (graph.num_vertices == 0) {
    return ValidateEmptyGraph(graph);
  }

  if (graph.source_vertex < 0 || graph.source_vertex >= graph.num_vertices) {
    return false;
  }

  const size_t expected_row_ptr_size = static_cast<size_t>(graph.num_vertices) + 1;
  if (graph.row_ptr.size() != expected_row_ptr_size) {
    return false;
  }

  if (!ValidateRowPtr(graph)) {
    return false;
  }

  return ValidateIndicesAndValues(graph);
}

void BroadcastGraphData(int rank, int32_t &num_vertices, int32_t &source_vertex, int32_t &num_edges,
                        std::vector<int32_t> &row_ptr, std::vector<int32_t> &col_idx, std::vector<double> &values,
                        const CRSGraph &graph) {
  if (rank == 0) {
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
}

void PrepareBuffers(int rank, int32_t num_vertices, int32_t num_edges, std::vector<int32_t> &row_ptr,
                    std::vector<int32_t> &col_idx, std::vector<double> &values) {
  if (rank != 0) {
    const size_t row_ptr_size = (num_vertices > 0) ? static_cast<size_t>(num_vertices) + 1 : 1;
    const auto data_size = static_cast<size_t>(num_edges);

    row_ptr.resize(row_ptr_size, 0);
    col_idx.resize(data_size, 0);
    values.resize(data_size, 0.0);
  }
}

void BroadcastBuffers(int32_t num_vertices, int32_t num_edges, std::vector<int32_t> &row_ptr,
                      std::vector<int32_t> &col_idx, std::vector<double> &values) {
  const int32_t row_ptr_bcast_size = (num_vertices > 0) ? static_cast<int32_t>(num_vertices + 1) : 1;
  if (row_ptr_bcast_size > 0) {
    MPI_Bcast(row_ptr.data(), row_ptr_bcast_size, MPI_INT32_T, 0, MPI_COMM_WORLD);
  }

  if (num_edges > 0) {
    MPI_Bcast(col_idx.data(), num_edges, MPI_INT32_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(values.data(), num_edges, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  }
}

std::vector<double> RunMPIBellmanFord(int world_size, int rank, int32_t num_vertices, int32_t source_vertex,
                                      int32_t num_edges, const std::vector<int32_t> &row_ptr,
                                      const std::vector<int32_t> &col_idx, const std::vector<double> &values) {
  if (num_vertices <= 0) {
    return std::vector<double>{};
  }

  std::vector<double> distances(static_cast<size_t>(num_vertices), std::numeric_limits<double>::infinity());

  const bool valid_source = (source_vertex >= 0) && (source_vertex < num_vertices);
  if (valid_source) {
    distances[static_cast<size_t>(source_vertex)] = 0.0;
  }

  for (int32_t iter = 0; iter < num_vertices - 1; ++iter) {
    bool updated = false;

    for (int32_t vertex = rank; vertex < num_vertices; vertex += world_size) {
      if (ProcessVertex(vertex, row_ptr, col_idx, values, distances, num_vertices, num_edges)) {
        updated = true;
      }
    }

    MPI_Allreduce(MPI_IN_PLACE, distances.data(), num_vertices, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);

    const int global_updated = updated ? 1 : 0;
    int global_result = 0;
    MPI_Allreduce(&global_updated, &global_result, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);

    if (global_result == 0) {
      break;
    }
  }

  return distances;
}

}  // namespace

BellmanFordCRSMPI::BellmanFordCRSMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = OutType{};
}

bool BellmanFordCRSMPI::ValidationImpl() {
  int mpi_initialized = 0;
  MPI_Initialized(&mpi_initialized);

  int rank = 0;
  if (mpi_initialized != 0) {
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank != 0) {
      return true;
    }
  }

  return ValidateCRSGraph(GetInput());
}

bool BellmanFordCRSMPI::PreProcessingImpl() {
  GetOutput().clear();
  GetOutput().shrink_to_fit();
  return true;
}

bool BellmanFordCRSMPI::RunImpl() {
  int mpi_initialized = 0;
  MPI_Initialized(&mpi_initialized);

  if (mpi_initialized == 0) {
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

  BroadcastGraphData(rank, num_vertices, source_vertex, num_edges, row_ptr, col_idx, values, GetInput());

  PrepareBuffers(rank, num_vertices, num_edges, row_ptr, col_idx, values);
  BroadcastBuffers(num_vertices, num_edges, row_ptr, col_idx, values);

  GetOutput() = RunMPIBellmanFord(world_size, rank, num_vertices, source_vertex, num_edges, row_ptr, col_idx, values);
  return true;
}

bool BellmanFordCRSMPI::PostProcessingImpl() {
  if (GetOutput().capacity() > GetOutput().size() * 2) {
    GetOutput().shrink_to_fit();
  }
  return true;
}

}  // namespace artyushkina_bellman_ford_crs
