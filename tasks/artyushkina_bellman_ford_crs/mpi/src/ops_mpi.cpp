#include "artyushkina_bellman_ford_crs/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

#include "artyushkina_bellman_ford_crs/common/include/common.hpp"

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

bool ValidateSourceVertex(const CRSGraph &graph) {
  return graph.source_vertex >= 0 && graph.source_vertex < graph.num_vertices;
}

bool ValidateRowPtrSize(const CRSGraph &graph) {
  const size_t expected_row_ptr_size = static_cast<size_t>(graph.num_vertices) + 1;
  return graph.row_ptr.size() == expected_row_ptr_size;
}

bool ValidateArraysSize(const CRSGraph &graph) {
  return graph.col_idx.size() == static_cast<size_t>(graph.num_edges) &&
         graph.values.size() == static_cast<size_t>(graph.num_edges);
}

bool ValidateRowPtrBasic(const CRSGraph &graph) {
  if (graph.row_ptr.empty()) {
    return false;
  }
  return graph.row_ptr[0] == 0 && graph.row_ptr[static_cast<size_t>(graph.num_vertices)] == graph.num_edges;
}

bool ValidateRowPtrMonotonic(const CRSGraph &graph) {
  for (size_t i = 1; i < graph.row_ptr.size(); ++i) {
    if (graph.row_ptr[i] < graph.row_ptr[i - 1]) {
      return false;
    }
  }
  return true;
}

bool ValidateIndices(const CRSGraph &graph) {
  return std::ranges::all_of(graph.col_idx, [&graph](int32_t idx) { return idx >= 0 && idx < graph.num_vertices; });
}

bool ValidateCRSGraph(const CRSGraph &graph) {
  if (graph.num_vertices < 0) {
    return false;
  }

  if (graph.num_vertices == 0) {
    return ValidateEmptyGraph(graph);
  }

  if (!ValidateSourceVertex(graph)) {
    return false;
  }

  if (!ValidateRowPtrSize(graph)) {
    return false;
  }

  if (!ValidateArraysSize(graph)) {
    return false;
  }

  if (!ValidateRowPtrBasic(graph)) {
    return false;
  }

  if (!ValidateRowPtrMonotonic(graph)) {
    return false;
  }

  return ValidateIndices(graph);
}

struct GraphData {
  int num_vertices{0};
  int source_vertex{0};
  int num_edges{0};
  std::vector<int> row_ptr;
  std::vector<int> col_idx;
  std::vector<double> values;
};

GraphData GetGraphData(int rank, const CRSGraph &input_graph) {
  GraphData data;

  if (rank == 0) {
    data.num_vertices = static_cast<int>(input_graph.num_vertices);
    data.source_vertex = static_cast<int>(input_graph.source_vertex);
    data.num_edges = static_cast<int>(input_graph.num_edges);

    data.row_ptr.reserve(input_graph.row_ptr.size());
    for (const auto &val : input_graph.row_ptr) {
      data.row_ptr.push_back(static_cast<int>(val));
    }

    data.col_idx.reserve(input_graph.col_idx.size());
    for (const auto &val : input_graph.col_idx) {
      data.col_idx.push_back(static_cast<int>(val));
    }

    data.values = input_graph.values;
  }

  return data;
}

void BroadcastBasicData(GraphData &data) {
  MPI_Bcast(&data.num_vertices, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&data.source_vertex, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&data.num_edges, 1, MPI_INT, 0, MPI_COMM_WORLD);
}

void ResizeBuffers(int rank, GraphData &data) {
  if (rank != 0) {
    const auto row_ptr_size = (data.num_vertices > 0) ? static_cast<size_t>(data.num_vertices) + 1 : 1;
    const auto data_size = static_cast<size_t>(data.num_edges);

    data.row_ptr.resize(row_ptr_size, 0);
    data.col_idx.resize(data_size, 0);
    data.values.resize(data_size, 0.0);
  }
}

void BroadcastBuffers(GraphData &data) {
  if (data.num_vertices > 0) {
    const int row_ptr_bcast_size = data.num_vertices + 1;
    MPI_Bcast(data.row_ptr.data(), row_ptr_bcast_size, MPI_INT, 0, MPI_COMM_WORLD);
  }

  if (data.num_edges > 0) {
    MPI_Bcast(data.col_idx.data(), data.num_edges, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(data.values.data(), data.num_edges, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  }
}

std::vector<double> InitializeDistances(int num_vertices, int source_vertex) {
  if (num_vertices <= 0) {
    return std::vector<double>{};
  }

  std::vector<double> distances(static_cast<size_t>(num_vertices), std::numeric_limits<double>::infinity());

  if (source_vertex >= 0 && source_vertex < num_vertices) {
    distances[static_cast<size_t>(source_vertex)] = 0.0;
  }

  return distances;
}

bool PerformBellmanFordIteration(int world_size, int rank, const GraphData &data, std::vector<double> &distances) {
  bool updated = false;

  for (int vertex = rank; vertex < data.num_vertices; vertex += world_size) {
    auto vertex_32 = static_cast<int32_t>(vertex);
    auto num_vertices_32 = static_cast<int32_t>(data.num_vertices);
    auto num_edges_32 = static_cast<int32_t>(data.num_edges);

    std::vector<int32_t> row_ptr_32(data.row_ptr.begin(), data.row_ptr.end());
    std::vector<int32_t> col_idx_32(data.col_idx.begin(), data.col_idx.end());

    if (ProcessVertex(vertex_32, row_ptr_32, col_idx_32, data.values, distances, num_vertices_32, num_edges_32)) {
      updated = true;
    }
  }

  MPI_Allreduce(MPI_IN_PLACE, distances.data(), data.num_vertices, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);

  const int local_updated = updated ? 1 : 0;
  int global_updated = 0;
  MPI_Allreduce(&local_updated, &global_updated, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);

  return (global_updated != 0);
}

std::vector<double> RunMPIBellmanFord(int world_size, int rank, GraphData &data) {
  auto distances = InitializeDistances(data.num_vertices, data.source_vertex);

  if (distances.empty() || data.num_vertices <= 0) {
    return distances;
  }

  for (int iter = 0; iter < data.num_vertices - 1; ++iter) {
    if (!PerformBellmanFordIteration(world_size, rank, data, distances)) {
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

BellmanFordCRSMPI::~BellmanFordCRSMPI() = default;

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

  if (world_size <= 0) {
    GetOutput() = RunSequentialVersion(GetInput());
    return true;
  }

  MPI_Barrier(MPI_COMM_WORLD);

  GraphData data = GetGraphData(rank, GetInput());

  BroadcastBasicData(data);
  ResizeBuffers(rank, data);
  BroadcastBuffers(data);

  GetOutput() = RunMPIBellmanFord(world_size, rank, data);

  MPI_Barrier(MPI_COMM_WORLD);

  return true;
}

bool BellmanFordCRSMPI::PostProcessingImpl() {
  int mpi_initialized = 0;
  MPI_Initialized(&mpi_initialized);

  if (mpi_initialized != 0) {
    MPI_Barrier(MPI_COMM_WORLD);
  }

  if (GetOutput().capacity() > GetOutput().size() * 2) {
    GetOutput().shrink_to_fit();
  }

  if (mpi_initialized != 0) {
    MPI_Barrier(MPI_COMM_WORLD);
  }

  return true;
}

}  // namespace artyushkina_bellman_ford_crs
