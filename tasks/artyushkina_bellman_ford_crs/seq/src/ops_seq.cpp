#include "artyushkina_bellman_ford_crs/seq/include/ops_seq.hpp"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

#include "artyushkina_bellman_ford_crs/common/include/common.hpp"

namespace artyushkina_bellman_ford_crs {

BellmanFordCRSSEQ::BellmanFordCRSSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = OutType{};
}

namespace {

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

bool ProcessVertexSeq(int32_t vertex, const CRSGraph &graph, std::vector<double> &distances) {
  const auto vertex_idx = static_cast<size_t>(vertex);

  if (distances[vertex_idx] == std::numeric_limits<double>::infinity()) {
    return false;
  }

  if (vertex_idx + 1 >= graph.row_ptr.size()) {
    return false;
  }

  const int32_t start = graph.row_ptr[vertex_idx];
  const int32_t end = graph.row_ptr[vertex_idx + 1];

  if (start < 0 || end < start || end > graph.num_edges) {
    return false;
  }

  bool updated = false;
  for (int32_t j = start; j < end; ++j) {
    const auto j_idx = static_cast<size_t>(j);
    if (j_idx >= graph.col_idx.size() || j_idx >= graph.values.size()) {
      continue;
    }

    const int32_t v = graph.col_idx[j_idx];
    const double weight = graph.values[j_idx];

    if (v < 0 || v >= graph.num_vertices) {
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

}  // namespace

bool BellmanFordCRSSEQ::ValidationImpl() {
  return ValidateCRSGraph(GetInput());
}

bool BellmanFordCRSSEQ::PreProcessingImpl() {
  GetOutput().clear();
  GetOutput().shrink_to_fit();
  return true;
}

bool BellmanFordCRSSEQ::RunImpl() {
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

    for (int32_t vertex = 0; vertex < graph.num_vertices; ++vertex) {
      if (ProcessVertexSeq(vertex, graph, distances)) {
        updated = true;
      }
    }

    if (!updated) {
      break;
    }
  }

  GetOutput() = distances;
  return true;
}

bool BellmanFordCRSSEQ::PostProcessingImpl() {
  if (GetOutput().capacity() > GetOutput().size() * 2) {
    GetOutput().shrink_to_fit();
  }
  return true;
}

}  // namespace artyushkina_bellman_ford_crs
