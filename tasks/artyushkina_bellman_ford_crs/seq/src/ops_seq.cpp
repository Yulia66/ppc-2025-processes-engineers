#include "artyushkina_bellman_ford_crs/seq/include/ops_seq.hpp"

#include <algorithm>
#include <cstddef>
#include <limits>
#include <vector>

namespace artyushkina_bellman_ford_crs {

BellmanFordCRSSEQ::BellmanFordCRSSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = OutType{};
}

bool BellmanFordCRSSEQ::ValidationImpl() {
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

  if (graph.col_idx.size() != static_cast<size_t>(graph.num_edges)) {
    return false;
  }

  if (graph.values.size() != static_cast<size_t>(graph.num_edges)) {
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

bool BellmanFordCRSSEQ::PostProcessingImpl() {
  if (GetOutput().capacity() > GetOutput().size() * 2) {
    GetOutput().shrink_to_fit();
  }
  return true;
}

}  // namespace artyushkina_bellman_ford_crs
