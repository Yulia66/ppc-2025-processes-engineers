#pragma once

#include <cstdint>
#include <tuple>
#include <utility>
#include <vector>

#include "task/include/task.hpp"

namespace artyushkina_bellman_ford_crs {

struct CRSGraph {
  std::vector<int32_t> row_ptr;
  std::vector<int32_t> col_idx;
  std::vector<double> values;
  int32_t num_vertices{0};
  int32_t num_edges{0};
  int32_t source_vertex{0};
};

using InType = CRSGraph;
using OutType = std::vector<double>;
using TestType = std::tuple<int, CRSGraph, std::vector<double>>;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace artyushkina_bellman_ford_crs
