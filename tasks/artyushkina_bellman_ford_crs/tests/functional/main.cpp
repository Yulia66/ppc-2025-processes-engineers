#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <string>
#include <tuple>
#include <vector>

#include "artyushkina_bellman_ford_crs/common/include/common.hpp"
#include "artyushkina_bellman_ford_crs/mpi/include/ops_mpi.hpp"
#include "artyushkina_bellman_ford_crs/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace artyushkina_bellman_ford_crs {

class BellmanFordCRSFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    int test_id = std::get<0>(test_param);
    const auto &graph = std::get<1>(test_param);
    return "test_" + std::to_string(test_id) + "_" + std::to_string(graph.num_vertices) + "v_" +
           std::to_string(graph.num_edges) + "e";
  }

 protected:
  void SetUp() override {
    const auto &params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    input_data_ = std::get<1>(params);
    expected_ = std::get<2>(params);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    if (output_data.size() != expected_.size()) {
      return false;
    }

    for (size_t i = 0; i < output_data.size(); ++i) {
      if (std::isnan(output_data[i]) || std::isnan(expected_[i])) {
        return false;
      }

      bool output_is_inf = std::isinf(output_data[i]);
      bool expected_is_inf = std::isinf(expected_[i]);

      if (output_is_inf && expected_is_inf) {
        if ((output_data[i] > 0 && expected_[i] > 0) || (output_data[i] < 0 && expected_[i] < 0)) {
          continue;
        } else {
          return false;
        }
      } else if (output_is_inf != expected_is_inf) {
        return false;
      }

      if (std::abs(output_data[i] - expected_[i]) > 1e-9) {
        return false;
      }
    }

    return true;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
  OutType expected_;
};

namespace {

CRSGraph CreateSimpleGraph() {
  CRSGraph graph;
  graph.num_vertices = 4;
  graph.num_edges = 5;
  graph.source_vertex = 0;

  graph.row_ptr = {0, 2, 4, 5, 5};
  graph.col_idx = {1, 2, 2, 3, 3};
  graph.values = {1.0, 4.0, 2.0, 6.0, 3.0};

  return graph;
}

CRSGraph CreateGraphWithNegativeWeights() {
  CRSGraph graph;
  graph.num_vertices = 3;
  graph.num_edges = 3;
  graph.source_vertex = 0;

  graph.row_ptr = {0, 2, 3, 3};
  graph.col_idx = {1, 2, 2};
  graph.values = {-1.0, 4.0, 3.0};

  return graph;
}

CRSGraph CreateDisconnectedGraph() {
  CRSGraph graph;
  graph.num_vertices = 4;
  graph.num_edges = 2;
  graph.source_vertex = 0;

  graph.row_ptr = {0, 1, 1, 2, 2};
  graph.col_idx = {1, 3};
  graph.values = {1.0, 1.0};

  return graph;
}

CRSGraph CreateSingleVertexGraph() {
  CRSGraph graph;
  graph.num_vertices = 1;
  graph.num_edges = 0;
  graph.source_vertex = 0;
  graph.row_ptr = {0, 0};
  graph.col_idx = {};
  graph.values = {};
  return graph;
}

const std::array<TestType, 4> kTestParam = {
    std::make_tuple(1, CreateSimpleGraph(), std::vector<double>{0.0, 1.0, 3.0, 6.0}),

    std::make_tuple(2, CreateGraphWithNegativeWeights(), std::vector<double>{0.0, -1.0, 2.0}),

    std::make_tuple(3, CreateDisconnectedGraph(),
                    std::vector<double>{0.0, 1.0, std::numeric_limits<double>::infinity(),
                                        std::numeric_limits<double>::infinity()}),

    std::make_tuple(4, CreateSingleVertexGraph(), std::vector<double>{0.0})};

TEST_P(BellmanFordCRSFuncTests, BellmanFordAlgorithm) {
  ExecuteTest(GetParam());
}

const auto kTestTasksList = std::tuple_cat(
    ppc::util::AddFuncTask<BellmanFordCRSMPI, InType>(kTestParam, PPC_SETTINGS_artyushkina_bellman_ford_crs),
    ppc::util::AddFuncTask<BellmanFordCRSSEQ, InType>(kTestParam, PPC_SETTINGS_artyushkina_bellman_ford_crs));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);
const auto kPerfTestName = BellmanFordCRSFuncTests::PrintFuncTestName<BellmanFordCRSFuncTests>;

INSTANTIATE_TEST_SUITE_P(BellmanFordTests, BellmanFordCRSFuncTests, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace artyushkina_bellman_ford_crs
