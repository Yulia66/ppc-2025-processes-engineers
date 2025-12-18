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
    return std::to_string(test_id);
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
        continue;
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
    {TestType{1, CreateSimpleGraph(), std::vector<double>{0.0, 1.0, 3.0, 6.0}},
     TestType{2, CreateGraphWithNegativeWeights(), std::vector<double>{0.0, -1.0, 2.0}},
     TestType{3, CreateDisconnectedGraph(),
              std::vector<double>{0.0, 1.0, std::numeric_limits<double>::infinity(),
                                  std::numeric_limits<double>::infinity()}},
     TestType{4, CreateSingleVertexGraph(), std::vector<double>{0.0}}}};

TEST_P(BellmanFordCRSFuncTests, BellmanFordAlgorithm) {
  ExecuteTest(GetParam());
}

const auto kTestTasksList =
    ppc::util::AddFuncTask<BellmanFordCRSSEQ, InType>(kTestParam, PPC_SETTINGS_artyushkina_bellman_ford_crs);

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

std::string TestNamingFunction(
    const testing::TestParamInfo<std::tuple<std::function<std::shared_ptr<ppc::task::Task<InType, OutType>>(InType)>,
                                            std::string, TestType>> &info) {
  const auto &test_case = std::get<2>(info.param);
  int test_id = std::get<0>(test_case);
  const std::string &task_name = std::get<1>(info.param);

  return "Test_" + std::to_string(test_id) + "_" + task_name;
}

INSTANTIATE_TEST_SUITE_P(BellmanFordTests, BellmanFordCRSFuncTests, kGtestValues, TestNamingFunction);

}  // namespace

}  // namespace artyushkina_bellman_ford_crs
