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

namespace artyushkina_bellman_ford_crs {

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

CRSGraph CreateEmptyGraph() {
  CRSGraph graph;
  graph.num_vertices = 0;
  graph.num_edges = 0;
  graph.source_vertex = 0;
  graph.row_ptr = {0};
  graph.col_idx = {};
  graph.values = {};
  return graph;
}

// Объявление функции для генерации имен тестов
std::string SEQTestNamingFunction(const testing::TestParamInfo<std::tuple<int, CRSGraph, std::vector<double>>> &info);

}  // namespace

// Простые тесты без сложных шаблонов
class BellmanFordSEQTest : public ::testing::TestWithParam<std::tuple<int, CRSGraph, std::vector<double>>> {
 protected:
  void SetUp() override {
    test_param_ = GetParam();
    input_data_ = std::get<1>(test_param_);
    expected_ = std::get<2>(test_param_);
  }

  bool CheckOutput(const OutType &output) {
    if (output.size() != expected_.size()) {
      return false;
    }

    for (size_t i = 0; i < output.size(); ++i) {
      if (std::isnan(output[i]) || std::isnan(expected_[i])) {
        return false;
      }

      bool output_is_inf = std::isinf(output[i]);
      bool expected_is_inf = std::isinf(expected_[i]);

      if (output_is_inf && expected_is_inf) {
        // Оба inf, проверяем знак
        if (std::signbit(output[i]) != std::signbit(expected_[i])) {
          return false;
        }
        continue;
      } else if (output_is_inf != expected_is_inf) {
        return false;
      }

      if (std::abs(output[i] - expected_[i]) > 1e-9) {
        return false;
      }
    }

    return true;
  }

  std::tuple<int, CRSGraph, std::vector<double>> test_param_;
  InType input_data_;
  OutType expected_;
};

TEST_P(BellmanFordSEQTest, SequentialAlgorithm) {
  BellmanFordCRSSEQ algorithm(input_data_);

  EXPECT_TRUE(algorithm.Validation());

  algorithm.Run();

  auto output = algorithm.GetOutput();
  EXPECT_TRUE(CheckOutput(output));
}

const std::array<std::tuple<int, CRSGraph, std::vector<double>>, 5> kSEQTestCases = {
    {std::make_tuple(1, CreateSimpleGraph(), std::vector<double>{0.0, 1.0, 3.0, 6.0}),
     std::make_tuple(2, CreateGraphWithNegativeWeights(), std::vector<double>{0.0, -1.0, 2.0}),
     std::make_tuple(3, CreateDisconnectedGraph(),
                     std::vector<double>{0.0, 1.0, std::numeric_limits<double>::infinity(),
                                         std::numeric_limits<double>::infinity()}),
     std::make_tuple(4, CreateSingleVertexGraph(), std::vector<double>{0.0}),
     std::make_tuple(5, CreateEmptyGraph(), std::vector<double>{})}};

namespace {

// Реализация функции для генерации имен тестов
std::string SEQTestNamingFunction(const testing::TestParamInfo<std::tuple<int, CRSGraph, std::vector<double>>> &info) {
  int test_id = std::get<0>(info.param);
  return "SEQ_Test_" + std::to_string(test_id);
}

}  // namespace

INSTANTIATE_TEST_SUITE_P(SEQTests, BellmanFordSEQTest, testing::ValuesIn(kSEQTestCases), SEQTestNamingFunction);

// Простые тесты без параметризации
TEST(BellmanFordSEQ, SimpleGraphDirect) {
  CRSGraph graph = CreateSimpleGraph();
  BellmanFordCRSSEQ algorithm(graph);

  EXPECT_TRUE(algorithm.Validation());
  algorithm.Run();

  auto result = algorithm.GetOutput();
  std::vector<double> expected = {0.0, 1.0, 3.0, 6.0};

  ASSERT_EQ(result.size(), expected.size());
  for (size_t i = 0; i < result.size(); ++i) {
    EXPECT_NEAR(result[i], expected[i], 1e-9);
  }
}

TEST(BellmanFordSEQ, NegativeWeights) {
  CRSGraph graph = CreateGraphWithNegativeWeights();
  BellmanFordCRSSEQ algorithm(graph);

  EXPECT_TRUE(algorithm.Validation());
  algorithm.Run();

  auto result = algorithm.GetOutput();
  std::vector<double> expected = {0.0, -1.0, 2.0};

  ASSERT_EQ(result.size(), expected.size());
  for (size_t i = 0; i < result.size(); ++i) {
    EXPECT_NEAR(result[i], expected[i], 1e-9);
  }
}

TEST(BellmanFordSEQ, DisconnectedGraph) {
  CRSGraph graph = CreateDisconnectedGraph();
  BellmanFordCRSSEQ algorithm(graph);

  EXPECT_TRUE(algorithm.Validation());
  algorithm.Run();

  auto result = algorithm.GetOutput();
  std::vector<double> expected = {0.0, 1.0, std::numeric_limits<double>::infinity(),
                                  std::numeric_limits<double>::infinity()};

  ASSERT_EQ(result.size(), expected.size());
  for (size_t i = 0; i < result.size(); ++i) {
    if (std::isinf(expected[i])) {
      EXPECT_TRUE(std::isinf(result[i]));
      EXPECT_EQ(std::signbit(result[i]), std::signbit(expected[i]));
    } else {
      EXPECT_NEAR(result[i], expected[i], 1e-9);
    }
  }
}

TEST(BellmanFordSEQ, SingleVertex) {
  CRSGraph graph = CreateSingleVertexGraph();
  BellmanFordCRSSEQ algorithm(graph);

  EXPECT_TRUE(algorithm.Validation());
  algorithm.Run();

  auto result = algorithm.GetOutput();
  std::vector<double> expected = {0.0};

  ASSERT_EQ(result.size(), expected.size());
  EXPECT_NEAR(result[0], expected[0], 1e-9);
}

TEST(BellmanFordSEQ, EmptyGraph) {
  CRSGraph graph = CreateEmptyGraph();
  BellmanFordCRSSEQ algorithm(graph);

  EXPECT_TRUE(algorithm.Validation());
  algorithm.Run();

  auto result = algorithm.GetOutput();
  EXPECT_TRUE(result.empty());
}

}  // namespace artyushkina_bellman_ford_crs
