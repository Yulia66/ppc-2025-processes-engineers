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

bool CompareResults(const std::vector<double> &actual, const std::vector<double> &expected) {
  if (actual.size() != expected.size()) {
    return false;
  }

  for (size_t i = 0; i < actual.size(); ++i) {
    if (std::isnan(actual[i]) || std::isnan(expected[i])) {
      return false;
    }

    bool actual_is_inf = std::isinf(actual[i]);
    bool expected_is_inf = std::isinf(expected[i]);

    if (actual_is_inf && expected_is_inf) {
      if (std::signbit(actual[i]) != std::signbit(expected[i])) {
        return false;
      }
      continue;
    } else if (actual_is_inf != expected_is_inf) {
      return false;
    }

    if (std::abs(actual[i] - expected[i]) > 1e-9) {
      return false;
    }
  }

  return true;
}

}  // namespace

TEST(BellmanFordSEQTest, SimpleGraph) {
  CRSGraph graph = CreateSimpleGraph();
  BellmanFordCRSSEQ algorithm(graph);

  EXPECT_TRUE(algorithm.Validation());
  algorithm.Run();

  auto result = algorithm.GetOutput();
  std::vector<double> expected = {0.0, 1.0, 3.0, 6.0};

  EXPECT_TRUE(CompareResults(result, expected));
}

TEST(BellmanFordSEQTest, NegativeWeights) {
  CRSGraph graph = CreateGraphWithNegativeWeights();
  BellmanFordCRSSEQ algorithm(graph);

  EXPECT_TRUE(algorithm.Validation());
  algorithm.Run();

  auto result = algorithm.GetOutput();
  std::vector<double> expected = {0.0, -1.0, 2.0};

  EXPECT_TRUE(CompareResults(result, expected));
}

TEST(BellmanFordSEQTest, DisconnectedGraph) {
  CRSGraph graph = CreateDisconnectedGraph();
  BellmanFordCRSSEQ algorithm(graph);

  EXPECT_TRUE(algorithm.Validation());
  algorithm.Run();

  auto result = algorithm.GetOutput();
  std::vector<double> expected = {0.0, 1.0, std::numeric_limits<double>::infinity(),
                                  std::numeric_limits<double>::infinity()};

  EXPECT_TRUE(CompareResults(result, expected));
}

TEST(BellmanFordSEQTest, SingleVertex) {
  CRSGraph graph = CreateSingleVertexGraph();
  BellmanFordCRSSEQ algorithm(graph);

  EXPECT_TRUE(algorithm.Validation());
  algorithm.Run();

  auto result = algorithm.GetOutput();
  std::vector<double> expected = {0.0};

  EXPECT_TRUE(CompareResults(result, expected));
}

TEST(BellmanFordSEQTest, EmptyGraph) {
  CRSGraph graph = CreateEmptyGraph();
  BellmanFordCRSSEQ algorithm(graph);

  EXPECT_TRUE(algorithm.Validation());
  algorithm.Run();

  auto result = algorithm.GetOutput();
  EXPECT_TRUE(result.empty());
}

TEST(BellmanFordMPITest, SimpleGraphBasic) {
  CRSGraph graph = CreateSimpleGraph();
  BellmanFordCRSMPI algorithm(graph);

#ifndef RUNNING_UNDER_VALGRIND
  EXPECT_TRUE(algorithm.Validation());
  algorithm.Run();

  auto result = algorithm.GetOutput();
  EXPECT_EQ(result.size(), static_cast<size_t>(graph.num_vertices));
#endif
}

TEST(BellmanFordMPITest, SingleVertexBasic) {
  CRSGraph graph = CreateSingleVertexGraph();
  BellmanFordCRSMPI algorithm(graph);

#ifndef RUNNING_UNDER_VALGRIND
  EXPECT_TRUE(algorithm.Validation());
  algorithm.Run();

  auto result = algorithm.GetOutput();
  EXPECT_EQ(result.size(), 1u);
#endif
}

}  // namespace artyushkina_bellman_ford_crs
