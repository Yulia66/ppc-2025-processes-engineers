#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

#include "artyushkina_bellman_ford_crs/common/include/common.hpp"
#include "artyushkina_bellman_ford_crs/mpi/include/ops_mpi.hpp"
#include "artyushkina_bellman_ford_crs/seq/include/ops_seq.hpp"

namespace artyushkina_bellman_ford_crs {

namespace {

bool AreDistancesEqual(const std::vector<double> &actual, const std::vector<double> &expected) {
  if (actual.size() != expected.size()) {
    return false;
  }

  for (size_t i = 0; i < actual.size(); ++i) {
    const bool actual_is_inf = std::isinf(actual[i]);
    const bool expected_is_inf = std::isinf(expected[i]);

    if (actual_is_inf && expected_is_inf) {
      continue;
    }

    if (actual_is_inf != expected_is_inf) {
      return false;
    }

    if (std::fabs(actual[i] - expected[i]) > 1e-9) {
      return false;
    }
  }

  return true;
}

}  // namespace

TEST(BellmanFordMPITest, SimpleGraphBasic) {
  CRSGraph graph;
  graph.num_vertices = 4;
  graph.num_edges = 5;
  graph.source_vertex = 0;

  graph.row_ptr = {0, 2, 3, 4, 5};
  graph.col_idx = {1, 2, 2, 3, 0};
  graph.values = {1.0, 4.0, 2.0, 3.0, 1.0};

  std::vector<double> expected = {0.0, 1.0, 3.0, 6.0};

  BellmanFordCRSMPI algorithm(graph);

  EXPECT_TRUE(algorithm.Validation());
  EXPECT_TRUE(algorithm.PreProcessing());
  EXPECT_TRUE(algorithm.Run());
  EXPECT_TRUE(algorithm.PostProcessing());

  auto result = algorithm.GetOutput();
  EXPECT_TRUE(AreDistancesEqual(result, expected));
}

TEST(BellmanFordMPITest, SingleVertex) {
  CRSGraph graph;
  graph.num_vertices = 1;
  graph.num_edges = 0;
  graph.source_vertex = 0;

  graph.row_ptr = {0, 0};
  graph.col_idx = {};
  graph.values = {};

  std::vector<double> expected = {0.0};

  BellmanFordCRSMPI algorithm(graph);

  EXPECT_TRUE(algorithm.Validation());
  EXPECT_TRUE(algorithm.PreProcessing());
  EXPECT_TRUE(algorithm.Run());
  EXPECT_TRUE(algorithm.PostProcessing());

  auto result = algorithm.GetOutput();
  EXPECT_TRUE(AreDistancesEqual(result, expected));
}

TEST(BellmanFordMPITest, DisconnectedGraph) {
  CRSGraph graph;
  graph.num_vertices = 5;
  graph.num_edges = 3;
  graph.source_vertex = 0;

  graph.row_ptr = {0, 1, 2, 2, 2, 3};
  graph.col_idx = {1, 2, 4};
  graph.values = {2.0, 3.0, 1.0};

  std::vector<double> expected = {0.0, 2.0, 3.0, std::numeric_limits<double>::infinity(),
                                  std::numeric_limits<double>::infinity()};

  BellmanFordCRSMPI algorithm(graph);

  EXPECT_TRUE(algorithm.Validation());
  EXPECT_TRUE(algorithm.PreProcessing());
  EXPECT_TRUE(algorithm.Run());
  EXPECT_TRUE(algorithm.PostProcessing());

  auto result = algorithm.GetOutput();
  EXPECT_TRUE(AreDistancesEqual(result, expected));
}

TEST(BellmanFordMPITest, NegativeWeights) {
  CRSGraph graph;
  graph.num_vertices = 3;
  graph.num_edges = 3;
  graph.source_vertex = 0;

  graph.row_ptr = {0, 2, 3, 3};
  graph.col_idx = {1, 2, 2};
  graph.values = {4.0, -1.0, 2.0};

  std::vector<double> expected = {0.0, 4.0, -1.0};

  BellmanFordCRSMPI algorithm(graph);

  EXPECT_TRUE(algorithm.Validation());
  EXPECT_TRUE(algorithm.PreProcessing());
  EXPECT_TRUE(algorithm.Run());
  EXPECT_TRUE(algorithm.PostProcessing());

  auto result = algorithm.GetOutput();
  EXPECT_TRUE(AreDistancesEqual(result, expected));
}

TEST(BellmanFordMPITest, EmptyGraph) {
  CRSGraph graph;
  graph.num_vertices = 0;
  graph.num_edges = 0;
  graph.source_vertex = 0;

  graph.row_ptr = {0};
  graph.col_idx = {};
  graph.values = {};

  std::vector<double> expected = {};

  BellmanFordCRSMPI algorithm(graph);

  EXPECT_TRUE(algorithm.Validation());
  EXPECT_TRUE(algorithm.PreProcessing());
  EXPECT_TRUE(algorithm.Run());
  EXPECT_TRUE(algorithm.PostProcessing());

  auto result = algorithm.GetOutput();
  EXPECT_TRUE(AreDistancesEqual(result, expected));
}

TEST(BellmanFordSEQTest, SimpleGraphBasic) {
  CRSGraph graph;
  graph.num_vertices = 4;
  graph.num_edges = 5;
  graph.source_vertex = 0;

  graph.row_ptr = {0, 2, 3, 4, 5};
  graph.col_idx = {1, 2, 2, 3, 0};
  graph.values = {1.0, 4.0, 2.0, 3.0, 1.0};

  std::vector<double> expected = {0.0, 1.0, 3.0, 6.0};

  BellmanFordCRSSEQ algorithm(graph);

  EXPECT_TRUE(algorithm.Validation());
  EXPECT_TRUE(algorithm.PreProcessing());
  EXPECT_TRUE(algorithm.Run());
  EXPECT_TRUE(algorithm.PostProcessing());

  auto result = algorithm.GetOutput();
  EXPECT_TRUE(AreDistancesEqual(result, expected));
}

TEST(BellmanFordSEQTest, SingleVertex) {
  CRSGraph graph;
  graph.num_vertices = 1;
  graph.num_edges = 0;
  graph.source_vertex = 0;

  graph.row_ptr = {0, 0};
  graph.col_idx = {};
  graph.values = {};

  std::vector<double> expected = {0.0};

  BellmanFordCRSSEQ algorithm(graph);

  EXPECT_TRUE(algorithm.Validation());
  EXPECT_TRUE(algorithm.PreProcessing());
  EXPECT_TRUE(algorithm.Run());
  EXPECT_TRUE(algorithm.PostProcessing());

  auto result = algorithm.GetOutput();
  EXPECT_TRUE(AreDistancesEqual(result, expected));
}

}  // namespace artyushkina_bellman_ford_crs
