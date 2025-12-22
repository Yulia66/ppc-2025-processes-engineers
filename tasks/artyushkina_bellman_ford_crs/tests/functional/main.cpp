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
  graph.num_vertices = 4;
  graph.num_edges = 2;
  graph.source_vertex = 0;

  graph.row_ptr = {0, 1, 2, 2, 2};
  graph.col_idx = {1, 2};
  graph.values = {2.0, 3.0};

  std::vector<double> expected = {0.0, 2.0, 5.0, std::numeric_limits<double>::infinity()};

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

TEST(BellmanFordValidationTest, NegativeVertices) {
  CRSGraph invalid_graph;
  invalid_graph.num_vertices = -1;
  invalid_graph.num_edges = 0;
  invalid_graph.source_vertex = 0;
  invalid_graph.row_ptr = {0};
  invalid_graph.col_idx = {};
  invalid_graph.values = {};

  BellmanFordCRSSEQ seq_invalid(invalid_graph);
  EXPECT_FALSE(seq_invalid.Validation());

  BellmanFordCRSMPI mpi_invalid(invalid_graph);
  EXPECT_FALSE(mpi_invalid.Validation());
}

TEST(BellmanFordValidationTest, InvalidSourceVertex) {
  CRSGraph graph;
  graph.num_vertices = 3;
  graph.num_edges = 2;
  graph.source_vertex = 5;
  graph.row_ptr = {0, 1, 2, 2};
  graph.col_idx = {1, 2};
  graph.values = {1.0, 2.0};

  BellmanFordCRSSEQ seq_algo(graph);
  EXPECT_FALSE(seq_algo.Validation());

  BellmanFordCRSMPI mpi_algo(graph);
  EXPECT_FALSE(mpi_algo.Validation());
}

TEST(BellmanFordValidationTest, NonMonotonicRowPtr) {
  CRSGraph graph;
  graph.num_vertices = 2;
  graph.num_edges = 1;
  graph.source_vertex = 0;
  graph.row_ptr = {0, 2, 1};
  graph.col_idx = {1};
  graph.values = {1.0};

  BellmanFordCRSSEQ seq_algo(graph);
  EXPECT_FALSE(seq_algo.Validation());
}

TEST(BellmanFordValidationTest, TwoVerticesNoEdges) {
  CRSGraph graph;
  graph.num_vertices = 2;
  graph.num_edges = 0;
  graph.source_vertex = 0;
  graph.row_ptr = {0, 0, 0};
  graph.col_idx = {};
  graph.values = {};

  BellmanFordCRSSEQ seq_algo(graph);
  EXPECT_TRUE(seq_algo.Validation());
  EXPECT_TRUE(seq_algo.PreProcessing());
  EXPECT_TRUE(seq_algo.Run());
  EXPECT_TRUE(seq_algo.PostProcessing());

  auto result = seq_algo.GetOutput();
  EXPECT_EQ(result.size(), 2u);

  EXPECT_DOUBLE_EQ(result[0], 0.0);
  EXPECT_TRUE(std::isinf(result[1]));
}

TEST(BellmanFordValidationTest, GraphWithNegativeCycle) {
  CRSGraph graph;
  graph.num_vertices = 3;
  graph.num_edges = 3;
  graph.source_vertex = 0;
  graph.row_ptr = {0, 1, 2, 3};
  graph.col_idx = {1, 2, 0};
  graph.values = {1.0, 2.0, -4.0};

  BellmanFordCRSSEQ seq_algo(graph);
  EXPECT_TRUE(seq_algo.Validation());
  EXPECT_TRUE(seq_algo.PreProcessing());
  EXPECT_TRUE(seq_algo.Run());
  EXPECT_TRUE(seq_algo.PostProcessing());

  auto result = seq_algo.GetOutput();
  EXPECT_EQ(result.size(), 3u);
}

}  // namespace artyushkina_bellman_ford_crs
