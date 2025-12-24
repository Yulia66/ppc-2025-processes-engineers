#include <gtest/gtest.h>
#include <mpi.h>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <vector>

#include "artyushkina_bellman_ford_crs/common/include/common.hpp"
#include "artyushkina_bellman_ford_crs/mpi/include/ops_mpi.hpp"
#include "artyushkina_bellman_ford_crs/seq/include/ops_seq.hpp"

namespace artyushkina_bellman_ford_crs {

TEST(BellmanFordPerformance, SequentialSmallGraph) {
  const int num_vertices = 50;
  const int edges_per_vertex = 3;

  CRSGraph graph;
  graph.num_vertices = num_vertices;
  graph.source_vertex = 0;

  const size_t row_ptr_size = static_cast<size_t>(num_vertices) + 1;
  graph.row_ptr.resize(row_ptr_size, 0);
  size_t edge_count = 0;

  for (int i = 0; i < num_vertices; ++i) {
    graph.row_ptr[static_cast<size_t>(i)] = static_cast<int32_t>(edge_count);

    for (int j = 0; j < edges_per_vertex; ++j) {
      if (j < num_vertices) {
        int target = (i + j + 1) % num_vertices;
        graph.col_idx.push_back(static_cast<int32_t>(target));
        graph.values.push_back(static_cast<double>(((i + target) % 10) + 1));
        ++edge_count;
      }
    }
  }

  graph.row_ptr[static_cast<size_t>(num_vertices)] = static_cast<int32_t>(edge_count);
  graph.num_edges = static_cast<int32_t>(edge_count);

  BellmanFordCRSSEQ algorithm(graph);

  algorithm.Validation();
  algorithm.PreProcessing();

  auto start = std::chrono::high_resolution_clock::now();
  algorithm.Run();
  auto end = std::chrono::high_resolution_clock::now();

  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

  auto result = algorithm.GetOutput();
  EXPECT_EQ(result.size(), static_cast<size_t>(graph.num_vertices));

  algorithm.PostProcessing();

  std::cout << "SEQ Time for " << num_vertices << " vertices: " << duration.count() << " microseconds\n";
}

#ifndef RUNNING_UNDER_VALGRIND
TEST(BellmanFordPerformance, MPISmallGraph) {
  int mpi_initialized = 0;
  MPI_Initialized(&mpi_initialized);

  if (!mpi_initialized) {
    GTEST_SKIP() << "MPI not initialized, skipping MPI performance test";
  }

  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  const int num_vertices = 50;
  const int edges_per_vertex = 3;

  CRSGraph graph;

  if (rank == 0) {
    graph.num_vertices = num_vertices;
    graph.source_vertex = 0;

    const size_t row_ptr_size = static_cast<size_t>(num_vertices) + 1;
    graph.row_ptr.resize(row_ptr_size, 0);
    size_t edge_count = 0;

    for (int i = 0; i < num_vertices; ++i) {
      graph.row_ptr[static_cast<size_t>(i)] = static_cast<int32_t>(edge_count);

      for (int j = 0; j < edges_per_vertex; ++j) {
        if (j < num_vertices) {
          int target = (i + j + 1) % num_vertices;
          graph.col_idx.push_back(static_cast<int32_t>(target));
          graph.values.push_back(static_cast<double>(((i + target) % 10) + 1));
          ++edge_count;
        }
      }
    }

    graph.row_ptr[static_cast<size_t>(num_vertices)] = static_cast<int32_t>(edge_count);
    graph.num_edges = static_cast<int32_t>(edge_count);
  }

  BellmanFordCRSMPI algorithm(graph);

  algorithm.Validation();
  algorithm.PreProcessing();

  std::chrono::high_resolution_clock::time_point start, end;
  if (rank == 0) {
    start = std::chrono::high_resolution_clock::now();
  }

  algorithm.Run();

  if (rank == 0) {
    end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    auto result = algorithm.GetOutput();
    EXPECT_EQ(result.size(), static_cast<size_t>(graph.num_vertices));

    std::cout << "MPI Time for " << num_vertices << " vertices: " << duration.count() << " microseconds\n";
  }

  algorithm.PostProcessing();

  MPI_Barrier(MPI_COMM_WORLD);
}
#endif

}  // namespace artyushkina_bellman_ford_crs
