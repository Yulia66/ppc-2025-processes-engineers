#include <gtest/gtest.h>

#include <chrono>
#include <cstddef>
#include <vector>

#include "artyushkina_bellman_ford_crs/common/include/common.hpp"
#include "artyushkina_bellman_ford_crs/mpi/include/ops_mpi.hpp"
#include "artyushkina_bellman_ford_crs/seq/include/ops_seq.hpp"

namespace artyushkina_bellman_ford_crs {

namespace {

// Объявление функции
CRSGraph CreateTestGraph(int num_vertices, int edges_per_vertex);

}  // namespace

// SEQ performance тест
TEST(BellmanFordPerformance, SequentialSmallGraph) {
  // Используем небольшие числа для теста
  const int num_vertices = 50;
  const int edges_per_vertex = 3;

  // Создаем граф вручную в тесте
  CRSGraph graph;
  graph.num_vertices = num_vertices;
  graph.source_vertex = 0;

  graph.row_ptr.resize(static_cast<size_t>(num_vertices + 1), 0);
  size_t edge_count = 0;

  for (int i = 0; i < num_vertices; ++i) {
    graph.row_ptr[static_cast<size_t>(i)] = static_cast<int32_t>(edge_count);

    for (int j = 0; j < edges_per_vertex && j < num_vertices; ++j) {
      int target = (i + j + 1) % num_vertices;
      graph.col_idx.push_back(static_cast<int32_t>(target));
      graph.values.push_back(static_cast<double>((i + target) % 10 + 1));
      ++edge_count;
    }
  }

  graph.row_ptr[static_cast<size_t>(num_vertices)] = static_cast<int32_t>(edge_count);
  graph.num_edges = static_cast<int32_t>(edge_count);

  BellmanFordCRSSEQ algorithm(graph);

  EXPECT_TRUE(algorithm.Validation());

  auto start = std::chrono::high_resolution_clock::now();
  algorithm.Run();
  auto end = std::chrono::high_resolution_clock::now();

  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

  auto result = algorithm.GetOutput();
  EXPECT_EQ(result.size(), static_cast<size_t>(graph.num_vertices));

  // Просто выводим время, не проверяем
  std::cout << "SEQ Time for " << num_vertices << " vertices: " << duration.count() << " microseconds" << std::endl;
}

// MPI performance тест - только если не под Valgrind
#ifndef RUNNING_UNDER_VALGRIND
TEST(BellmanFordPerformance, MPISmallGraph) {
  const int num_vertices = 50;
  const int edges_per_vertex = 3;

  // Создаем граф вручную в тесте
  CRSGraph graph;
  graph.num_vertices = num_vertices;
  graph.source_vertex = 0;

  graph.row_ptr.resize(static_cast<size_t>(num_vertices + 1), 0);
  size_t edge_count = 0;

  for (int i = 0; i < num_vertices; ++i) {
    graph.row_ptr[static_cast<size_t>(i)] = static_cast<int32_t>(edge_count);

    for (int j = 0; j < edges_per_vertex && j < num_vertices; ++j) {
      int target = (i + j + 1) % num_vertices;
      graph.col_idx.push_back(static_cast<int32_t>(target));
      graph.values.push_back(static_cast<double>((i + target) % 10 + 1));
      ++edge_count;
    }
  }

  graph.row_ptr[static_cast<size_t>(num_vertices)] = static_cast<int32_t>(edge_count);
  graph.num_edges = static_cast<int32_t>(edge_count);

  BellmanFordCRSMPI algorithm(graph);

  EXPECT_TRUE(algorithm.Validation());

  auto start = std::chrono::high_resolution_clock::now();
  algorithm.Run();
  auto end = std::chrono::high_resolution_clock::now();

  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

  auto result = algorithm.GetOutput();
  EXPECT_EQ(result.size(), static_cast<size_t>(graph.num_vertices));

  std::cout << "MPI Time for " << num_vertices << " vertices: " << duration.count() << " microseconds" << std::endl;
}
#endif

}  // namespace artyushkina_bellman_ford_crs
