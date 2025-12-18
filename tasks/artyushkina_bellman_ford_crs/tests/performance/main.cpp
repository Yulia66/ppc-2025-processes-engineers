#include <gtest/gtest.h>

#include <cstddef>
#include <vector>

#include "artyushkina_bellman_ford_crs/common/include/common.hpp"
#include "artyushkina_bellman_ford_crs/mpi/include/ops_mpi.hpp"
#include "artyushkina_bellman_ford_crs/seq/include/ops_seq.hpp"

namespace artyushkina_bellman_ford_crs {

class BellmanFordCRSPerfTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Создаем тестовый граф
    graph_.num_vertices = 1000;
    graph_.source_vertex = 0;

    // Генерируем простой граф
    size_t edge_count = 0;
    graph_.row_ptr.resize(graph_.num_vertices + 1, 0);

    // Каждая вершина имеет 5 случайных исходящих ребер
    const size_t edges_per_vertex = 5;

    for (int32_t i = 0; i < graph_.num_vertices; ++i) {
      graph_.row_ptr[i] = static_cast<int32_t>(edge_count);

      for (size_t j = 0; j < edges_per_vertex && j < static_cast<size_t>(graph_.num_vertices); ++j) {
        int32_t target = (i + j + 1) % graph_.num_vertices;
        graph_.col_idx.push_back(target);
        graph_.values.push_back(static_cast<double>((i + target) % 10 + 1));
        ++edge_count;
      }
    }
    graph_.row_ptr[graph_.num_vertices] = static_cast<int32_t>(edge_count);
    graph_.num_edges = static_cast<int32_t>(edge_count);
  }

  CRSGraph graph_;
};

TEST_F(BellmanFordCRSPerfTest, SequentialPerformance) {
  BellmanFordCRSSEQ algorithm(graph_);

  EXPECT_TRUE(algorithm.Validation());

  // Измеряем время выполнения
  auto start = std::chrono::high_resolution_clock::now();
  algorithm.Run();
  auto end = std::chrono::high_resolution_clock::now();

  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

  auto result = algorithm.GetOutput();

  // Проверяем базовые свойства результата
  EXPECT_EQ(result.size(), static_cast<size_t>(graph_.num_vertices));
  EXPECT_DOUBLE_EQ(result[graph_.source_vertex], 0.0);

  // Логируем время выполнения
  std::cout << "SEQ Time: " << duration.count() << " ms" << std::endl;
}

// MPI тест только если не под Valgrind
#ifndef VALGRIND_TEST
TEST_F(BellmanFordCRSPerfTest, MPIPerformance) {
  BellmanFordCRSMPI algorithm(graph_);

  EXPECT_TRUE(algorithm.Validation());

  // Измеряем время выполнения
  auto start = std::chrono::high_resolution_clock::now();
  algorithm.Run();
  auto end = std::chrono::high_resolution_clock::now();

  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

  auto result = algorithm.GetOutput();

  // Проверяем базовые свойства результата
  EXPECT_EQ(result.size(), static_cast<size_t>(graph_.num_vertices));
  EXPECT_DOUBLE_EQ(result[graph_.source_vertex], 0.0);

  // Логируем время выполнения
  std::cout << "MPI Time: " << duration.count() << " ms" << std::endl;
}
#endif

}  // namespace artyushkina_bellman_ford_crs
