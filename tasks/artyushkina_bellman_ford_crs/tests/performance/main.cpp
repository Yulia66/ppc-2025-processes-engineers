#include <gtest/gtest.h>

#include <cstddef>
#include <vector>

#include "artyushkina_bellman_ford_crs/common/include/common.hpp"
#include "artyushkina_bellman_ford_crs/mpi/include/ops_mpi.hpp"
#include "artyushkina_bellman_ford_crs/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace artyushkina_bellman_ford_crs {

class BellmanFordCRSPerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
 public:
  static constexpr size_t kVertices = 500;
  static constexpr size_t kEdgesPerVertex = 10;

 protected:
  void SetUp() override {
    graph_.num_vertices = static_cast<int32_t>(kVertices);
    graph_.source_vertex = 0;

    // Генерируем случайный граф
    graph_.row_ptr.resize(kVertices + 1, 0);
    graph_.col_idx.reserve(kVertices * kEdgesPerVertex);
    graph_.values.reserve(kVertices * kEdgesPerVertex);

    size_t edge_count = 0;
    for (size_t i = 0; i < kVertices; ++i) {
      graph_.row_ptr[i] = static_cast<int32_t>(edge_count);

      // Добавляем несколько случайных исходящих ребер
      for (size_t j = 0; j < kEdgesPerVertex && j < kVertices; ++j) {
        size_t target = (i + j + 1) % kVertices;  // Простая детерминированная генерация
        graph_.col_idx.push_back(static_cast<int32_t>(target));
        graph_.values.push_back(static_cast<double>((i + target) % 10 + 1));  // Вес от 1 до 10
        ++edge_count;
      }
    }
    graph_.row_ptr[kVertices] = static_cast<int32_t>(edge_count);
    graph_.num_edges = static_cast<int32_t>(edge_count);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return !output_data.empty();
  }

  InType GetTestInputData() final {
    return graph_;
  }

 private:
  CRSGraph graph_;
};

TEST_P(BellmanFordCRSPerfTests, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks = ppc::util::MakeAllPerfTasks<InType, BellmanFordCRSMPI, BellmanFordCRSSEQ>(
    PPC_SETTINGS_artyushkina_bellman_ford_crs);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);
const auto kPerfTestName = BellmanFordCRSPerfTests::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(PerfTests, BellmanFordCRSPerfTests, kGtestValues, kPerfTestName);

}  // namespace artyushkina_bellman_ford_crs
