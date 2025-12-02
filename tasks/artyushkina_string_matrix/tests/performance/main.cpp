#include <gtest/gtest.h>

#include <random>
#include <vector>

#include "artyushkina_string_matrix/common/include/common.hpp"
#include "artyushkina_string_matrix/mpi/include/ops_mpi.hpp"
#include "artyushkina_string_matrix/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace artyushkina_string_matrix {

class ArtyushkinaRunPerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
 protected:
  void SetUp() override {
    // Генерация большой матрицы для тестов производительности
    const int rows = 1000;
    const int cols = 1000;
    input_data_.resize(rows);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dist(1, 1000);

    for (int i = 0; i < rows; ++i) {
      input_data_[i].resize(cols);
      for (int j = 0; j < cols; ++j) {
        input_data_[i][j] = dist(gen);
      }
    }
  }

  bool CheckTestOutputData(OutType &output_data) final {
    // Проверяем, что количество минимумов равно количеству строк
    return output_data.size() == input_data_.size();
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
};

TEST_P(ArtyushkinaRunPerfTests, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks = ppc::util::MakeAllPerfTasks<InType, ArtyushkinaATestTaskMPI, ArtyushkinaATestTaskSEQ>(
    PPC_SETTINGS_artyushkina_string_matrix);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = ArtyushkinaRunPerfTests::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, ArtyushkinaRunPerfTests, kGtestValues, kPerfTestName);

}  // namespace artyushkina_string_matrix
