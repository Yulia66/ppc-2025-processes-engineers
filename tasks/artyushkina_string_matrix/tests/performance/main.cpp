#include <gtest/gtest.h>

#include <algorithm>
#include <climits>
#include <cstddef>
#include <random>

#include "artyushkina_string_matrix/common/include/common.hpp"
#include "artyushkina_string_matrix/mpi/include/ops_mpi.hpp"
#include "artyushkina_string_matrix/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace artyushkina_string_matrix {

class ArtyushkinaRunPerfTestsProcesses : public ppc::util::BaseRunPerfTests<InType, OutType> {
 protected:
  void SetUp() override {
    constexpr int kMatrixSizeRows = 5000;
    constexpr int kMatrixSizeCols = 5000;
    const int rows = kMatrixSizeRows;
    const int cols = kMatrixSizeCols;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dist(-1000, 1000);

    input_data_.resize(rows);
    expected_output_.resize(rows);

    for (int i = 0; i < rows; ++i) {
      input_data_[i].resize(cols);
      int row_min = INT_MAX;

      for (int j = 0; j < cols; ++j) {
        int val = dist(gen);
        input_data_[i][j] = val;
        row_min = std::min(val, row_min);
      }

      expected_output_[i] = row_min;
    }
  }

  bool CheckTestOutputData(OutType &output_data) final {
    if (output_data.empty()) {
      return true;
    }

    if (output_data.size() != expected_output_.size()) {
      return false;
    }

    for (std::size_t i = 0; i < output_data.size(); ++i) {
      if (output_data[i] != expected_output_[i]) {
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
  OutType expected_output_;
};

TEST_P(ArtyushkinaRunPerfTestsProcesses, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks = ppc::util::MakeAllPerfTasks<InType, ArtyushkinaStringMatrixMPI, ArtyushkinaStringMatrixSEQ>(
    PPC_SETTINGS_artyushkina_string_matrix);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = ArtyushkinaRunPerfTestsProcesses::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, ArtyushkinaRunPerfTestsProcesses, kGtestValues, kPerfTestName);

}  // namespace artyushkina_string_matrix
