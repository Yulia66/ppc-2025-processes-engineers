#include <gtest/gtest.h>

#include <cstddef>
#include <utility>
#include <vector>

#include "artyushkina_vector/common/include/common.hpp"
#include "artyushkina_vector/mpi/include/ops_mpi.hpp"
#include "artyushkina_vector/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace artyushkina_vector {

class VerticalStripMatVecPerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
 public:
  static constexpr size_t kSize = 2000;

 protected:
  void SetUp() override {
    matrix_ = Matrix(kSize, std::vector<double>(kSize));
    vector_ = Vector(kSize);

    for (size_t i = 0; i < kSize; ++i) {
      for (size_t j = 0; j < kSize; ++j) {
        if ((i + j) % 5 == 0) {
          matrix_[i][j] = static_cast<double>((i * kSize) + j) * 0.001;
        } else {
          matrix_[i][j] = 0.0;
        }
      }
      vector_[i] = static_cast<double>(i) * 0.002;
    }
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return !output_data.empty();
  }

  InType GetTestInputData() final {
    return std::make_pair(matrix_, vector_);
  }

 private:
  Matrix matrix_;
  Vector vector_;
};

TEST_P(VerticalStripMatVecPerfTests, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks = ppc::util::MakeAllPerfTasks<InType, VerticalStripMatVecMPI, VerticalStripMatVecSEQ>(
    PPC_SETTINGS_artyushkina_vector);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);
const auto kPerfTestName = VerticalStripMatVecPerfTests::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(PerfTests, VerticalStripMatVecPerfTests, kGtestValues, kPerfTestName);

}  // namespace artyushkina_vector
