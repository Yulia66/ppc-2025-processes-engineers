#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "artyushkina_vector/common/include/common.hpp"
#include "artyushkina_vector/mpi/include/ops_mpi.hpp"
#include "artyushkina_vector/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace artyushkina_vector {

class VerticalStripMatVecFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    int test_id = std::get<0>(test_param);
    const auto &matrix = std::get<1>(test_param);
    const auto &vector = std::get<2>(test_param);

    return "test_" + std::to_string(test_id) + "_" + std::to_string(matrix.size()) + "x" +
           (matrix.empty() ? "0" : std::to_string(matrix[0].size())) + "_vec_" + std::to_string(vector.size());
  }

 protected:
  void SetUp() override {
    const auto &params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());

    const auto &matrix = std::get<1>(params);
    const auto &vector = std::get<2>(params);

    input_data_ = std::make_pair(matrix, vector);
    expected_ = std::get<3>(params);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    if (output_data.size() != expected_.size()) {
      return false;
    }

    for (size_t i = 0; i < output_data.size(); ++i) {
      if (std::abs(output_data[i] - expected_[i]) > 1e-9) {
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
  OutType expected_;
};

namespace {

TestType CreateVectorTest(int test_id, int rows, int cols) {
  Matrix matrix(rows, std::vector<double>(cols));
  Vector vector(cols);
  Vector result(rows, 0.0);

  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
      matrix[i][j] = (i * cols) + j + 1;
    }
  }

  for (int j = 0; j < cols; ++j) {
    vector[j] = j + 1;
  }

  for (int i = 0; i < rows; ++i) {
    double sum = 0.0;
    for (int j = 0; j < cols; ++j) {
      sum += matrix[i][j] * vector[j];
    }
    result[i] = sum;
  }

  return std::make_tuple(test_id, matrix, vector, result);
}

const std::array<TestType, 7> kTestParam = {
    std::make_tuple(1, Matrix{{1, 2}, {3, 4}}, Vector{5, 6}, Vector{(1 * 5) + (2 * 6), (3 * 5) + (4 * 6)}),

    std::make_tuple(2, Matrix{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}}, Vector{2, 3, 4}, Vector{2, 3, 4}),

    CreateVectorTest(3, 2, 3),

    CreateVectorTest(4, 4, 2),

    std::make_tuple(5, Matrix{{0, 0, 0}, {0, 0, 0}}, Vector{1, 2, 3}, Vector{0, 0}),

    std::make_tuple(6, Matrix{{2.5}}, Vector{3.0}, Vector{7.5}),

    CreateVectorTest(7, 1, 5)};

TEST_P(VerticalStripMatVecFuncTests, MatrixVectorMultiplication) {
  ExecuteTest(GetParam());
}

const auto kTestTasksList =
    std::tuple_cat(ppc::util::AddFuncTask<VerticalStripMatVecMPI, InType>(kTestParam, PPC_SETTINGS_artyushkina_vector),
                   ppc::util::AddFuncTask<VerticalStripMatVecSEQ, InType>(kTestParam, PPC_SETTINGS_artyushkina_vector));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);
const auto kPerfTestName = VerticalStripMatVecFuncTests::PrintFuncTestName<VerticalStripMatVecFuncTests>;

INSTANTIATE_TEST_SUITE_P(VectorMultiplicationTests, VerticalStripMatVecFuncTests, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace artyushkina_vector
