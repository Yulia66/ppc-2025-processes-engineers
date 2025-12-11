#include <gtest/gtest.h>

#include <utility>
#include <vector>

#include "artyushkina_string_matrix/common/include/common.hpp"
#include "artyushkina_string_matrix/mpi/include/ops_mpi.hpp"
#include "artyushkina_string_matrix/seq/include/ops_seq.hpp"
#include "task/include/task.hpp"

namespace artyushkina_string_matrix {

namespace {

std::pair<InType, OutType> GetTestData(int test_num) {
  switch (test_num) {
    case 1:
      return {{{3, 1}, {4, 2}}, {1, 2}};
    case 2:
      return {{{-5, 10, -3}, {8, -2, 0}}, {-5, -2}};
    case 3:
      return {{{5, 2, 8, 1, 9}}, {1}};
    case 4:
      return {{{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}}, {1, 5, 9}};
    default:
      return {{}, {}};
  }
}

}  // namespace

// ==================== SEQ ТЕСТЫ ====================

TEST(ArtyushkinaFunctional, Test1) {
  const auto [matrix, expected] = GetTestData(1);

  ArtyushkinaStringMatrixSEQ task(matrix);
  EXPECT_TRUE(task.Validation());
  EXPECT_TRUE(task.PreProcessing());
  EXPECT_TRUE(task.Run());

  const auto &result = task.GetOutput();
  EXPECT_EQ(result, expected);
}

TEST(ArtyushkinaFunctional, Test2) {
  const auto [matrix, expected] = GetTestData(2);

  ArtyushkinaStringMatrixSEQ task(matrix);
  EXPECT_TRUE(task.Validation());
  EXPECT_TRUE(task.PreProcessing());
  EXPECT_TRUE(task.Run());

  const auto &result = task.GetOutput();
  EXPECT_EQ(result, expected);
}

TEST(ArtyushkinaFunctional, Test3) {
  const auto [matrix, expected] = GetTestData(3);

  ArtyushkinaStringMatrixSEQ task(matrix);
  EXPECT_TRUE(task.Validation());
  EXPECT_TRUE(task.PreProcessing());
  EXPECT_TRUE(task.Run());

  const auto &result = task.GetOutput();
  EXPECT_EQ(result, expected);
}

TEST(ArtyushkinaFunctional, Test4) {
  const auto [matrix, expected] = GetTestData(4);

  ArtyushkinaStringMatrixSEQ task(matrix);
  EXPECT_TRUE(task.Validation());
  EXPECT_TRUE(task.PreProcessing());
  EXPECT_TRUE(task.Run());

  const auto &result = task.GetOutput();
  EXPECT_EQ(result, expected);
}

TEST(ArtyushkinaFunctional, SingleElement) {
  const InType matrix = {{5}};
  const OutType expected = {5};

  ArtyushkinaStringMatrixSEQ task(matrix);
  EXPECT_TRUE(task.Validation());
  EXPECT_TRUE(task.PreProcessing());
  EXPECT_TRUE(task.Run());

  const auto &result = task.GetOutput();
  EXPECT_EQ(result, expected);
}

TEST(ArtyushkinaFunctional, AllSameElements) {
  const InType matrix = {{7, 7, 7}, {7, 7, 7}};
  const OutType expected = {7, 7};

  ArtyushkinaStringMatrixSEQ task(matrix);
  EXPECT_TRUE(task.Validation());
  EXPECT_TRUE(task.PreProcessing());
  EXPECT_TRUE(task.Run());

  const auto &result = task.GetOutput();
  EXPECT_EQ(result, expected);
}

TEST(ArtyushkinaFunctional, NegativeNumbers) {
  const InType matrix = {{-10, -5, -3}, {-2, -8, -1}};
  const OutType expected = {-10, -8};

  ArtyushkinaStringMatrixSEQ task(matrix);
  EXPECT_TRUE(task.Validation());
  EXPECT_TRUE(task.PreProcessing());
  EXPECT_TRUE(task.Run());

  const auto &result = task.GetOutput();
  EXPECT_EQ(result, expected);
}

TEST(ArtyushkinaValidation, EmptyMatrix) {
  const InType matrix = {};
  ArtyushkinaStringMatrixSEQ task(matrix);
  EXPECT_FALSE(task.Validation());
}

TEST(ArtyushkinaValidation, EmptyRow) {
  const InType matrix = {{}, {1, 2}};
  ArtyushkinaStringMatrixSEQ task(matrix);
  EXPECT_FALSE(task.Validation());
}

TEST(ArtyushkinaValidation, DifferentRowSizes) {
  const InType matrix = {{1, 2, 3}, {4, 5}};
  ArtyushkinaStringMatrixSEQ task(matrix);
  EXPECT_FALSE(task.Validation());
}

// ==================== MPI ТЕСТЫ ====================

TEST(ArtyushkinaFunctionalMPI, ConstructorAndValidationNoMPI) {
  const auto [matrix, expected] = GetTestData(1);

  ArtyushkinaStringMatrixMPI task(matrix);
  EXPECT_TRUE(task.Validation());
  EXPECT_TRUE(task.PreProcessing());
}

TEST(ArtyushkinaFunctionalMPI, InvalidMatrixNoMPI) {
  const InType empty_matrix = {};
  ArtyushkinaStringMatrixMPI task(empty_matrix);
  EXPECT_FALSE(task.Validation());
}

// ==================== UNIT ТЕСТЫ ====================

TEST(ArtyushkinaUnitTests, SEQConstructorAndValidation) {
  const InType matrix = {{1, 2, 3}, {4, 5, 6}};
  ArtyushkinaStringMatrixSEQ task(matrix);

  EXPECT_TRUE(task.Validation());
  EXPECT_TRUE(task.PreProcessing());
}

TEST(ArtyushkinaUnitTests, SEQInvalidMatrixEmpty) {
  const InType empty_matrix = {};
  ArtyushkinaStringMatrixSEQ task(empty_matrix);
  EXPECT_FALSE(task.Validation());
}

TEST(ArtyushkinaUnitTests, SEQInvalidMatrixJagged) {
  const InType jagged_matrix = {{1, 2}, {3}};
  ArtyushkinaStringMatrixSEQ task(jagged_matrix);
  EXPECT_FALSE(task.Validation());
}

TEST(ArtyushkinaUnitTests, SEQRunAndGetOutput) {
  const InType matrix = {{1, 2, 3}, {4, 5, 6}};
  ArtyushkinaStringMatrixSEQ task(matrix);

  EXPECT_TRUE(task.Validation());
  EXPECT_TRUE(task.PreProcessing());
  EXPECT_TRUE(task.Run());

  const auto &result = task.GetOutput();
  EXPECT_EQ(result.size(), 2U);
  EXPECT_EQ(result[0], 1);
  EXPECT_EQ(result[1], 4);
}

TEST(ArtyushkinaUnitTests, SEQPostProcessing) {
  const InType matrix = {{1, 2}};
  ArtyushkinaStringMatrixSEQ task(matrix);

  EXPECT_TRUE(task.Validation());
  EXPECT_TRUE(task.PreProcessing());
  EXPECT_TRUE(task.Run());
  EXPECT_TRUE(task.PostProcessing());
}

TEST(ArtyushkinaUnitTests, MPIConstructorAndValidation) {
  const InType matrix = {{1, 2}, {3, 4}};
  ArtyushkinaStringMatrixMPI task(matrix);

  EXPECT_TRUE(task.Validation());
  EXPECT_TRUE(task.PreProcessing());
}

TEST(ArtyushkinaUnitTests, MPIInvalidMatrix) {
  const InType empty_matrix = {};
  ArtyushkinaStringMatrixMPI task(empty_matrix);
  EXPECT_FALSE(task.Validation());
}

TEST(ArtyushkinaUnitTests, MPIFlattenMatrixMethod) {
  const InType matrix = {{1, 2}, {3, 4}};
  const auto flat = ArtyushkinaStringMatrixMPI::FlattenMatrix(matrix);

  EXPECT_EQ(flat.size(), 4U);
  EXPECT_EQ(flat[0], 1);
  EXPECT_EQ(flat[1], 2);
  EXPECT_EQ(flat[2], 3);
  EXPECT_EQ(flat[3], 4);
}

TEST(ArtyushkinaUnitTests, MPIFlattenEmptyMatrix) {
  const InType empty_matrix = {};
  const auto flat = ArtyushkinaStringMatrixMPI::FlattenMatrix(empty_matrix);

  EXPECT_TRUE(flat.empty());
}

TEST(ArtyushkinaUnitTests, MPIPostProcessing) {
  const InType matrix = {{1, 2}};
  ArtyushkinaStringMatrixMPI task(matrix);

  EXPECT_TRUE(task.Validation());
  EXPECT_TRUE(task.PreProcessing());
}
TEST(ArtyushkinaUnitTests, SEQGetStaticTypeOfTask) {
  EXPECT_EQ(ArtyushkinaStringMatrixSEQ::GetStaticTypeOfTask(), ppc::task::TypeOfTask::kSEQ);
}

TEST(ArtyushkinaUnitTests, MPIGetStaticTypeOfTask) {
  EXPECT_EQ(ArtyushkinaStringMatrixMPI::GetStaticTypeOfTask(), ppc::task::TypeOfTask::kMPI);
}

}  // namespace artyushkina_string_matrix
