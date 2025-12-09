#include <gtest/gtest.h>

#include <utility>
#include <vector>

#include "artyushkina_string_matrix/common/include/common.hpp"
#include "artyushkina_string_matrix/mpi/include/ops_mpi.hpp"
#include "artyushkina_string_matrix/seq/include/ops_seq.hpp"

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

TEST(ArtyushkinaFunctionalMPI, Test1) {
  const auto [matrix, expected] = GetTestData(1);

  ArtyushkinaStringMatrixMPI task(matrix);
  EXPECT_TRUE(task.Validation());
  EXPECT_TRUE(task.PreProcessing());
  EXPECT_TRUE(task.Run());

  const auto &result = task.GetOutput();
  if (!result.empty()) {
    EXPECT_EQ(result, expected);
  }
}

TEST(ArtyushkinaFunctionalMPI, Test2) {
  const auto [matrix, expected] = GetTestData(2);

  ArtyushkinaStringMatrixMPI task(matrix);
  EXPECT_TRUE(task.Validation());
  EXPECT_TRUE(task.PreProcessing());
  EXPECT_TRUE(task.Run());

  const auto &result = task.GetOutput();
  if (!result.empty()) {
    EXPECT_EQ(result, expected);
  }
}

TEST(ArtyushkinaFunctionalMPI, LargeMatrix) {
  const InType matrix = {{1, 2, 3, 4, 5}, {6, 7, 8, 9, 10}, {11, 12, 13, 14, 15}};
  const OutType expected = {1, 6, 11};

  ArtyushkinaStringMatrixMPI task(matrix);
  EXPECT_TRUE(task.Validation());
  EXPECT_TRUE(task.PreProcessing());
  EXPECT_TRUE(task.Run());

  const auto &result = task.GetOutput();
  if (!result.empty()) {
    EXPECT_EQ(result, expected);
  }
}

}  // namespace artyushkina_string_matrix
