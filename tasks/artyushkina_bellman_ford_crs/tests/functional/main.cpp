#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <string>
#include <tuple>
#include <vector>

#include "artyushkina_bellman_ford_crs/common/include/common.hpp"
#include "artyushkina_bellman_ford_crs/mpi/include/ops_mpi.hpp"
#include "artyushkina_bellman_ford_crs/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace artyushkina_bellman_ford_crs {

class BellmanFordCRSFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    int test_id = std::get<0>(test_param);
    return std::to_string(test_id);
  }

 protected:
  void SetUp() override {
    const auto &params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    input_data_ = std::get<1>(params);
    expected_ = std::get<2>(params);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    if (output_data.size() != expected_.size()) {
      return false;
    }

    for (size_t i = 0; i < output_data.size(); ++i) {
      if (std::isnan(output_data[i]) || std::isnan(expected_[i])) {
        return false;
      }

      bool output_is_inf = std::isinf(output_data[i]);
      bool expected_is_inf = std::isinf(expected_[i]);

      if (output_is_inf && expected_is_inf) {
        continue;
      } else if (output_is_inf != expected_is_inf) {
        return false;
      }

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

CRSGraph CreateSimpleGraph() {
  CRSGraph graph;
  graph.num_vertices = 4;
  graph.num_edges = 5;
  graph.source_vertex = 0;
  graph.row_ptr = {0, 2, 4, 5, 5};
  graph.col_idx = {1, 2, 2, 3, 3};
  graph.values = {1.0, 4.0, 2.0, 6.0, 3.0};
  return graph;
}

CRSGraph CreateGraphWithNegativeWeights() {
  CRSGraph graph;
  graph.num_vertices = 3;
  graph.num_edges = 3;
  graph.source_vertex = 0;
  graph.row_ptr = {0, 2, 3, 3};
  graph.col_idx = {1, 2, 2};
  graph.values = {-1.0, 4.0, 3.0};
  return graph;
}

CRSGraph CreateDisconnectedGraph() {
  CRSGraph graph;
  graph.num_vertices = 4;
  graph.num_edges = 2;
  graph.source_vertex = 0;
  graph.row_ptr = {0, 1, 1, 2, 2};
  graph.col_idx = {1, 3};
  graph.values = {1.0, 1.0};
  return graph;
}

CRSGraph CreateSingleVertexGraph() {
  CRSGraph graph;
  graph.num_vertices = 1;
  graph.num_edges = 0;
  graph.source_vertex = 0;
  graph.row_ptr = {0, 0};
  graph.col_idx = {};
  graph.values = {};
  return graph;
}

CRSGraph CreateEmptyGraph() {
  CRSGraph graph;
  graph.num_vertices = 0;
  graph.num_edges = 0;
  graph.source_vertex = 0;
  graph.row_ptr = {0};
  graph.col_idx = {};
  graph.values = {};
  return graph;
}

const std::array<TestType, 5> kTestParam = {
    {TestType{1, CreateSimpleGraph(), std::vector<double>{0.0, 1.0, 3.0, 6.0}},
     TestType{2, CreateGraphWithNegativeWeights(), std::vector<double>{0.0, -1.0, 2.0}},
     TestType{3, CreateDisconnectedGraph(),
              std::vector<double>{0.0, 1.0, std::numeric_limits<double>::infinity(),
                                  std::numeric_limits<double>::infinity()}},
     TestType{4, CreateSingleVertexGraph(), std::vector<double>{0.0}},
     TestType{5, CreateEmptyGraph(), std::vector<double>{}}}};

TEST_P(BellmanFordCRSFuncTests, BellmanFordAlgorithm) {
  ExecuteTest(GetParam());
}

// Создаем тип для фабрики задач
using TaskFactoryType = std::function<std::shared_ptr<ppc::task::Task<InType, OutType>>(InType)>;

// Создаем список тестовых данных
const auto kTestData = std::make_tuple(
    // SEQ тесты
    std::make_tuple(
        [](const InType &in) -> std::shared_ptr<ppc::task::Task<InType, OutType>> {
  return std::make_shared<BellmanFordCRSSEQ>(in);
}, "seq",
        std::array<TestType, 5>{{TestType{1, CreateSimpleGraph(), std::vector<double>{0.0, 1.0, 3.0, 6.0}},
                                 TestType{2, CreateGraphWithNegativeWeights(), std::vector<double>{0.0, -1.0, 2.0}},
                                 TestType{3, CreateDisconnectedGraph(),
                                          std::vector<double>{0.0, 1.0, std::numeric_limits<double>::infinity(),
                                                              std::numeric_limits<double>::infinity()}},
                                 TestType{4, CreateSingleVertexGraph(), std::vector<double>{0.0}},
                                 TestType{5, CreateEmptyGraph(), std::vector<double>{}}}})
    // Можно добавить MPI тесты здесь, если нужно
);

// Альтернативный подход - простые тесты без сложных шаблонов
class BellmanFordSEQTest : public ::testing::TestWithParam<TestType> {
 protected:
  void SetUp() override {
    test_param_ = GetParam();
    input_data_ = std::get<1>(test_param_);
    expected_ = std::get<2>(test_param_);
  }

  bool CheckOutput(const OutType &output) {
    if (output.size() != expected_.size()) {
      return false;
    }

    for (size_t i = 0; i < output.size(); ++i) {
      if (std::isnan(output[i]) || std::isnan(expected_[i])) {
        return false;
      }

      bool output_is_inf = std::isinf(output[i]);
      bool expected_is_inf = std::isinf(expected_[i]);

      if (output_is_inf && expected_is_inf) {
        continue;
      } else if (output_is_inf != expected_is_inf) {
        return false;
      }

      if (std::abs(output[i] - expected_[i]) > 1e-9) {
        return false;
      }
    }

    return true;
  }

  TestType test_param_;
  InType input_data_;
  OutType expected_;
};

TEST_P(BellmanFordSEQTest, SimpleBellmanFord) {
  BellmanFordCRSSEQ algorithm(input_data_);

  // Проверяем валидацию
  EXPECT_TRUE(algorithm.Validation());

  // Запускаем алгоритм
  algorithm.Run();

  // Получаем результат
  auto output = algorithm.GetOutput();

  // Проверяем результат
  EXPECT_TRUE(CheckOutput(output));
}

// Определяем тестовые случаи для SEQ
const std::array<TestType, 5> kSEQTestCases = {
    {TestType{1, CreateSimpleGraph(), std::vector<double>{0.0, 1.0, 3.0, 6.0}},
     TestType{2, CreateGraphWithNegativeWeights(), std::vector<double>{0.0, -1.0, 2.0}},
     TestType{3, CreateDisconnectedGraph(),
              std::vector<double>{0.0, 1.0, std::numeric_limits<double>::infinity(),
                                  std::numeric_limits<double>::infinity()}},
     TestType{4, CreateSingleVertexGraph(), std::vector<double>{0.0}},
     TestType{5, CreateEmptyGraph(), std::vector<double>{}}}};

// Функция для генерации имен тестов
std::string TestNamingFunctionSEQ(const testing::TestParamInfo<TestType> &info) {
  int test_id = std::get<0>(info.param);
  return "SEQ_Test_" + std::to_string(test_id);
}

INSTANTIATE_TEST_SUITE_P(SEQTests, BellmanFordSEQTest, testing::ValuesIn(kSEQTestCases), TestNamingFunctionSEQ);

}  // namespace

}  // namespace artyushkina_bellman_ford_crs
