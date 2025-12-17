#pragma once

#include <vector>

#include "artyushkina_string_matrix/common/include/common.hpp"
#include "task/include/task.hpp"

namespace artyushkina_string_matrix {

class ArtyushkinaStringMatrixMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit ArtyushkinaStringMatrixMPI(const InType &in);

  static std::vector<int> FlattenMatrix(const std::vector<std::vector<int>> &matrix);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  std::pair<int, int> PrepareDimensions(const std::vector<std::vector<int>> &matrix, int rank, int &size);
  std::pair<int, int> CalculateProcessInfo(int total_rows, int size, int rank);
  std::vector<int> ScatterData(const std::vector<std::vector<int>> &matrix, int total_rows, int total_cols, int size,
                               int rank, int my_rows);
  std::vector<int> ComputeLocalMinima(const std::vector<int> &local_data, int my_rows, int total_cols);
  std::vector<int> GatherResults(const std::vector<int> &local_minima, int total_rows, int size, int rank, int my_rows);
};

}  // namespace artyushkina_string_matrix
