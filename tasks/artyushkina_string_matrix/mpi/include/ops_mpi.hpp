#pragma once

#include "artyushkina_string_matrix/common/include/common.hpp"
#include "task/include/task.hpp"

namespace artyushkina_string_matrix {

class ArtyushkinaStringMatrixMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit ArtyushkinaStringMatrixMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  static std::vector<int> FlattenMatrix(const std::vector<std::vector<int>> &matrix);
};

}  // namespace artyushkina_string_matrix
