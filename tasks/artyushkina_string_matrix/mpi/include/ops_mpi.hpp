#pragma once

#include "artyushkina_string_matrix/common/include/common.hpp"
#include "task/include/task.hpp"

namespace artyushkina_string_matrix {

class ArtyushkinaATestTaskMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit ArtyushkinaATestTaskMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace artyushkina_string_matrix
