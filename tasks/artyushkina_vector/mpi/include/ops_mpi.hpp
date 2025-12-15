#pragma once

#include "artyushkina_vector/common/include/common.hpp"
#include "task/include/task.hpp"

namespace artyushkina_vector {

class VerticalStripMatVecMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit VerticalStripMatVecMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace artyushkina_vector
