#pragma once

#include "artyushkina_vector/common/include/common.hpp"
#include "task/include/task.hpp"

namespace artyushkina_vector {

class VerticalStripMatVecSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }
  explicit VerticalStripMatVecSEQ(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace artyushkina_vector
