#pragma once

#include "artyushkina_bellman_ford_crs/common/include/common.hpp"
#include "task/include/task.hpp"

namespace artyushkina_bellman_ford_crs {

class BellmanFordCRSSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }
  explicit BellmanFordCRSSEQ(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace artyushkina_bellman_ford_crs
