#pragma once

#include "artyushkina_bellman_ford_crs/common/include/common.hpp"
#include "task/include/task.hpp"

namespace artyushkina_bellman_ford_crs {

class BellmanFordCRSMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit BellmanFordCRSMPI(const InType &in);

  ~BellmanFordCRSMPI() override = default;

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace artyushkina_bellman_ford_crs
