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

  ~BellmanFordCRSMPI();

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  // Удаляем неиспользуемое поле или помечаем как [[maybe_unused]]
  // bool mpi_initialized_by_me{false}; // Удалить эту строку
};

}  // namespace artyushkina_bellman_ford_crs
