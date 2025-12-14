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

  void DistributeVectorColumns(int world_size, int base, int rem, std::vector<double> &local_vector, int rank,
                               int matrix_cols);
  void ComputeLocalStrip(const std::vector<double> &matrix_flat, const std::vector<double> &local_vector,
                         std::vector<double> &partial_result, int rows, int cols, int local_width, int local_start);
  void CollectResults(int world_size, int rank, int rows, int base, int rem, const std::vector<double> &partial_result,
                      int local_width, int local_start, std::vector<double> &final_result);
};

}  // namespace artyushkina_vector
