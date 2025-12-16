#include "artyushkina_vector/seq/include/ops_seq.hpp"

#include <cstddef>
#include <utility>
#include <vector>

#include "artyushkina_vector/common/include/common.hpp"

#ifdef __GNUC__
#  pragma GCC diagnostic push
#  pragma GCC diagnostic ignored "-Wnull-dereference"
#endif

namespace artyushkina_vector {

VerticalStripMatVecSEQ::VerticalStripMatVecSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = Vector{};
}

bool VerticalStripMatVecSEQ::ValidationImpl() {
  const auto &[matrix, vector] = GetInput();

  if (matrix.empty() || vector.empty()) {
    return false;
  }

  const size_t rows = matrix.size();
  const size_t cols = matrix[0].size();
  const size_t vec_size = vector.size();

  for (size_t i = 1; i < rows; ++i) {
    if (matrix[i].size() != cols) {
      return false;
    }
  }

  return vec_size == cols;
}

bool VerticalStripMatVecSEQ::PreProcessingImpl() {
  GetOutput().clear();
  return true;
}

bool VerticalStripMatVecSEQ::RunImpl() {
  const auto &[matrix, vector] = GetInput();

  if (matrix.empty() || vector.empty()) {
    GetOutput() = Vector{};
    return true;
  }

  const size_t rows = matrix.size();
  const size_t cols = matrix[0].size();

  Vector result(rows, 0.0);

  for (size_t i = 0; i < rows; ++i) {
    double sum = 0.0;
    for (size_t j = 0; j < cols; ++j) {
      sum += matrix[i][j] * vector[j];
    }
    result[i] = sum;
  }

  GetOutput() = result;
  return true;
}

bool VerticalStripMatVecSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace artyushkina_vector

#ifdef __GNUC__
#  pragma GCC diagnostic pop
#endif
