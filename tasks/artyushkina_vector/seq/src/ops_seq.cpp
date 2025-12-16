#include "artyushkina_vector/seq/include/ops_seq.hpp"

#include <cstddef>
#include <utility>
#include <vector>

#ifdef __GNUC__
#  pragma GCC diagnostic push
#  pragma GCC diagnostic ignored "-Wnull-dereference"
#endif

namespace artyushkina_vector {

VerticalStripMatVecSEQ::VerticalStripMatVecSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());

  Matrix matrix_copy = in.first;
  Vector vector_copy = in.second;

  GetInput().first = std::move(matrix_copy);
  GetInput().second = std::move(vector_copy);

  GetOutput() = Vector{};
}

bool VerticalStripMatVecSEQ::ValidationImpl() {
  const auto &[matrix, vector] = GetInput();

  if (matrix.empty() || vector.empty()) {
    return false;
  }

  size_t rows = matrix.size();
  size_t cols = matrix[0].size();
  size_t vec_size = vector.size();

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

  size_t rows = matrix.size();
  size_t cols = matrix[0].size();

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
