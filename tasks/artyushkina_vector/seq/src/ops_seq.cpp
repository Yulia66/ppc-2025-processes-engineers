#include "artyushkina_vector/seq/include/ops_seq.hpp"

#include <cstddef>
#include <vector>

namespace artyushkina_vector {

VerticalStripMatVecSEQ::VerticalStripMatVecSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());

  // Полностью избегаем декомпозиции
  std::pair<Matrix, Vector> &input = GetInput();
  input.first = in.first;
  input.second = in.second;

  GetOutput() = Vector{};  // Явно создаем пустой вектор
}

bool VerticalStripMatVecSEQ::ValidationImpl() {
  const auto &[matrix, vector] = GetInput();

  if (matrix.empty() || vector.empty()) {
    return false;
  }

  size_t rows = matrix.size();
  size_t cols = matrix[0].size();
  size_t vec_size = vector.size();

  // Проверка прямоугольности матрицы
  for (size_t i = 1; i < rows; ++i) {
    if (matrix[i].size() != cols) {
      return false;
    }
  }

  // Размер вектора должен совпадать с количеством столбцов матрицы
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

  // Классическое умножение матрицы на вектор
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
