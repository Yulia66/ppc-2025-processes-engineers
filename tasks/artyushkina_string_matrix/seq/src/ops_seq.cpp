#include "artyushkina_string_matrix/seq/include/ops_seq.hpp"

#include <algorithm>
#include <climits>
#include <vector>

namespace artyushkina_string_matrix {

ArtyushkinaStringMatrixSEQ::ArtyushkinaStringMatrixSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = std::vector<std::vector<int>>(in);  // Явное копирование
  GetOutput() = std::vector<int>();
}

bool ArtyushkinaStringMatrixSEQ::ValidationImpl() {
  if (GetInput().empty()) {
    return false;
  }

  size_t cols = GetInput()[0].size();
  if (cols == 0) {
    return false;
  }

  for (const auto &row : GetInput()) {
    if (row.size() != cols) {
      return false;
    }
  }

  return true;
}

bool ArtyushkinaStringMatrixSEQ::PreProcessingImpl() {
  GetOutput() = std::vector<int>();
  return true;
}

bool ArtyushkinaStringMatrixSEQ::RunImpl() {
  const auto &matrix = GetInput();
  auto &result = GetOutput();

  result.reserve(matrix.size());

  for (const auto &row : matrix) {
    int min_val = INT_MAX;
    for (int val : row) {
      if (val < min_val) {
        min_val = val;
      }
    }
    result.push_back(min_val);
  }

  return true;
}

bool ArtyushkinaStringMatrixSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace artyushkina_string_matrix
