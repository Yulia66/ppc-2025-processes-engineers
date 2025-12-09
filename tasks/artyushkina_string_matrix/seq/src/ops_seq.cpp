#include "artyushkina_string_matrix/seq/include/ops_seq.hpp"

#include <algorithm>
#include <climits>
#include <vector>

namespace artyushkina_string_matrix {

ArtyushkinaStringMatrixSEQ::ArtyushkinaStringMatrixSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput().clear();
}

bool ArtyushkinaStringMatrixSEQ::ValidationImpl() {
  const auto& input = GetInput();
  if (input.empty()) {
    return false;
  }

  const size_t cols = input[0].size();
  if (cols == 0) {
    return false;
  }

  for (const auto &row : input) {
    if (row.size() != cols) {
      return false;
    }
  }

  return true;
}

bool ArtyushkinaStringMatrixSEQ::PreProcessingImpl() {
  GetOutput().clear();
  return true;
}

bool ArtyushkinaStringMatrixSEQ::RunImpl() {
  const auto &matrix = GetInput();
  auto &result = GetOutput();

  result.clear();
  result.reserve(matrix.size());

  for (const auto &row : matrix) {
    int min_val = INT_MAX;
    for (const int val : row) {
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