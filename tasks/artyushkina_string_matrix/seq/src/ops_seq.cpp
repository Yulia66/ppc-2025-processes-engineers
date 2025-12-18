#include "artyushkina_string_matrix/seq/include/ops_seq.hpp"

#include <algorithm>
#include <climits>
#include <cstddef>
#include <ranges>  // IWYU pragma: keep
#include <vector>

#include "artyushkina_string_matrix/common/include/common.hpp"

namespace artyushkina_string_matrix {

ArtyushkinaStringMatrixSEQ::ArtyushkinaStringMatrixSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());

  if (!in.empty()) {
    GetInput() = in;
  } else {
    GetInput() = InType{};
  }

  GetOutput() = OutType{};
}

bool ArtyushkinaStringMatrixSEQ::ValidationImpl() {
  const auto &input = GetInput();
  if (input.empty()) {
    return false;
  }

  const std::size_t cols = input[0].size();
  if (cols == 0) {
    return false;
  }

  return std::ranges::all_of(input, [cols](const auto &row) { return row.size() == cols; });
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
      min_val = std::min(val, min_val);
    }
    result.push_back(min_val);
  }

  return true;
}

bool ArtyushkinaStringMatrixSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace artyushkina_string_matrix
