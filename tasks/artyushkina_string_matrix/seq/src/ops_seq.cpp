#include "artyushkina_string_matrix/seq/include/ops_seq.hpp"

#include <algorithm>
#include <climits>
#include <vector>

namespace artyushkina_string_matrix {

ArtyushkinaATestTaskSEQ::ArtyushkinaATestTaskSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = {};
}

bool ArtyushkinaATestTaskSEQ::ValidationImpl() {
  return !GetInput().empty() && !GetInput()[0].empty();
}

bool ArtyushkinaATestTaskSEQ::PreProcessingImpl() {
  GetOutput() = std::vector<int>();
  return true;
}

bool ArtyushkinaATestTaskSEQ::RunImpl() {
  const auto &matrix = GetInput();
  auto &result = GetOutput();

  for (const auto &row : matrix) {
    if (row.empty()) {
      result.push_back(0);
      continue;
    }
    int min_val = row[0];
    for (size_t i = 1; i < row.size(); ++i) {
      if (row[i] < min_val) {
        min_val = row[i];
      }
    }
    result.push_back(min_val);
  }

  return true;
}

bool ArtyushkinaATestTaskSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace artyushkina_string_matrix
