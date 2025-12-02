#include "artyushkina_string_matrix/mpi/include/ops_mpi.hpp"
#include "C:/Program Files (x86)/Microsoft SDKs/MPI/Include/mpi.h"
#include <algorithm>
#include <climits>

namespace artyushkina_string_matrix {

ArtyushkinaATestTaskMPI::ArtyushkinaATestTaskMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = {};
}

bool ArtyushkinaATestTaskMPI::ValidationImpl() {
    return !GetInput().empty() && !GetInput()[0].empty();
}

bool ArtyushkinaATestTaskMPI::PreProcessingImpl() {
    GetOutput() = std::vector<int>();
    return true;
}

bool ArtyushkinaATestTaskMPI::RunImpl() {
    const auto& matrix = GetInput();
    auto& result = GetOutput();
    
    for (const auto& row : matrix) {
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

bool ArtyushkinaATestTaskMPI::PostProcessingImpl() {
    return true;
}

}  // namespace artyushkina_string_matrix