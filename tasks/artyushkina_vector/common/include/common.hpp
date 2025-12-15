#pragma once

#include <tuple>
#include <utility>
#include <vector>

#include "task/include/task.hpp"

namespace artyushkina_vector {

using Matrix = std::vector<std::vector<double>>;
using Vector = std::vector<double>;
using InType = std::pair<Matrix, Vector>;                  // Матрица и вектор
using OutType = Vector;                                    // Результат - вектор
using TestType = std::tuple<int, Matrix, Vector, Vector>;  // ID, матрица, вектор, ожидаемый результат
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace artyushkina_vector
