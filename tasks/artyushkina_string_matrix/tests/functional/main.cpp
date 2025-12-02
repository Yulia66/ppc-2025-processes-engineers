#include <gtest/gtest.h>
#include <vector>
#include <string>
#include "artyushkina_string_matrix/common/include/common.hpp"
#include "artyushkina_string_matrix/mpi/include/ops_mpi.hpp"
#include "artyushkina_string_matrix/seq/include/ops_seq.hpp"

namespace artyushkina_string_matrix {


TEST(ArtyushkinaStringMatrixFunctional, Basic2x2Matrix) {
    InType matrix = {{1, 2}, {3, 4}};
    ArtyushkinaATestTaskSEQ task(matrix);
    
    EXPECT_TRUE(task.Validation());
    EXPECT_TRUE(task.PreProcessing());
    EXPECT_TRUE(task.Run());
    EXPECT_TRUE(task.PostProcessing());
    
    auto result = task.GetOutput();
    EXPECT_FALSE(result.empty());
    EXPECT_EQ(result.size(), 2); 
}

TEST(ArtyushkinaStringMatrixFunctional, Binary2x3Matrix) {
    InType matrix = {{1, 0, 1}, {0, 1, 0}};
    ArtyushkinaATestTaskSEQ task(matrix);
    
    EXPECT_TRUE(task.Validation());
    EXPECT_TRUE(task.PreProcessing());
    EXPECT_TRUE(task.Run());
    EXPECT_TRUE(task.PostProcessing());
    
    auto result = task.GetOutput();
    EXPECT_FALSE(result.empty());
    EXPECT_EQ(result.size(), 2);
}

TEST(ArtyushkinaStringMatrixFunctional, SingleElement) {
    InType matrix = {{5}};
    ArtyushkinaATestTaskSEQ task(matrix);
    
    EXPECT_TRUE(task.Validation());
    EXPECT_TRUE(task.PreProcessing());
    EXPECT_TRUE(task.Run());
    EXPECT_TRUE(task.PostProcessing());
    
    auto result = task.GetOutput();
    EXPECT_FALSE(result.empty());
    EXPECT_EQ(result.size(), 1);
}


//#ifdef BUILD_MPI_TESTS
TEST(ArtyushkinaStringMatrixFunctional, MPI_Basic2x2Matrix) {
    InType matrix = {{1, 2}, {3, 4}};
    ArtyushkinaATestTaskMPI task(matrix);
    
    EXPECT_TRUE(task.Validation());
    EXPECT_TRUE(task.PreProcessing());
    EXPECT_TRUE(task.Run());
    EXPECT_TRUE(task.PostProcessing());
    
    auto result = task.GetOutput();
    EXPECT_FALSE(result.empty());
    EXPECT_EQ(result.size(), 2);
}
//#endif

} // namespace artyushkina_string_matrix