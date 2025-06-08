#include <gtest/gtest.h>
#include "matrix.hpp"

using namespace matrix;

// Test basic matrix multiplication with simple 2x2 matrices
TEST(MatrixTest, BasicMultiplication2x2) {
    // Create matrices A and B
    Matrix A(2, 2);
    Matrix B(2, 2);
    Matrix result(2, 2);

    // Set up A = [[1, 2], [3, 4]]
    A(0, 0) = 1.0f; A(0, 1) = 2.0f;
    A(1, 0) = 3.0f; A(1, 1) = 4.0f;

    // Set up B = [[5, 6], [7, 8]]
    B(0, 0) = 5.0f; B(0, 1) = 6.0f;
    B(1, 0) = 7.0f; B(1, 1) = 8.0f;

    // Perform multiplication
    A.multiply_into(B, result);

    // Expected result: [[19, 22], [43, 50]]
    EXPECT_FLOAT_EQ(result(0, 0), 19.0f);
    EXPECT_FLOAT_EQ(result(0, 1), 22.0f);
    EXPECT_FLOAT_EQ(result(1, 0), 43.0f);
    EXPECT_FLOAT_EQ(result(1, 1), 50.0f);
}

// Test identity matrix multiplication
TEST(MatrixTest, IdentityMultiplication) {
    // Create a 3x3 matrix and identity matrix
    Matrix A(3, 3);
    Matrix I(3, 3);
    Matrix result(3, 3);

    // Set up A with some values
    A(0, 0) = 1.0f; A(0, 1) = 2.0f; A(0, 2) = 3.0f;
    A(1, 0) = 4.0f; A(1, 1) = 5.0f; A(1, 2) = 6.0f;
    A(2, 0) = 7.0f; A(2, 1) = 8.0f; A(2, 2) = 9.0f;

    // Set up identity matrix
    I(0, 0) = 1.0f; I(0, 1) = 0.0f; I(0, 2) = 0.0f;
    I(1, 0) = 0.0f; I(1, 1) = 1.0f; I(1, 2) = 0.0f;
    I(2, 0) = 0.0f; I(2, 1) = 0.0f; I(2, 2) = 1.0f;

    // A * I should equal A
    A.multiply_into(I, result);

    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            EXPECT_FLOAT_EQ(result(i, j), A(i, j));
        }
    }
}

// Test multiplication with non-square matrices
TEST(MatrixTest, NonSquareMultiplication) {
    // 2x3 * 3x2 = 2x2
    Matrix A(2, 3);
    Matrix B(3, 2);
    Matrix result(2, 2);

    // Set up A = [[1, 2, 3], [4, 5, 6]]
    A(0, 0) = 1.0f; A(0, 1) = 2.0f; A(0, 2) = 3.0f;
    A(1, 0) = 4.0f; A(1, 1) = 5.0f; A(1, 2) = 6.0f;

    // Set up B = [[7, 8], [9, 10], [11, 12]]
    B(0, 0) = 7.0f;  B(0, 1) = 8.0f;
    B(1, 0) = 9.0f;  B(1, 1) = 10.0f;
    B(2, 0) = 11.0f; B(2, 1) = 12.0f;

    // Perform multiplication
    A.multiply_into(B, result);

    // Expected result: [[58, 64], [139, 154]]
    EXPECT_FLOAT_EQ(result(0, 0), 58.0f);
    EXPECT_FLOAT_EQ(result(0, 1), 64.0f);
    EXPECT_FLOAT_EQ(result(1, 0), 139.0f);
    EXPECT_FLOAT_EQ(result(1, 1), 154.0f);
}

// Test zero matrix multiplication
TEST(MatrixTest, ZeroMatrixMultiplication) {
    Matrix A(2, 2, 5.0f);  // Fill with 5.0f
    Matrix B(2, 2, 0.0f);  // Fill with 0.0f (zero matrix)
    Matrix result(2, 2);

    // A * 0 should equal 0
    A.multiply_into(B, result);

    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 2; ++j) {
            EXPECT_FLOAT_EQ(result(i, j), 0.0f);
        }
    }
}

// Test dimension mismatch errors
TEST(MatrixTest, DimensionMismatchError) {
    Matrix A(2, 3);  // 2x3
    Matrix B(2, 2);  // 2x2 (incompatible: A.M != B.N)
    Matrix result(2, 2);

    // Should throw runtime_error for incompatible dimensions
    EXPECT_THROW(A.multiply_into(B, result), std::runtime_error);
}

// Test incorrect result matrix dimensions
TEST(MatrixTest, IncorrectResultDimensionsError) {
    Matrix A(2, 3);  // 2x3
    Matrix B(3, 2);  // 3x2
    Matrix result(3, 3);  // Wrong size! Should be 2x2

    // Should throw runtime_error for incorrect result dimensions
    EXPECT_THROW(A.multiply_into(B, result), std::runtime_error);
}

// Test single element matrices
TEST(MatrixTest, SingleElementMultiplication) {
    Matrix A(1, 1, 3.0f);
    Matrix B(1, 1, 4.0f);
    Matrix result(1, 1);

    A.multiply_into(B, result);

    EXPECT_FLOAT_EQ(result(0, 0), 12.0f);
}

// Test matrix constructor with lambda function
TEST(MatrixTest, LambdaConstructorMultiplication) {
    // Create a matrix using lambda where A[i][j] = i + j
    auto lambda_a = [](size_t i, size_t j) -> float {
        return static_cast<float>(i + j);
    };

    // Create a matrix using lambda where B[i][j] = i * j + 1
    auto lambda_b = [](size_t i, size_t j) -> float {
        return static_cast<float>(i * j + 1);
    };

    Matrix A(2, 2, lambda_a);
    Matrix B(2, 2, lambda_b);
    Matrix result(2, 2);

    // A = [[0, 1], [1, 2]]
    // B = [[1, 1], [1, 2]]
    A.multiply_into(B, result);

    // Expected result: [[1, 2], [3, 5]]
    EXPECT_FLOAT_EQ(result(0, 0), 1.0f);
    EXPECT_FLOAT_EQ(result(0, 1), 2.0f);
    EXPECT_FLOAT_EQ(result(1, 0), 3.0f);
    EXPECT_FLOAT_EQ(result(1, 1), 5.0f);
}

// Test operator() bounds checking
TEST(MatrixTest, IndexOperatorBoundsCheck) {
    Matrix A(2, 2);

    // Valid access should work
    EXPECT_NO_THROW(A(0, 0) = 1.0f);
    EXPECT_NO_THROW(A(1, 1) = 2.0f);

    // Out of bounds access should throw
    EXPECT_THROW(A(2, 0), std::out_of_range);
    EXPECT_THROW(A(0, 2), std::out_of_range);
    EXPECT_THROW(A(2, 2), std::out_of_range);
}

// Tests for multiply_transpose_into function

// Test basic transpose multiplication with simple 2x2 matrices
TEST(MatrixTest, BasicTransposeMultiplication2x2) {
    // Create matrices A and B
    Matrix A(2, 2);
    Matrix B(2, 2);
    Matrix result(2, 2);

    // Set up A = [[1, 2], [3, 4]]
    A(0, 0) = 1.0f; A(0, 1) = 2.0f;
    A(1, 0) = 3.0f; A(1, 1) = 4.0f;

    // Set up B = [[5, 6], [7, 8]]
    B(0, 0) = 5.0f; B(0, 1) = 6.0f;
    B(1, 0) = 7.0f; B(1, 1) = 8.0f;

    // Perform A * B^T where B^T = [[5, 7], [6, 8]]
    A.multiply_transpose_into(B, result);

    // Expected result: A * B^T = [[1*5+2*6, 1*7+2*8], [3*5+4*6, 3*7+4*8]]
    //                           = [[17, 23], [39, 53]]
    EXPECT_FLOAT_EQ(result(0, 0), 17.0f);
    EXPECT_FLOAT_EQ(result(0, 1), 23.0f);
    EXPECT_FLOAT_EQ(result(1, 0), 39.0f);
    EXPECT_FLOAT_EQ(result(1, 1), 53.0f);
}

// Test transpose multiplication with non-square matrices
TEST(MatrixTest, NonSquareTransposeMultiplication) {
    // 2x3 * (2x3)^T = 2x3 * 3x2 = 2x2
    Matrix A(2, 3);
    Matrix B(2, 3);
    Matrix result(2, 2);

    // Set up A = [[1, 2, 3], [4, 5, 6]]
    A(0, 0) = 1.0f; A(0, 1) = 2.0f; A(0, 2) = 3.0f;
    A(1, 0) = 4.0f; A(1, 1) = 5.0f; A(1, 2) = 6.0f;

    // Set up B = [[7, 8, 9], [10, 11, 12]]
    B(0, 0) = 7.0f;  B(0, 1) = 8.0f;  B(0, 2) = 9.0f;
    B(1, 0) = 10.0f; B(1, 1) = 11.0f; B(1, 2) = 12.0f;

    // B^T = [[7, 10], [8, 11], [9, 12]]
    // Perform multiplication A * B^T
    A.multiply_transpose_into(B, result);

    // Expected result: [[1*7+2*8+3*9, 1*10+2*11+3*12], [4*7+5*8+6*9, 4*10+5*11+6*12]]
    //                  = [[50, 68], [122, 167]]
    EXPECT_FLOAT_EQ(result(0, 0), 50.0f);
    EXPECT_FLOAT_EQ(result(0, 1), 68.0f);
    EXPECT_FLOAT_EQ(result(1, 0), 122.0f);
    EXPECT_FLOAT_EQ(result(1, 1), 167.0f);
}

// Test transpose multiplication with identity matrix
TEST(MatrixTest, IdentityTransposeMultiplication) {
    // Create a 2x2 matrix and identity matrix
    Matrix A(2, 2);
    Matrix I(2, 2);
    Matrix result(2, 2);

    // Set up A with some values
    A(0, 0) = 3.0f; A(0, 1) = 4.0f;
    A(1, 0) = 5.0f; A(1, 1) = 6.0f;

    // Set up identity matrix
    I(0, 0) = 1.0f; I(0, 1) = 0.0f;
    I(1, 0) = 0.0f; I(1, 1) = 1.0f;

    // A * I^T should equal A (since I^T = I for identity matrix)
    A.multiply_transpose_into(I, result);

    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 2; ++j) {
            EXPECT_FLOAT_EQ(result(i, j), A(i, j));
        }
    }
}

// Test zero matrix transpose multiplication
TEST(MatrixTest, ZeroMatrixTransposeMultiplication) {
    Matrix A(2, 2, 5.0f);  // Fill with 5.0f
    Matrix B(2, 2, 0.0f);  // Fill with 0.0f (zero matrix)
    Matrix result(2, 2);

    // A * 0^T should equal 0
    A.multiply_transpose_into(B, result);

    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 2; ++j) {
            EXPECT_FLOAT_EQ(result(i, j), 0.0f);
        }
    }
}

// Test dimension mismatch errors for transpose multiplication
TEST(MatrixTest, TransposeDimensionMismatchError) {
    Matrix A(2, 3);  // 2x3
    Matrix B(2, 2);  // 2x2, B^T would be 2x2, but A.M (3) != B.M (2)
    Matrix result(2, 2);

    // Should throw runtime_error for incompatible dimensions
    EXPECT_THROW(A.multiply_transpose_into(B, result), std::runtime_error);
}

// Test incorrect result matrix dimensions for transpose multiplication
TEST(MatrixTest, TransposeIncorrectResultDimensionsError) {
    Matrix A(2, 3);  // 2x3
    Matrix B(2, 3);  // 2x3, B^T is 3x2, so result should be 2x2
    Matrix result(3, 3);  // Wrong size! Should be 2x2

    // Should throw runtime_error for incorrect result dimensions
    EXPECT_THROW(A.multiply_transpose_into(B, result), std::runtime_error);
}

// Test single element transpose multiplication
TEST(MatrixTest, SingleElementTransposeMultiplication) {
    Matrix A(1, 1, 3.0f);
    Matrix B(1, 1, 4.0f);
    Matrix result(1, 1);

    A.multiply_transpose_into(B, result);

    // 3 * 4^T = 3 * 4 = 12
    EXPECT_FLOAT_EQ(result(0, 0), 12.0f);
}

// Test rectangular matrices with different dimensions
TEST(MatrixTest, RectangularTransposeMultiplication) {
    // 3x2 * (4x2)^T = 3x2 * 2x4 = 3x4
    Matrix A(3, 2);
    Matrix B(4, 2);
    Matrix result(3, 4);

    // Set up A = [[1, 2], [3, 4], [5, 6]]
    A(0, 0) = 1.0f; A(0, 1) = 2.0f;
    A(1, 0) = 3.0f; A(1, 1) = 4.0f;
    A(2, 0) = 5.0f; A(2, 1) = 6.0f;

    // Set up B = [[7, 8], [9, 10], [11, 12], [13, 14]]
    B(0, 0) = 7.0f;  B(0, 1) = 8.0f;
    B(1, 0) = 9.0f;  B(1, 1) = 10.0f;
    B(2, 0) = 11.0f; B(2, 1) = 12.0f;
    B(3, 0) = 13.0f; B(3, 1) = 14.0f;

    // B^T = [[7, 9, 11, 13], [8, 10, 12, 14]]
    A.multiply_transpose_into(B, result);

    // Verify a few key elements
    // result(0,0) = 1*7 + 2*8 = 23
    // result(0,1) = 1*9 + 2*10 = 29
    // result(1,0) = 3*7 + 4*8 = 53
    // result(2,3) = 5*13 + 6*14 = 149
    EXPECT_FLOAT_EQ(result(0, 0), 23.0f);
    EXPECT_FLOAT_EQ(result(0, 1), 29.0f);
    EXPECT_FLOAT_EQ(result(1, 0), 53.0f);
    EXPECT_FLOAT_EQ(result(2, 3), 149.0f);
}

// Tests for transpose_multiply_into function (A^T * B)

// Test basic transpose multiplication with simple 2x2 matrices
TEST(MatrixTest, BasicTransposeMultiplyInto2x2) {
    // Create matrices A and B
    Matrix A(2, 2);
    Matrix B(2, 2);
    Matrix result(2, 2);

    // Set up A = [[1, 2], [3, 4]]
    A(0, 0) = 1.0f; A(0, 1) = 2.0f;
    A(1, 0) = 3.0f; A(1, 1) = 4.0f;

    // Set up B = [[5, 6], [7, 8]]
    B(0, 0) = 5.0f; B(0, 1) = 6.0f;
    B(1, 0) = 7.0f; B(1, 1) = 8.0f;

    // Perform A^T * B where A^T = [[1, 3], [2, 4]]
    A.transpose_multiply_into(B, result);

    // Expected result: A^T * B = [[1*5+3*7, 1*6+3*8], [2*5+4*7, 2*6+4*8]]
    //                            = [[26, 30], [38, 44]]
    EXPECT_FLOAT_EQ(result(0, 0), 26.0f);
    EXPECT_FLOAT_EQ(result(0, 1), 30.0f);
    EXPECT_FLOAT_EQ(result(1, 0), 38.0f);
    EXPECT_FLOAT_EQ(result(1, 1), 44.0f);
}

// Test transpose multiplication with non-square matrices
TEST(MatrixTest, NonSquareTransposeMultiplyInto) {
    // (3x2)^T * (3x2) = 2x3 * 3x2 = 2x2
    Matrix A(3, 2);
    Matrix B(3, 2);
    Matrix result(2, 2);

    // Set up A = [[1, 2], [3, 4], [5, 6]]
    A(0, 0) = 1.0f; A(0, 1) = 2.0f;
    A(1, 0) = 3.0f; A(1, 1) = 4.0f;
    A(2, 0) = 5.0f; A(2, 1) = 6.0f;

    // Set up B = [[7, 8], [9, 10], [11, 12]]
    B(0, 0) = 7.0f;  B(0, 1) = 8.0f;
    B(1, 0) = 9.0f;  B(1, 1) = 10.0f;
    B(2, 0) = 11.0f; B(2, 1) = 12.0f;

    // A^T = [[1, 3, 5], [2, 4, 6]]
    // Perform multiplication A^T * B
    A.transpose_multiply_into(B, result);

    // Expected result: [[1*7+3*9+5*11, 1*8+3*10+5*12], [2*7+4*9+6*11, 2*8+4*10+6*12]]
    //                  = [[89, 98], [116, 128]]
    EXPECT_FLOAT_EQ(result(0, 0), 89.0f);
    EXPECT_FLOAT_EQ(result(0, 1), 98.0f);
    EXPECT_FLOAT_EQ(result(1, 0), 116.0f);
    EXPECT_FLOAT_EQ(result(1, 1), 128.0f);
}

// Test transpose multiplication with identity matrix
TEST(MatrixTest, IdentityTransposeMultiplyInto) {
    // Create a 2x2 matrix and identity matrix
    Matrix A(2, 2);
    Matrix I(2, 2);
    Matrix result(2, 2);

    // Set up A with some values
    A(0, 0) = 3.0f; A(0, 1) = 4.0f;
    A(1, 0) = 5.0f; A(1, 1) = 6.0f;

    // Set up identity matrix
    I(0, 0) = 1.0f; I(0, 1) = 0.0f;
    I(1, 0) = 0.0f; I(1, 1) = 1.0f;

    // A^T * I should equal A^T
    A.transpose_multiply_into(I, result);

    // A^T = [[3, 5], [4, 6]]
    EXPECT_FLOAT_EQ(result(0, 0), 3.0f);
    EXPECT_FLOAT_EQ(result(0, 1), 5.0f);
    EXPECT_FLOAT_EQ(result(1, 0), 4.0f);
    EXPECT_FLOAT_EQ(result(1, 1), 6.0f);
}

// Test zero matrix transpose multiplication
TEST(MatrixTest, ZeroMatrixTransposeMultiplyInto) {
    Matrix A(2, 2, 5.0f);  // Fill with 5.0f
    Matrix B(2, 2, 0.0f);  // Fill with 0.0f (zero matrix)
    Matrix result(2, 2);

    // A^T * 0 should equal 0
    A.transpose_multiply_into(B, result);

    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 2; ++j) {
            EXPECT_FLOAT_EQ(result(i, j), 0.0f);
        }
    }
}

// Test dimension mismatch errors for transpose multiplication
TEST(MatrixTest, TransposeMultiplyIntoDimensionMismatchError) {
    Matrix A(2, 3);  // 2x3, A^T would be 3x2
    Matrix B(3, 2);  // 3x2, but A.N (2) != B.N (3)
    Matrix result(3, 2);

    // Should throw runtime_error for incompatible dimensions
    EXPECT_THROW(A.transpose_multiply_into(B, result), std::runtime_error);
}

// Test incorrect result matrix dimensions for transpose multiplication
TEST(MatrixTest, TransposeMultiplyIntoIncorrectResultDimensionsError) {
    Matrix A(3, 2);  // 3x2, A^T is 2x3
    Matrix B(3, 2);  // 3x2, so result should be 2x2
    Matrix result(3, 3);  // Wrong size! Should be 2x2

    // Should throw runtime_error for incorrect result dimensions
    EXPECT_THROW(A.transpose_multiply_into(B, result), std::runtime_error);
}

// Test single element transpose multiplication
TEST(MatrixTest, SingleElementTransposeMultiplyInto) {
    Matrix A(1, 1, 3.0f);
    Matrix B(1, 1, 4.0f);
    Matrix result(1, 1);

    A.transpose_multiply_into(B, result);

    // 3^T * 4 = 3 * 4 = 12
    EXPECT_FLOAT_EQ(result(0, 0), 12.0f);
}

// Test rectangular matrices with different dimensions
TEST(MatrixTest, RectangularTransposeMultiplyInto) {
    // (4x3)^T * (4x2) = 3x4 * 4x2 = 3x2
    Matrix A(4, 3);
    Matrix B(4, 2);
    Matrix result(3, 2);

    // Set up A = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]
    A(0, 0) = 1.0f;  A(0, 1) = 2.0f;  A(0, 2) = 3.0f;
    A(1, 0) = 4.0f;  A(1, 1) = 5.0f;  A(1, 2) = 6.0f;
    A(2, 0) = 7.0f;  A(2, 1) = 8.0f;  A(2, 2) = 9.0f;
    A(3, 0) = 10.0f; A(3, 1) = 11.0f; A(3, 2) = 12.0f;

    // Set up B = [[13, 14], [15, 16], [17, 18], [19, 20]]
    B(0, 0) = 13.0f; B(0, 1) = 14.0f;
    B(1, 0) = 15.0f; B(1, 1) = 16.0f;
    B(2, 0) = 17.0f; B(2, 1) = 18.0f;
    B(3, 0) = 19.0f; B(3, 1) = 20.0f;

    // A^T = [[1, 4, 7, 10], [2, 5, 8, 11], [3, 6, 9, 12]]
    A.transpose_multiply_into(B, result);

    // Verify a few key elements
    // result(0,0) = 1*13 + 4*15 + 7*17 + 10*19 = 13 + 60 + 119 + 190 = 382
    // result(0,1) = 1*14 + 4*16 + 7*18 + 10*20 = 14 + 64 + 126 + 200 = 404
    // result(1,0) = 2*13 + 5*15 + 8*17 + 11*19 = 26 + 75 + 136 + 209 = 446
    // result(2,1) = 3*14 + 6*16 + 9*18 + 12*20 = 42 + 96 + 162 + 240 = 540
    EXPECT_FLOAT_EQ(result(0, 0), 382.0f);
    EXPECT_FLOAT_EQ(result(0, 1), 404.0f);
    EXPECT_FLOAT_EQ(result(1, 0), 446.0f);
    EXPECT_FLOAT_EQ(result(2, 1), 540.0f);
}

// Test with large matrix to verify performance and correctness
TEST(MatrixTest, LargeMatrixTransposeMultiplyInto) {
    // Create larger matrices to test performance
    Matrix A(100, 50);
    Matrix B(100, 30);
    Matrix result(50, 30);

    // Fill with simple values for predictable results
    for (size_t i = 0; i < 100; ++i) {
        for (size_t j = 0; j < 50; ++j) {
            A(i, j) = static_cast<float>(i + j);
        }
    }

    for (size_t i = 0; i < 100; ++i) {
        for (size_t j = 0; j < 30; ++j) {
            B(i, j) = static_cast<float>(i * j + 1);
        }
    }

    // This should not throw and complete in reasonable time
    EXPECT_NO_THROW(A.transpose_multiply_into(B, result));

    // Verify the result has correct dimensions
    EXPECT_EQ(result.N, 50);
    EXPECT_EQ(result.M, 30);
}
