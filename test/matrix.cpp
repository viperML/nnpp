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
