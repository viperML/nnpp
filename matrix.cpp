#include "matrix.hpp"

namespace matrix {

// Performs result = this * B
// Result must be a valid matrix with the proper size.
void Matrix::multiply_into(const Matrix& B, Matrix& result) {
    // Check if multiplication is valid: this->M must equal B.N
    if (this->M != B.N) {
        throw std::runtime_error(
            "Matrix dimensions incompatible for multiplication");
    }

    // Check if result matrix has correct dimensions: this->N x B.M
    if (result.N != this->N || result.M != B.M) {
        throw std::runtime_error("Result matrix has incorrect dimensions");
    }

    // Perform matrix multiplication: C[i][j] = sum(A[i][k] * B[k][j])
    for (size_t i = 0; i < this->N; ++i) {
        for (size_t j = 0; j < B.M; ++j) {
            float sum = 0.0f;
            for (size_t k = 0; k < this->M; ++k) {
                sum += this->data[i * this->M + k] * B.data[k * B.M + j];
            }
            result.data[i * result.M + j] = sum;
        }
    }
}

// Performs result = this * B^T (multiply by transpose of B)
void Matrix::multiply_transpose_into(const Matrix& B, Matrix& result) {
    // Check if multiplication is valid: this->M must equal B.M (since we're using B^T)
    if (this->M != B.M) {
        throw std::runtime_error(
            "Matrix dimensions incompatible for multiplication with transpose");
    }

    // Check if result matrix has correct dimensions: this->N x B.N
    if (result.N != this->N || result.M != B.N) {
        throw std::runtime_error("Result matrix has incorrect dimensions");
    }

    // Perform matrix multiplication with transpose: C[i][j] = sum(A[i][k] * B[j][k])
    for (size_t i = 0; i < this->N; ++i) {
        for (size_t j = 0; j < B.N; ++j) {
            float sum = 0.0f;
            for (size_t k = 0; k < this->M; ++k) {
                sum += this->data[i * this->M + k] * B.data[j * B.M + k];
            }
            result.data[i * result.M + j] = sum;
        }
    }
}

// Element-wise multiplication (Hadamard product)
void Matrix::hadamard_into(const Matrix& B, Matrix& result) {
    if (N != B.N || M != B.M || N != result.N || M != result.M) {
        throw std::invalid_argument("Matrix dimensions must match for Hadamard product");
    }
    for (size_t i = 0; i < N * M; ++i) {
        result.data[i] = data[i] * B.data[i];
    }
}

}  // namespace matrix