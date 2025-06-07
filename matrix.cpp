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

}  // namespace matrix