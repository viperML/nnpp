#ifndef MATRIX_HPP
#define MATRIX_HPP

#include <cstddef>
#include <iostream>
#include <vector>

namespace matrix {

class Matrix {
  private:
    std::vector<float> data;

  public:
    size_t N;
    size_t M;
    Matrix(size_t n, size_t m, float seed) : data(n * m, seed), N(n), M(m) {};
    Matrix(size_t n, size_t m) : Matrix(n, m, 0.0f) {};

    // Template constructor for function objects (lambdas with captures)
    template <typename Func>
    Matrix(size_t n, size_t m, Func func) : data(n * m), N(n), M(m) {
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j < m; ++j) {
                data[i * m + j] = func(i, j);
            }
        }
    }

    void print() {
        std::cout << "Matrix " << this->N << "x" << this->M << ": "
                  << std::endl;

        size_t max_rows = std::min(N, static_cast<size_t>(5));
        size_t max_cols = std::min(M, static_cast<size_t>(5));

        for (size_t i = 0; i < max_rows; ++i) {
            for (size_t j = 0; j < max_cols; ++j) {
                std::cout << std::scientific;
                std::cout.precision(3);
                std::cout << data[i * M + j] << " ";
            }
            if (M > 5) {
                std::cout << "... ";
            }
            std::cout << std::endl;
        }

        if (N > 5) {
            for (size_t j = 0; j < max_cols; ++j) {
                std::cout << "... ";
            }
            if (M > 5) {
                std::cout << "... ";
            }
            std::cout << std::endl;
        }
    }

    // Performs result=this*B
    // Result must be a valid matrix with the proper size.
    void multiply_into(const Matrix& B, Matrix& result);

    // Performs result=this*B^T (multiply by transpose of B)
    void multiply_transpose_into(const Matrix& B, Matrix& result);

    // Element-wise multiplication (Hadamard product)
    void hadamard_into(const Matrix& B, Matrix& result);

    // Index a value
    float& operator()(size_t i, size_t j) {
        if (i >= N || j >= M) {
            throw std::out_of_range("Index out of bounds");
        }
        return data[i * M + j];
    }

    // Const version of operator()
    const float& operator()(size_t i, size_t j) const {
        if (i >= N || j >= M) {
            throw std::out_of_range("Index out of bounds");
        }
        return data[i * M + j];
    }

    template <typename Func>
    void apply(Func func) {
        for (auto& value : data) {
            value = func(value);
        }
    }

    // Element-wise sum all elements of B into this
    void sum_into(Matrix& B) {
        if (N != B.N || M != B.M) {
            throw std::invalid_argument("Matrix dimensions must match for sum");
        }
        for (size_t i = 0; i < N * M; ++i) {
            data[i] += B.data[i];
        }
    }

    // Element-wise subtraction: this = this - B
    void subtract(const Matrix& B) {
        if (N != B.N || M != B.M) {
            throw std::invalid_argument("Matrix dimensions must match for subtraction");
        }
        for (size_t i = 0; i < N * M; ++i) {
            data[i] -= B.data[i];
        }
    }

    // Scalar multiplication: this = this * scalar
    void scale(float scalar) {
        for (auto& value : data) {
            value *= scalar;
        }
    }

    // Create transpose matrix
    Matrix transpose() const {
        Matrix result(M, N);
        for (size_t i = 0; i < N; ++i) {
            for (size_t j = 0; j < M; ++j) {
                result(j, i) = (*this)(i, j);
            }
        }
        return result;
    }

    // Copy constructor and assignment operator
    Matrix(const Matrix& other) : data(other.data), N(other.N), M(other.M) {}
    Matrix& operator=(const Matrix& other) {
        if (this != &other) {
            data = other.data;
            N = other.N;
            M = other.M;
        }
        return *this;
    }

    // Fill matrix with zeros
    void zero() {
        std::fill(data.begin(), data.end(), 0.0f);
    }
};

}  // namespace matrix

#endif  // MATRIX_HPP