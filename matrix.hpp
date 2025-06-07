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
    Matrix(size_t n, size_t m, float seed) : N(n), M(m), data(n * m, seed) {};
    Matrix(size_t n, size_t m) : Matrix(n, m, 0.0f) {};

    // From a lambda n,m to float
    Matrix(size_t n, size_t m, float (*lambda)(size_t, size_t))
        : N(n), M(m), data(n * m) {
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j < m; ++j) {
                data[i * m + j] = lambda(i, j);
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

    // Index a value
    float& operator()(size_t i, size_t j) {
        if (i >= N || j >= M) {
            throw std::out_of_range("Index out of bounds");
        }
        return data[i * M + j];
    }

    void apply(float (*func)(float)) {
        for (auto& value : data) {
            value = func(value);
        }
    }
};

}  // namespace matrix

#endif  // MATRIX_HPP