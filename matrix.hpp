#include <cstddef>
#include <iostream>
#include <vector>

namespace matrix {

class Matrix {
  private:
    size_t N;
    size_t M;
    std::vector<float> data;

  public:
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
        for (size_t i = 0; i < N; ++i) {
            for (size_t j = 0; j < M; ++j) {
                std::cout << std::scientific;
                std::cout.precision(3);
                std::cout << data[i * M + j] << " ";
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
};

}  // namespace matrix