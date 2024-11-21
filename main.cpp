#include "matrix.hpp"
#include <cstdlib>
#include <iostream>
#include <cstddef>
#include <cstdlib>
#include <memory>

template <typename T> Matrix<T>::Matrix(size_t size) {
  this->size = size;
  auto p = (T *)calloc(size, sizeof(T) * this->size);

  this->ptr = std::unique_ptr<T>(p);
}


int main() {
  std::cout << "Hello" << std::endl;

  auto m = Matrix<double>(2);

  std::cout << m.size << std::endl;

  return EXIT_SUCCESS;
}
