#pragma once

#include <cstddef>
#include <type_traits>
template <typename T,
          typename = typename std::enable_if<std::is_arithmetic_v<T>>>
class Matrix {
public:
  size_t n;
  size_t m;

  Matrix(size_t n, size_t m) : n(n), m(m) { this->data = new T[n * m](); }
  ~Matrix() { delete[] this->data; }

private:
  T *data;
};
