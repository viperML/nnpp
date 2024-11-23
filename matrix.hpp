#pragma once

#include <cstddef>
#include <iostream>
#include <memory>

template <typename T> class Matrix {
public:
  Matrix<T>(size_t n, size_t m) { this->ptr = std::unique_ptr<T>(new T[n*m]); }
  ~Matrix<T>() { this->ptr.release(); }

private:
  std::unique_ptr<T> ptr;
};
