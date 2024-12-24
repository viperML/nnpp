#pragma once

#include <cstddef>
#include <format>
#include <iostream>
#include <ranges>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

template <typename T,
          typename = typename std::enable_if<std::is_arithmetic_v<T>>>
class Matrix {
public:
  size_t n;
  size_t m;

  Matrix(size_t n, size_t m) : n(n), m(m), data(n * m, 0) {}
  ~Matrix() {}

  std::string display() {
    auto res = std::string{};
    for (auto i : std::views::iota(size_t(0), this->n)) {
      for (auto j : std::views::iota(size_t(0), this->m)) {
        auto s = std::format("{:>5} ", this->get(i, j));
        res.append(s);
      }
      res.append("\n");
    }
    return res;
  }

  T &get(size_t i, size_t j) {
    return this->data.at(this->m * i + j);
  }

  void set(size_t i, size_t j, T value) {
    this->get(i, j) = value;
  }

private:
  std::vector<T> data;
};
