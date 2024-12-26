#pragma once

#include <cstddef>
#include <format>
#include <iostream>
#include <ranges>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

template<class T>
concept Numeric = std::is_arithmetic_v<T>;

template<Numeric T>
class Matrix {
  public:
    size_t n;
    size_t m;

    Matrix(size_t n, size_t m) : n(n), m(m), data(n * m, 0) {}

    Matrix(size_t n, size_t m, T seed) : n(n), m(m), data(n * m, seed) {}

    // ~Matrix() {}

    std::string display() {
        auto res = std::string {};
        for (auto i : std::views::iota(size_t(0), this->n)) {
            for (auto j : std::views::iota(size_t(0), this->m)) {
                auto s = std::format("{:>5} ", this->get(i, j));
                res.append(s);
            }
            res.append("\n");
        }
        return res;
    }

    T& get(size_t i, size_t j) {
        return this->data.at(this->m * i + j);
    }

    T& operator[](size_t i, size_t j) {
        return this->get(i, j);
    }

    void set(size_t i, size_t j, T value) {
        this->get(i, j) = value;
    }

    Matrix<T> operator*(Matrix<T>& B) {
        if (this->m != B.n)
            throw std::out_of_range("Matrix dimensions don't match");

        auto C = Matrix<T>(this->n, B.m);

        for (size_t i : std::views::iota(size_t(0), C.n)) {
            for (size_t j : std::views::iota(size_t(0), C.m)) {
                auto res = 0;
                // iterate A on row i
                // iterate B on column j
                for (auto k : std::views::iota(size_t(0), this->m)) {
                    res += this->get(i, k) * B.get(k, j);
                }

                C[i, j] = res;
            }
        }

        return C;
    }

  private:
    std::vector<T> data;
};
