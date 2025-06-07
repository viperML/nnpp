#include <cstddef>
#include "matrix.hpp"

using namespace matrix;

int main() {

    auto a = Matrix(2, 3, [](size_t i, size_t j) {
        return 1.0f * (i*3 + j + 1);
    });

    auto b = Matrix(3,2, [](auto i, auto j) {
        return 1.0f * ((i+1)*10 + j);
    });


    a.print();
    b.print();

    auto res = Matrix(2,2);

    a.multiply_into(b, res);
    res.print();

    return 0;
}
