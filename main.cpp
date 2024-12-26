#include <cstdlib>
#include <iostream>
#include <optional>
#include <structopt/app.hpp>
#include <structopt/exception.hpp>

#include "matrix.h"
#include "mnist.h"

struct Cli {
    std::optional<bool> verbose = false;
};

STRUCTOPT(Cli, verbose);

int main(int argc, char** argv) {
    Cli cli {};
    try {
        cli = structopt::app("nn++").parse<Cli>(argc, argv);

    } catch (structopt::exception& e) {
        std::cout << e.what() << std::endl;
        std::cout << e.help() << std::endl;
        return EXIT_FAILURE;
    }

    auto A = Matrix<double>(3, 3);
    A[0, 0] = 2.0;
    A[0, 1] = 3.0;
    A[0, 2] = 2.0;
    A[1, 0] = 2.0;
    A[1, 1] = 3.0;
    A[1, 2] = 1.0;
    A[2, 0] = 2.0;
    A[2, 1] = 1.0;
    A[2, 2] = 0.0;
    std::cout << A.display() << std::endl;

    auto B = Matrix<double>(3, 2);
    B[0, 0] = 3;
    B[0, 1] = 1;
    B[1, 0] = 3;
    B[1, 1] = 2;
    B[2, 0] = 1;
    B[2, 1] = 0;
    std::cout << B.display() << std::endl;

    auto C = A * B;
    std::cout << C.display() << std::endl;

    return EXIT_SUCCESS;
}
