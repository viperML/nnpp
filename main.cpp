#include "matrix.h"

#include <cstdlib>
#include <iostream>
#include <optional>
#include <structopt/app.hpp>
#include <structopt/exception.hpp>

struct Cli {
  std::optional<bool> verbose = false;
};
STRUCTOPT(Cli, verbose);

int main(int argc, char **argv) {
  Cli cli{};
  try {
    cli = structopt::app("nn++").parse<Cli>(argc, argv);

  } catch (structopt::exception &e) {
    std::cout << e.what() << std::endl;
    std::cout << e.help() << std::endl;
    return EXIT_FAILURE;
  }

  auto A = Matrix<double>(3, 3, -1.0);
  auto B = Matrix<double>(3, 3, 2.0);

  auto C = A * B;
  std::cout << C.display() << std::endl;

  return EXIT_SUCCESS;
}
