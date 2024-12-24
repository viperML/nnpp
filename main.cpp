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

  auto m = Matrix<double>(4, 3);

  m.set(0, 0, 1.1);
  m.set(1, 0, 2.1);
  m.set(1, 1, 3.1);
  std::cout << m.display() << std::endl;

  return EXIT_SUCCESS;
}
