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

  auto m = Matrix<double>(3, 3);

  return EXIT_SUCCESS;
}
