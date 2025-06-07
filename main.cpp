#include <cstddef>
#include <filesystem>
#include <iostream>

#include "matrix.hpp"
#include "mnist.hpp"

using namespace matrix;

int main() {
    auto cwd = std::filesystem::current_path();

    std::cout << "Loading MNIST dataset..." << std::endl;
    auto [images, labels] =
        mnist::load_mnist((cwd / "data" / "train-images.idx3-ubyte").string(),
                          (cwd / "data" / "train-labels.idx1-ubyte").string());
    std::cout << "Done" << std::endl;

    std::cout << "Number of images: " << images.size() << std::endl;
    std::cout << "Number of labels: " << labels.size() << std::endl;


    auto im = images[0];
    auto la = labels[0];

    std::cout << "First image dimensions: " << im.N << "x" << im.M << std::endl;
    std::cout << "First label: " << static_cast<int>(la) << std::endl;

    im.print();

    return 0;
}
