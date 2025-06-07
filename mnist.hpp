#include <vector>
#include <string>
#include <utility>
#include <cstdint>
#include "matrix.hpp"

namespace mnist {

// Function to load MNIST training dataset
std::pair<std::vector<matrix::Matrix>, std::vector<uint8_t>> load_mnist(
    const std::string& images_path = "./data/train-images.idx3-ubyte",
    const std::string& labels_path = "./data/train-labels.idx1-ubyte"
);

}