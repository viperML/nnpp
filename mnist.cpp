#include "mnist.hpp"

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <stdexcept>

#include "matrix.hpp"

namespace mnist {

// Helper function to read big-endian 32-bit integer
uint32_t read_uint32_be(std::ifstream& file) {
    uint32_t value;
    file.read(reinterpret_cast<char*>(&value), sizeof(value));
    // Convert from big-endian to host byte order
    return ((value & 0xFF000000) >> 24) | ((value & 0x00FF0000) >> 8) |
           ((value & 0x0000FF00) << 8) | ((value & 0x000000FF) << 24);
}

// Function to load MNIST training dataset
std::pair<std::vector<matrix::Matrix>, std::vector<uint8_t>> load_mnist(
    const std::string& images_path, const std::string& labels_path
) {
    std::vector<matrix::Matrix> images;
    std::vector<uint8_t> labels;

    // Open images file
    std::ifstream images_file(images_path, std::ios::binary);
    if (!images_file.is_open()) {
        throw std::runtime_error("Cannot open images file: " + images_path);
    }

    // Read images header
    uint32_t images_magic = read_uint32_be(images_file);
    if (images_magic != 2051) {
        throw std::runtime_error("Invalid magic number in images file");
    }

    uint32_t num_images = read_uint32_be(images_file);
    uint32_t rows = read_uint32_be(images_file);
    uint32_t cols = read_uint32_be(images_file);

    // Open labels file
    std::ifstream labels_file(labels_path, std::ios::binary);
    if (!labels_file.is_open()) {
        throw std::runtime_error("Cannot open labels file: " + labels_path);
    }

    // Read labels header
    uint32_t labels_magic = read_uint32_be(labels_file);
    if (labels_magic != 2049) {
        throw std::runtime_error("Invalid magic number in labels file");
    }

    uint32_t num_labels = read_uint32_be(labels_file);
    if (num_images != num_labels) {
        throw std::runtime_error("Number of images and labels don't match");
    }

    // Reserve space for efficiency
    images.reserve(num_images);
    labels.reserve(num_images);

    // Read image and label data
    // for (uint32_t i = 0; i < num_images; ++i) {
    for (size_t i = 0; i < num_images; i++) {
        matrix::Matrix image(rows * cols, 1);

        // Read pixel data
        for (size_t row = 0; row < rows; ++row) {
            for (size_t col = 0; col < cols; ++col) {
                uint8_t pixel;
                images_file.read(reinterpret_cast<char*>(&pixel), 1);
                // Normalize pixel value to [0,1] range
                image(row * cols + col, 0) = static_cast<float>(pixel) / 255.0f;
            }
        }

        // Read label
        uint8_t label;
        labels_file.read(reinterpret_cast<char*>(&label), 1);

        images.push_back(std::move(image));
        labels.push_back(label);
    }

    return std::make_pair(std::move(images), std::move(labels));
}

}  // namespace mnist
