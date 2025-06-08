#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <random>
#include <ranges>

#include "matrix.hpp"
#include "mnist.hpp"

using namespace matrix;

float sigmoid(float x) {
    auto res = std::exp(x);
    return res / (1 + res);
}

int main() {
    auto cwd = std::filesystem::current_path();

    std::cout << "Loading MNIST dataset..." << std::endl;
    auto [images, labels] = mnist::load_mnist(
        (cwd / "data" / "train-images.idx3-ubyte").string(),
        (cwd / "data" / "train-labels.idx1-ubyte").string()
    );
    std::cout << "Done" << std::endl;

    images[0].print();

    std::cout << "Number of images: " << images.size() << std::endl;
    std::cout << "Number of labels: " << labels.size() << std::endl;

    // RNG
    // std::random_device rd;
    std::mt19937 gen(0);
    std::uniform_real_distribution<> dist(-0.1, 0.1);
    auto init = [&dist, &gen](size_t i, size_t j) {
        return static_cast<float>(dist(gen));
    };

    // N0 is images[number]
    auto N1 = Matrix(16, 1);
    auto N2 = Matrix(16, 1);
    auto N3_predict = Matrix(10, 1);
    auto N3_actual = Matrix(10, 1);

    auto B1 = Matrix(N1.N, N1.M, init);
    auto B2 = Matrix(N2.N, N2.M, init);
    auto B3 = Matrix(N3_predict.N, N3_predict.M, init);

    auto W0 = Matrix(N1.N, images[0].N, init);
    auto W1 = Matrix(N2.N, N1.N, init);
    auto W2 = Matrix(N3_predict.N, N2.N, init);

    auto epochs = 100;

    for (auto epoch : std::views::iota(0, epochs)) {
        auto doPrint = epoch % 10 == 0;

        doPrint && std::cout <<  std::endl << "=> Epoch " << epoch << std::endl;

        auto im = images[epoch];
        auto la = labels[epoch];

        for (uint8_t i = 0; i < 10; i++) {
            N3_actual(i, 0) = (la == i) ? 1.0f : 0.0f;
        }

        // Forward propagation

        /// First hidden layer
        W0.multiply_into(im, N1);
        N1.sum_into(B1);
        N1.apply(sigmoid);

        /// Second hidden layer
        W1.multiply_into(N1, N2);
        N2.sum_into(B2);
        N2.apply(sigmoid);

        /// Final layer
        W2.multiply_into(N2, N3_predict);
        N3_predict.sum_into(B3);
        N3_predict.apply(sigmoid);
        doPrint ? N3_predict.print() : void();

        // Cost calculation
        float cost = 0.0f;
        for (size_t i = 0; i < 10; i++) {
            auto diff = N3_predict(i, 0) - N3_actual(i, 0);
            cost += diff * diff;
        }

        doPrint && std::cout << std::fixed << "Cost: " << cost  << std::endl;
    }

    return 0;
}
