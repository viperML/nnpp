#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <random>
#include <ranges>

#include "matrix.hpp"
#include "mnist.hpp"

using namespace matrix;
namespace views = std::views;

float relu(float x) {
    return (x > 0) ? x : 0.0f;
}

int main() {
    auto cwd = std::filesystem::current_path();

    std::cout << "Loading MNIST dataset..." << std::endl;
    auto [images, labels] =
        mnist::load_mnist((cwd / "data" / "train-images.idx3-ubyte").string(),
                          (cwd / "data" / "train-labels.idx1-ubyte").string());
    std::cout << "Done" << std::endl;

    std::cout << "Number of images: " << images.size() << std::endl;
    std::cout << "Number of labels: " << labels.size() << std::endl;

    // RNG
    // std::random_device rd;
    std::mt19937 gen(0);
    std::uniform_real_distribution<> dist(-1.0, 1.0);

    auto N1 = Matrix(28 * 28, 1);
    auto N2 = Matrix(16, 1);
    auto N3 = Matrix(16, 1);
    auto N_final = Matrix(10, 1);
    auto N_final_real = Matrix(10, 1);

    auto B1 = Matrix(N1.N, N1.M);
    auto B2 = Matrix(N2.N, N2.M);
    auto B3 = Matrix(N3.N, N3.M);

    auto W1 = Matrix(N2.N, N1.N, 0.5);
    auto W2 = Matrix(N3.N, N2.N, 0.5);
    auto W3 = Matrix(N_final.N, N3.N, 0.5);

    auto epochs = 3;

    for (auto epoch : std::views::iota(0, epochs)) {
        // if (epoch % 100 == 0) {
        std::cout << "Epoch: " << epoch << std::endl;
        // }

        // Unfold image into N1
        auto& im = images[epoch];
        for (auto i = 0; i < im.N; i++) {
            for (auto j = 0; j < im.M; j++) {
                N1(i * im.M + j, 0) = im(i, j);
            }
        }
        // Unfold label into N3_real
        for (auto i : std::views::iota(0, 10)) {
            N_final_real(i, 0) = (labels[epoch] == i) ? 1.0f : 0.0f;
        }

        // Forward propagation
        W1.multiply_into(N1, N2);
        N2.apply(relu);
        W2.multiply_into(N2, N3);
        N3.apply(relu);
        W3.multiply_into(N3, N_final);
        N_final.apply(relu);

        // Cost
        float cost = 0.0;
        for (auto i : views::iota(0, 10)) {
            auto res = N_final_real(i, 0) - N_final(i, 0);
            cost += res * res;
        }

    }

    return 0;
}
