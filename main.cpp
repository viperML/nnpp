#include <algorithm>
#include <backward.hpp>
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

float sigmoid_derivative(float x) {
    auto res = sigmoid(x);
    return res * (1 - res);
}

backward::SignalHandling sh{};

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
    Matrix Layer1(16, 1);
    Matrix Layer2(16, 1);
    Matrix Layer3(10, 1);
    Matrix Layer3_actual(10, 1);
    auto Activation1 = Layer1.cloned();
    auto Activation2 = Layer2.cloned();
    auto Activation3 = Layer3.cloned();

    // auto N1_prev = N1.cloned();
    // auto N2_prev = N2.cloned();
    // auto N3_prev = N3.cloned();

    Matrix Bias1(Layer1.N, Layer1.M, init);
    Matrix Bias2(Layer2.N, Layer2.M, init);
    Matrix Bias3(Layer3.N, Layer3.M, init);

    Matrix Weight0(Layer1.N, images[0].N, init);
    Matrix Weight1(Layer2.N, Layer1.N, init);
    Matrix Weight2(Layer3.N, Layer2.N, init);

    auto dW0 = Weight0.clone_seeded(0.0f);
    auto dW1 = Weight1.clone_seeded(0.0f);
    auto dW2 = Weight2.clone_seeded(0.0f);
    auto dB1 = Bias1.clone_seeded(0.0f);
    auto dB2 = Bias2.clone_seeded(0.0f);
    auto dB3 = Bias3.clone_seeded(0.0f);

    auto epochs = std::max(static_cast<size_t>(10000), images.size());
    bool do_break = false;

    for (auto epoch : std::views::iota(static_cast<size_t>(0), epochs)) {
        auto im = images[epoch];
        auto la = labels[epoch];

        for (uint8_t i = 0; i < 10; i++) {
            Layer3_actual(i, 0) = (la == i) ? 1.0f : 0.0f;
        }

        // Forward propagation

        /// First hidden layer
        Weight0.multiply_into(im, Activation1);
        Activation1.sum_into(Bias1);
        Activation1.clone_into(Layer1);
        Layer1.apply(sigmoid);

        /// Second hidden layer
        Weight1.multiply_into(Layer1, Activation2);
        Activation2.sum_into(Bias2);
        Activation2.clone_into(Layer2);
        Layer2.apply(sigmoid);

        /// Final layer
        Weight2.multiply_into(Layer2, Activation3);
        Activation3.sum_into(Bias3);
        Activation3.clone_into(Layer3);
        Layer3.apply(sigmoid);

        // Cost calculation
        float cost = 0.0f;
        for (size_t i = 0; i < 10; i++) {
            auto diff = Layer3(i, 0) - Layer3_actual(i, 0);
            cost += diff * diff;
        }
        if (cost <= 0.001f) {
            do_break = true;
        }
        float learning_rate = 0.1f;  // Fixed learning rate

        // N3.elementwise_into(N3_actual, dB3, [learning_rate](auto n3, auto
        // n3a) {
        //     return (n3 - n3a) * learning_rate;
        // });
        // dB3.multiply_transpose_into(N2, dW2);
        // W2 -= dW2;

        /*
            Calculate dB's
        */
        Layer3.elementwise_into(Layer3_actual, dB3, [](auto n3, auto n3a) {
            return n3 - n3a;
        });

        Weight2.transpose_multiply_into(dB3, dB2);
        dB2.elementwise_into(Activation2, dB2, [](auto left, auto right) {
            return left * sigmoid_derivative(right);
        });

        Weight1.transpose_multiply_into(dB2, dB1);
        dB1.elementwise_into(Activation1, dB1, [](auto left, auto right) {
            return left * sigmoid_derivative(right);
        });

        /*
            Calculate dW's
        */
        dB3.multiply_transpose_into(Layer2, dW2);
        dB2.multiply_transpose_into(Layer1, dW1);
        dB1.multiply_transpose_into(im, dW0);

        /*
            Apply deltas
        */
        Bias3.elementwise_into(dB3, Bias3, [learning_rate](auto b, auto db) {
            return b - db * learning_rate;
        });
        Bias2.elementwise_into(dB2, Bias2, [learning_rate](auto b, auto db) {
            return b - db * learning_rate;
        });
        Bias1.elementwise_into(dB1, Bias1, [learning_rate](auto b, auto db) {
            return b - db * learning_rate;
        });
        Weight2.elementwise_into(dW2, Weight2, [learning_rate](auto w, auto dw) {
            return w - dw * learning_rate;
        });
        Weight1.elementwise_into(dW1, Weight1, [learning_rate](auto w, auto dw) {
            return w - dw * learning_rate;
        });
        Weight0.elementwise_into(dW0, Weight0, [learning_rate](auto w, auto dw) {
            return w - dw * learning_rate;
        });

        if (epoch % (epochs / 10) == 0 || do_break) {
            std::cout << std::fixed;
            std::cout << std::endl << "=> Epoch: " << epoch << std::endl;
            std::cout << "Cost: " << cost << std::endl;

            // Prediction
            std::cout << "Predicted:\t";
            for (size_t i = 0; i < 10; i++) {
                std::cout << Layer3(i, 0) << " ";
            }
            std::cout << std::endl;
            std::cout << "Actual:\t\t";
            for (size_t i = 0; i < 10; i++) {
                std::cout << Layer3_actual(i, 0) << " ";
            }
        }

        if (do_break) break;
    }

    return 0;
}
