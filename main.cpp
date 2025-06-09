#include <algorithm>
#include <backward.hpp>
#include <chrono>
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
    std::mt19937 gen(0);
    std::uniform_real_distribution<> dist(-0.1, 0.1);
    auto init = [&dist, &gen](size_t i, size_t j) {
        return static_cast<float>(dist(gen));
    };

    // Declarative layers
    Matrix RealLayer(10, 1);
    std::vector<Matrix> Layer = {
        Matrix(images[0].N, 1),
        // Matrix(128, 1),
        Matrix(16, 1),
        Matrix(16, 1),
        Matrix(RealLayer.N, 1)
    };

    // Declarative Biases
    std::vector<Matrix> Bias{};
    std::vector<Matrix> Activation{};
    for (auto layer : Layer) {
        Bias.push_back(layer.cloned());
        Activation.push_back(layer.cloned());
    }

    // Declarative Weights
    std::vector<Matrix> Weight{};
    for (size_t i = 0; i < Layer.size() - 1; i++) {
        Weight.push_back(Matrix(Layer[i + 1].N, Layer[i].N, init));
    }

    // Declarative deltas
    std::vector<Matrix> dWeight{};
    for (auto w : Weight) {
        dWeight.push_back(w.clone_seeded(0.0f));
    }
    std::vector<Matrix> dBias{};
    for (auto b : Bias) {
        dBias.push_back(b.clone_seeded(0.0f));
    }

    auto epochs = std::max(static_cast<size_t>(10000), images.size());
    bool do_break = false;

    std::vector<float> window{};

    auto now = std::chrono::steady_clock::now();

    for (size_t epoch = 0; epoch < epochs; epoch++) {
        auto image = images[epoch];
        auto label = labels[epoch];

        // Populate real layer
        for (uint8_t i = 0; i < 10; i++) {
            RealLayer(i, 0) = (label == i) ? 1.0f : 0.0f;
        }

        // Copy input image to first layer
        image.clone_into(Layer[0]);
        image.clone_into(Activation[0]);

        // Forward propagation
        for (size_t i = 0; i < Weight.size(); i++) {
            Weight[i].multiply_into(Layer[i], Activation[i + 1]);
            Activation[i + 1].sum_into(Bias[i + 1]);
            Activation[i + 1].clone_into(Layer[i + 1]);
            Layer[i + 1].apply(sigmoid);
        }

        // Cost calculation
        float cost = 0.0f;
        for (size_t i = 0; i < RealLayer.N; i++) {
            auto diff = Layer.back()(i, 0) - RealLayer(i, 0);
            cost += diff * diff;
        }
        float learning_rate = 0.1f;  // Fixed learning rate

        // Backpropagation
        // Calculate output layer error
        Layer.back().elementwise_into(
            RealLayer, dBias.back(), [](auto n, auto r) { return n - r; }
        );

        // Backpropagate through hidden layers
        for (int i = static_cast<int>(Weight.size()) - 1; i >= 0; i--) {
            if (i > 0) {
                // Calculate dBias for hidden layers
                Weight[i].transpose_multiply_into(dBias[i + 1], dBias[i]);
                dBias[i].elementwise_into(
                    Activation[i],
                    dBias[i],
                    [](auto left, auto right) {
                        return left * sigmoid_derivative(right);
                    }
                );
            }

            // Calculate dWeight
            dBias[i + 1].multiply_transpose_into(Layer[i], dWeight[i]);
        }
        // Apply deltas to biases (skip input layer at index 0)
        for (size_t i = 1; i < Bias.size(); i++) {
            Bias[i].elementwise_into(
                dBias[i],
                Bias[i],
                [learning_rate](auto b, auto db) {
                    return b - db * learning_rate;
                }
            );
        }

        // Apply deltas to weights
        for (size_t i = 0; i < Weight.size(); i++) {
            Weight[i].elementwise_into(
                dWeight[i],
                Weight[i],
                [learning_rate](auto w, auto dw) {
                    return w - dw * learning_rate;
                }
            );
        }

        float max_pred = 0.0f;
        for (uint8_t i = 0; i<10; i++) {
            if (Layer.back()(i, 0) > max_pred) {
                max_pred = Layer.back()(i, 0);
            }
        }
        window.push_back(max_pred);
        if (epoch % 30 == 0) {
            // Calculate the average of the window
            float avg = std::accumulate(window.begin(), window.end(), 0.0f) / window.size();
            if (avg > 0.9f) {
                do_break = true;
            }
            window.clear();
        }


        if (epoch % 1000 == 0 || do_break) {
            std::cout << std::fixed;
            std::cout << std::endl << "=> Epoch: " << epoch << std::endl;
            std::cout << "Cost: " << cost << std::endl;

            // Prediction
            std::cout << "Predicted:\t";
            for (size_t i = 0; i < RealLayer.N; i++) {
                std::cout << Layer.back()(i, 0) << " ";
            }
            std::cout << std::endl;
            std::cout << "Actual:\t\t";
            for (size_t i = 0; i < RealLayer.N; i++) {
                std::cout << RealLayer(i, 0) << " ";
            }
        }

        if (do_break) {
            auto elapsed = std::chrono::steady_clock::now() - now;
            auto t = std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count();
            std::cout << std::endl << "Training complete in " << t << " ms." << std::endl;
            break;
        };
    }

    return 0;
}
