#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <random>
#include <ranges>

#include "matrix.hpp"
#include "mnist.hpp"

using namespace matrix;

float relu(float x) { return (x > 0) ? x : 0.0f; }
float relu_derivative(float x) { return (x > 0) ? 1.0f : 0.0f; }

// Softmax activation for output layer
void softmax(Matrix& x) {
    float max_val = x(0, 0);
    for (size_t i = 1; i < x.N; i++) {
        if (x(i, 0) > max_val) max_val = x(i, 0);
    }

    float sum = 0.0f;
    for (size_t i = 0; i < x.N; i++) {
        x(i, 0) = std::exp(x(i, 0) - max_val);
        sum += x(i, 0);
    }

    for (size_t i = 0; i < x.N; i++) {
        x(i, 0) /= sum;
    }
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
    std::uniform_real_distribution<> dist(
        -0.1, 0.1
    );  // Smaller weight initialization
    auto init = [&dist, &gen](size_t i, size_t j) {
        return static_cast<float>(dist(gen));
    };

    // N0 is images[number]
    auto N1 = Matrix(16, 1);
    auto N2 = Matrix(16, 1);
    auto N3_predict = Matrix(10, 1);
    auto N3_actual = Matrix(10, 1);

    // Store activations before applying activation function (needed for
    // backprop)
    auto Z1 = Matrix(16, 1);
    auto Z2 = Matrix(16, 1);
    auto Z3 = Matrix(10, 1);

    auto B1 = Matrix(N1.N, N1.M, 0.0f);
    auto B2 = Matrix(N2.N, N2.M, 0.0f);
    auto B3 = Matrix(N3_predict.N, N3_predict.M, 0.0f);

    auto W0 = Matrix(N1.N, images[0].N, init);
    auto W1 = Matrix(N2.N, N1.N, init);
    auto W2 = Matrix(N3_predict.N, N2.N, init);

    // Gradient matrices
    auto dW0 = Matrix(W0.N, W0.M);
    auto dW1 = Matrix(W1.N, W1.M);
    auto dW2 = Matrix(W2.N, W2.M);
    auto dB1 = Matrix(B1.N, B1.M);
    auto dB2 = Matrix(B2.N, B2.M);
    auto dB3 = Matrix(B3.N, B3.M);

    // Error/delta matrices for backpropagation
    auto delta3 = Matrix(10, 1);
    auto delta2 = Matrix(16, 1);
    auto delta1 = Matrix(16, 1);

    // Temporary matrices for computations
    auto temp1 = Matrix(16, 1);
    auto temp2 = Matrix(16, 1);
    auto temp3 = Matrix(10, 1);

    float learning_rate = 0.001f;  // Lower learning rate
    auto epochs = 1000;

    for (auto epoch : std::views::iota(0, epochs)) {
        if (epoch % 100 == 0) {
            std::cout << std::endl << "=> Epoch: " << epoch << std::endl;
        }

        float total_cost = 0.0f;
        int num_samples = std::min(
            1000, static_cast<int>(images.size())
        );  // Train on first 1000 samples

        for (int sample = 0; sample < num_samples; sample++) {
            auto im = images[sample];
            auto la = labels[sample];

            // Set up one-hot encoded target
            for (uint8_t i = 0; i < 10; i++) {
                N3_actual(i, 0) = (la == i) ? 1.0f : 0.0f;
            }

            // Forward propagation

            /// First hidden layer
            W0.multiply_into(im, Z1);
            Z1.sum_into(B1);
            N1 = Z1;  // Copy for storing pre-activation
            N1.apply(relu);

            /// Second hidden layer
            W1.multiply_into(N1, Z2);
            Z2.sum_into(B2);
            N2 = Z2;  // Copy for storing pre-activation
            N2.apply(relu);

            /// Final layer
            W2.multiply_into(N2, Z3);
            Z3.sum_into(B3);
            N3_predict = Z3;      // Copy for storing pre-activation
            softmax(N3_predict);  // Use softmax for output layer

            // Cost calculation (cross-entropy loss)
            float cost = 0.0f;
            for (size_t i = 0; i < 10; i++) {
                if (N3_actual(i, 0) > 0.5f) {  // This is the correct class
                    cost -= std::log(
                        N3_predict(i, 0) + 1e-15f
                    );  // Add small epsilon to prevent log(0)
                }
            }
            total_cost += cost;

            // Backpropagation

            // Output layer error (delta3) - for softmax + cross-entropy,
            // gradient is simply (predicted - actual)
            for (size_t i = 0; i < 10; i++) {
                delta3(i, 0) = N3_predict(i, 0) - N3_actual(i, 0);
            }

            // Hidden layer 2 error (delta2)
            W2.transpose().multiply_into(delta3, temp2);
            for (size_t i = 0; i < 16; i++) {
                delta2(i, 0) = temp2(i, 0) * relu_derivative(Z2(i, 0));
            }

            // Hidden layer 1 error (delta1)
            W1.transpose().multiply_into(delta2, temp1);
            for (size_t i = 0; i < 16; i++) {
                delta1(i, 0) = temp1(i, 0) * relu_derivative(Z1(i, 0));
            }

            // Compute gradients

            // dW2 = delta3 * N2^T
            delta3.multiply_transpose_into(N2, dW2);
            // dB3 = delta3
            dB3 = delta3;

            // dW1 = delta2 * N1^T
            delta2.multiply_transpose_into(N1, dW1);
            // dB2 = delta2
            dB2 = delta2;

            // dW0 = delta1 * im^T
            delta1.multiply_transpose_into(im, dW0);
            // dB1 = delta1
            dB1 = delta1;

            // Update weights and biases
            dW2.scale(learning_rate);
            W2.subtract(dW2);
            dB3.scale(learning_rate);
            B3.subtract(dB3);

            dW1.scale(learning_rate);
            W1.subtract(dW1);
            dB2.scale(learning_rate);
            B2.subtract(dB2);

            dW0.scale(learning_rate);
            W0.subtract(dW0);
            dB1.scale(learning_rate);
            B1.subtract(dB1);
        }

        // Test on a sample to see prediction
        if (epoch % 2 == 0) {
            auto test_im = images[0];
            auto test_label = labels[0];

            W0.multiply_into(test_im, N1);
            N1.sum_into(B1);
            N1.apply(relu);

            W1.multiply_into(N1, N2);
            N2.sum_into(B2);
            N2.apply(relu);

            W2.multiply_into(N2, N3_predict);
            N3_predict.sum_into(B3);
            softmax(N3_predict);

            // Find predicted class (highest probability)
            int predicted_class = 0;
            float max_prob = N3_predict(0, 0);
            for (int i = 1; i < 10; i++) {
                if (N3_predict(i, 0) > max_prob) {
                    max_prob = N3_predict(i, 0);
                    predicted_class = i;
                }
            }

            if (epoch % 10 == 0 || epoch == epochs) {
                std::cout << std::fixed
                          << "Average Cost: " << total_cost / num_samples
                          << std::endl;
                std::cout << "Test sample - True label: "
                          << static_cast<int>(test_label)
                          << ", Predicted: " << predicted_class
                          << " (confidence: " << std::fixed
                          << std::setprecision(3) << max_prob << ")"
                          << std::endl;
            }
        }
    }

    return 0;
}
