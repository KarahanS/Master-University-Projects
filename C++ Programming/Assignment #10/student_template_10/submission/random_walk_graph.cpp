#include "random_walk_graph.h"

#include <algorithm>
#include <fstream>
#include <ios>
#include <numeric>
#include <random>
#include <iostream>
#include <stdexcept>

void RandomWalkGraph::simulate_random_walk(uint32_t num_steps) {
    // TODO: 10.3 a)
    size_t num_nodes = size();
    std::uniform_int_distribution<uint32_t> dice {0, static_cast<uint32_t>(num_nodes-1)};
    uint32_t current_node = dice(prng);
    for (uint32_t step=0; step<num_steps; step++) {
        if (current_node >= edges.size() || edges[current_node].empty()) {
            (*this)[current_node] += 1;
            continue;
        }
        std::vector<std::pair<uint32_t, float>>& edgesFrom = edges[current_node];
        float total_weight_sum = 0;
        for (const std::pair<uint32_t, float>& edge : edgesFrom) {
            total_weight_sum += edge.second;
        }
        std::uniform_real_distribution<float> dice_weight {0.0f, total_weight_sum};
        float random_weight = dice_weight(prng);
        float weight_sum = 0;
        for (const std::pair<uint32_t, float>& edge : edgesFrom) {
            weight_sum += edge.second;
            if (weight_sum >= random_weight) {
                current_node = edge.first;
                break;
            }
        }
        (*this)[current_node] += 1;
    }

    if constexpr (/* DISABLED - generates too much output if run thousands of times */ false) {
        // example for generating random integers (upper bound included)
        std::uniform_int_distribution<uint32_t> dice {1, 6};
        for (uint32_t i=0; i<10; ++i)
            std::cout << dice(prng) << " ";
        std::cout << std::endl;

        // example for generating random floating point values (upper bound excluded)
        std::uniform_real_distribution<float> uniform_float {0.0f, 1.0f};
        for (uint32_t i=0; i<10; ++i)
            std::cout << uniform_float(prng) << " ";
        std::cout << std::endl;
    }
}

void RandomWalkGraph::write_histogram_pgm(const std::string& filename, uint32_t width, uint32_t height) const {
    if (width * height != size()) {
        throw std::runtime_error("Number of nodes in the graph does not match the number of pixels for the requested resolution.");
    }

    std::ofstream file;
    file.open(filename, std::ios::binary);  // Open in binary mode
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file.");
    }

    file << "P5\n" << width << " " << height << "\n255\n";


    std::vector<unsigned char> histogram = compute_normalized_histogram<unsigned char>(255);
    file.write(reinterpret_cast<const char*>(histogram.data()), histogram.size());
    file.close();
}
