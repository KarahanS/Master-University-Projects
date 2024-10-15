#pragma once

#include "adjacency_list_graph.h"

#include <iterator>
#include <random>
#include <algorithm>

class RandomWalkGraph : public AdjacencyListGraph<uint32_t, RandomWalkGraph> {
private:
    std::default_random_engine prng;

public:
    void simulate_random_walk(uint32_t num_steps);

    template<typename T>
    std::vector<T> compute_normalized_histogram(T max_value) const {
        std::vector<T> hist;
        hist.reserve(size());

        const uint32_t max_visited = *std::max_element(cbegin(), cend());
        const float normalization = max_visited > 0 ? static_cast<float>(max_value)/static_cast<float>(max_visited) : 0.0f;
        std::transform(cbegin(), cend(), std::back_inserter(hist), [=](int count) -> T { return static_cast<T>(static_cast<float>(count)*normalization); });

        return hist;
    }

    /// arrange the nodes in the graph as a 2d grid and export the histogram as a greyscale image
    void write_histogram_pgm(const std::string& filename, uint32_t width, uint32_t height) const;
};
