#pragma once

#include <array>
#include <limits>
#include <vector>
#include <string>
#include <optional>
#include <functional>

class ShortestPaths {
public:
    /// a row in the adjacency matrix:
    /// a location and a vector of outgoing edges
    class Location {
    public:
        /// e.g. city name
        std::string name;
        /// position for computing the A* heuristic (not for Dijkstra!)
        float pos_x, pos_y;

        // use these to access distances
        std::optional<float>& operator[](size_t i) { return distances.at(i); }
        const std::optional<float>& operator[](size_t i) const { return distances.at(i); }
        std::optional<float>& at(size_t i) { return distances.at(i); }
        const std::optional<float>& at(size_t i) const { return distances.at(i); }

        void resize(size_t num_nodes) { distances.resize(num_nodes); }

    private:
        /// distances towards other nodes or no value if not connected
        std::vector<std::optional<float>> distances;
    };

public:
    ShortestPaths() = default;
    ShortestPaths(size_t num_nodes) { resize(num_nodes); }

    void resize(size_t num_nodes) {
        adjacency_matrix.resize(num_nodes);
        for (auto& row : adjacency_matrix)
            row.resize(num_nodes);
    }

    size_t size() const { return adjacency_matrix.size(); }

    // use these to access nodes and edges
    Location& operator[](size_t i) { return adjacency_matrix.at(i); }
    const Location& operator[](size_t i) const { return adjacency_matrix.at(i); }
    Location& at(size_t i) { return adjacency_matrix.at(i); }
    const Location& at(size_t i) const { return adjacency_matrix.at(i); }

    size_t getNodeIdByName(const std::string& name) const;

    std::vector<size_t> compute_shortest_path(const std::string& from, const std::string& to) const {
        return compute_shortest_path(getNodeIdByName(from), getNodeIdByName(to));
    }

    std::vector<size_t> compute_shortest_path(size_t from, size_t to) const;

private:
    // adjacency_matrix - contains all locations and a full list of distances between all nodes
    std::vector<Location> adjacency_matrix;
};
