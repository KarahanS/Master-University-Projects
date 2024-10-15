#include "submission/shortest_paths.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

#include <fstream>
#include <ios>
#include <iostream>
#include <array>
#include <cassert>
#include <string>
#include <stdexcept>

int main() {

    ShortestPaths graph;

    // load cities
    {
        // data taken from https://simplemaps.com/data/de-cities
        std::ifstream city_data("../de.csv", std::ifstream::in);
        char buffer[1024];

        // determine number of lines in the file
        size_t num_cities = 0;
        // skip header
        city_data.getline(buffer, std::size(buffer));
        while (!city_data.eof()) {
            city_data.getline(buffer, std::size(buffer));
            if (buffer[0]) // ignore empty line at the end
                ++num_cities;
        }
        graph.resize(num_cities);
        // reset flags
        city_data.clear();
        // rewind file
        city_data.seekg(0, std::ifstream::beg);

        // skip header
        city_data.getline(buffer, std::size(buffer));
        for (size_t i=0; i<graph.size(); ++i) {
            if (!city_data.getline(buffer, std::size(buffer)))
                break;

            std::string line {buffer, std::size(buffer)-1};

            size_t first_comma = line.find_first_of(',');
            if (first_comma == std::string::npos)
                break;
            size_t second_comma = line.find_first_of(',', first_comma+1);
            if (second_comma == std::string::npos)
                break;
            size_t third_comma = line.find_first_of(',', second_comma+1);
            if (third_comma == std::string::npos)
                break;

            graph[i].name = line.substr(0, first_comma);
            const float lat = std::stof(line.substr(first_comma+1, second_comma-first_comma));
            const float lon = std::stof(line.substr(second_comma+1, third_comma-second_comma));

            const float theta = (90.0f-lat)*float(M_PI/180.0);
            const float phi = lon*float(M_PI/180.0);

            // approximate radius of the earth in kilometers
            constexpr const float r = 6378.137f;
            // reference location for (0, 0)
            constexpr float ref_lat = 48.5200f;
            constexpr float ref_lon = 9.0556f;

            constexpr float ref_theta = (90.0f-ref_lat)*float(M_PI/180.0);
            constexpr float ref_phi = ref_lon*float(M_PI/180.0);

            // produce some approximate values that produce plausible distances in kilometers
            graph[i].pos_x = (phi-ref_phi) * std::cos(ref_theta)*r;
            graph[i].pos_y = (theta-ref_theta) * r;
        }
        city_data.close();
    }

    // add edges between k closest cities
    {
        const size_t nearest_k = 5;
        if (nearest_k > graph.size())
            throw std::runtime_error("not enough data for knn");

        auto distance = [](const ShortestPaths::Location& a, const ShortestPaths::Location& b) -> float { return std::sqrt((a.pos_x-b.pos_x)*(a.pos_x-b.pos_x)+(a.pos_y-b.pos_y)*(a.pos_y-b.pos_y)); };
        for (size_t i=0; i<graph.size(); ++i) {
            std::vector<std::tuple<float, size_t>> nearest_neighbors;
            for (size_t j=0; j<graph.size(); ++j) {
                if (j == i)
                    continue;
                nearest_neighbors.emplace_back(distance(graph.at(i), graph.at(j)), j);
            }
            std::nth_element(nearest_neighbors.begin(), nearest_neighbors.begin()+nearest_k, nearest_neighbors.end(), [](const auto& a, const auto& b) -> bool { return std::get<0>(a) < std::get<0>(b); });

            // add edges between nearest k elements in both directions
            for (size_t j=0; j<nearest_k; ++j) {
                graph[i][std::get<1>(nearest_neighbors.at(j))] = std::get<0>(nearest_neighbors.at(j));
                graph[std::get<1>(nearest_neighbors.at(j))][i] = std::get<0>(nearest_neighbors.at(j));
            }
        }
    }

    // TODO: this is where you can test your code
    // Stuttgart - Ulm should be about 90 km
    // Berlin - Munich should be about 545 km

    auto path = graph.compute_shortest_path("Stuttgart", "Ulm");
    //auto path = graph.compute_shortest_path("Berlin", "Munich");
    if (!path.empty()) {
        std::cout << "Shortest path: ";
        for (size_t i=0; i<path.size()-1; ++i) {
            std::cout << graph.at(path[i]).name << " - ";
        }

        std::cout << graph.at(path.back()).name << std::endl;
    }
    else
        std::cout << "No route found!" << std::endl;

    return EXIT_SUCCESS;
}

