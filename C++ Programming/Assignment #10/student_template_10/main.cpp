#include "submission/adjacency_list_graph.h"
#include "submission/random_walk_graph.h"

#include <cstdint>
#include <cstdlib>

#include <fstream>
#include <ios>
#include <iostream>
#include <array>
#include <cassert>
#include <cstring>

/*
Oguz Ata Cal    - 6661014  
Karahan Saritas - 6661689
*/


int main() {

    // how to work with existing code bases:
    // https://dilbert.com/strip/2018-11-13
    
    // create a random walk graph using secret.graph
    std::cout << "Deserializing graph..." << std::endl;
    RandomWalkGraph graph = RandomWalkGraph::deserialize("/home/karab/cpp_course/Sheet 10/student_template_10/secret.graph");
    std::cout << "Graph has " << graph.size() << " nodes." << std::endl;

    // TODO: task 10.3 a)

    // TODO: task 10.3 c)
    // Simulate 1000 random walks with 100000 steps each in your main function. (
    for (uint32_t i=0; i<1000; i++) {
        graph.simulate_random_walk(100000);
    }
    // write it to histogram.pgm
    graph.write_histogram_pgm("histogram.pgm", 270, 354);

    return EXIT_SUCCESS;
}
