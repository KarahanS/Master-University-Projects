#include "adjacency_list_graph.h"

#include <stdexcept>
#include <algorithm>
#include <string>



void detail::AdjacencyListGraphBase::add_edge(uint32_t from, uint32_t to, float weight)
{
    size_t numNodes = get_num_nodes();
    if (from >= numNodes || to >= numNodes)
        throw std::runtime_error("Node index out of range.");
    
    bool edgeExists = false;
    // extend the outer vector of edges before you can insert the edge
    if (edges.size() <= from) edges.resize(numNodes);
    std::vector<std::pair<uint32_t, float>>& edgesFrom = edges[from];
    for (const std::pair<uint32_t, float>& edge : edgesFrom) {
        if (edge.first == to) {
            edgeExists = true;
            break;
        }
    }

    if (edgeExists)
        throw std::runtime_error("Edge already exists.");

    edgesFrom.push_back(std::make_pair(to, weight));
}

void detail::AdjacencyListGraphBase::remove_edge(uint32_t from, uint32_t to)
{
    size_t numNodes = get_num_nodes();
    if (from >= numNodes || to >= numNodes || edges.size() <= from)
        throw std::runtime_error("Node index out of range.");

    std::vector<std::pair<uint32_t, float>>& edgesFrom = edges[from];
    for (auto it = edgesFrom.begin(); it != edgesFrom.end(); it++) {
        if (it->first == to) {
            edgesFrom.erase(it);
            return; 
        }
    }

    // If the edge wasn't found, throw an exception
    throw std::runtime_error("Edge does not exist.");

}

std::optional<float> detail::AdjacencyListGraphBase::get_edge(uint32_t from, uint32_t to) const
{
    
    size_t numNodes = get_num_nodes();
    if (from >= numNodes || to >= numNodes)
        throw std::runtime_error("Node index out of range."); 

    if (edges.size() <= from)  return std::optional<float>();
    const std::vector<std::pair<uint32_t, float>>& edgesFrom = edges[from];
    for (const std::pair<uint32_t, float>& edge : edgesFrom) {
        if (edge.first == to) {
            return edge.second;
        }
    }
    return std::optional<float>();

}

const std::vector<std::pair<uint32_t, float>>& detail::AdjacencyListGraphBase::get_edges_starting_at(uint32_t node) const {
    static const std::vector<std::pair<uint32_t, float>> empty_edges;
    if (node >= get_num_nodes()) return empty_edges;
    if (node >= edges.size()) return empty_edges;
    
    return edges[node];
}