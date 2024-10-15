#pragma once

#include <cstdint>
#include <stdexcept>
#include <fstream>
#include <type_traits>
#include <vector>
#include <string>
#include <optional>

#include <iostream>

namespace detail {

    // This class contains all parts of AdjacencyListGraph that do not depend on the template parameter.
    // This allows us to write the implementation of these functions without template declaration.
    // Also, the compiler will not have to instantiate a copy of the function for each template type.
    // This reduces the compile time and the size of the compiled code, which should benefit the performance.
    class AdjacencyListGraphBase {
    protected:
        // these are the edges in the graph - each outbound edge comes with a weight
        std::vector<std::vector<std::pair<uint32_t, float>>> edges;

        virtual size_t get_num_nodes() const = 0; // call the derived class to get the number of nodes

    public:
        void add_edge(uint32_t from, uint32_t to, float weight=1.0f);
        void remove_edge(uint32_t from, uint32_t to);
        std::optional<float> get_edge(uint32_t from, uint32_t to) const;

        const std::vector<std::pair<uint32_t, float>>& get_edges_starting_at(uint32_t node) const;
    };
}

template <typename TNode, typename TDerived=void> // using CRTP to create instances of the derived type during deserialization
class AdjacencyListGraph : public detail::AdjacencyListGraphBase, private std::vector<TNode> {
private:
    size_t get_num_nodes() const override { return size(); }; // pass the number of nodes to the AdjacencyListGraphBase
public:
    using DerivedType = std::conditional_t<std::is_void_v<TDerived>, AdjacencyListGraph<TNode>, TDerived>;

    // While we derive from std::vector privately, we can still make some of its functions public with "using" without having to write a wrapper function.
    // We only allow functions that add nodes - they cannot be removed later. This way, we do not have to worry about updating the edges of the graph.
    using std::vector<TNode>::push_back;
    using std::vector<TNode>::size;
    using std::vector<TNode>::reserve;
    using std::vector<TNode>::operator[];
    using std::vector<TNode>::at;
    using std::vector<TNode>::begin;
    using std::vector<TNode>::end;
    using std::vector<TNode>::cbegin;
    using std::vector<TNode>::cend;

    /// resize to the given number of nodes (nodes cannot be removed)
    void initialize_nodes(uint32_t num_nodes) {
        if (num_nodes < size())
            throw std::logic_error("The graph already contains more nodes than requested.");
        this->resize(num_nodes);
    }

    /// serialize function for node types that are trivially copyable
    std::enable_if_t<std::is_trivially_copyable_v<TNode>, void> serialize(const std::string& filename) const
    {
        std::ofstream file(filename, std::ofstream::out|std::ofstream::trunc|std::ofstream::binary);
        // 1. write the number of nodes
        const size_t num_nodes = size();
        file.write(reinterpret_cast<const char*>(&num_nodes), sizeof(num_nodes));
        // 2. write the node data
        file.write(reinterpret_cast<const char*>(this->data()), static_cast<std::streamsize>(sizeof(TNode)*num_nodes));
        // 3. write the edge data
        // 3.1. loop over the nodes and write the edges per node
        for (size_t i=0; i<num_nodes; ++i) {
            const auto& edges = get_edges_starting_at(static_cast<uint32_t>(i));
            // 3.2 write the number of edges for this node
            const size_t num_edges = edges.size();
            file.write(reinterpret_cast<const char*>(&num_edges), sizeof(num_edges));
            // 3.3 loop over the edges
            for (const auto& [to, weight] : edges) {
                // make sure the types match our expectations (we don't want to just write arbitrary binary data)
                static_assert(std::is_same_v<decltype(to), const uint32_t>, "edge target needs to be an uint32_t");
                static_assert(std::is_same_v<decltype(weight), const float>, "edge weight needs to be a float");

                // 3.4 store the outgoing link and its weight
                file.write(reinterpret_cast<const char*>(&to), sizeof(to));
                file.write(reinterpret_cast<const char*>(&weight), sizeof(weight));
            }
        }

        file.close();
    }
    /// deserialize function for node types that are trivially copyable
    /// returns derived type (if any) provided via CRTP
    static std::enable_if_t<std::is_trivially_copyable_v<TNode>, DerivedType>
    deserialize(const std::string& filename)
    {
        std::ifstream file;
        file.open(filename);

        if (!file.is_open()) {
            throw std::runtime_error("Could not open file.");  // check if the file is open
        }
        size_t num_nodes;  
        file.read(reinterpret_cast<char*>(&num_nodes), sizeof(num_nodes)); // read the number of nodes

        DerivedType graph;
        graph.resize(num_nodes);
        file.read(reinterpret_cast<char*>(graph.data()), static_cast<std::streamsize>(sizeof(TNode)*num_nodes));  // read the node data

        for (size_t i=0; i<num_nodes; i++) {
            size_t num_edges;
            file.read(reinterpret_cast<char*>(&num_edges), sizeof(num_edges)); // get the number of edges for this node

            for (size_t j=0; j<num_edges; j++) {
                uint32_t neighbor;
                float w;
                file.read(reinterpret_cast<char*>(&neighbor), sizeof(neighbor));  // read the neighbor 
                file.read(reinterpret_cast<char*>(&w), sizeof(w));                // read the weight
                graph.add_edge(static_cast<uint32_t>(i), neighbor, w);           // add the edge to the graph
            }
        }

        file.close();
        return graph;
    }
};
