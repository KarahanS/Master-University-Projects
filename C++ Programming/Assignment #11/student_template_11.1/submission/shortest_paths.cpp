#include "shortest_paths.h"
#include <algorithm>
#include <utility>
#include <iostream>
#include <cmath>
#include <queue>

size_t ShortestPaths::getNodeIdByName(const std::string& name) const {
    // NOTE: if you like, you can make this more efficient by caching the mapping in a mutable hash map that gets reset when calling non-const functions
    const auto it = std::find_if(adjacency_matrix.begin(), adjacency_matrix.end(), [=](const Location& row) -> bool { return row.name == name; });
    if (it == adjacency_matrix.end())
        throw std::runtime_error("Location "+name+" not found");
    return static_cast<size_t>(std::distance(adjacency_matrix.begin(), it));
}

/*
std::vector<size_t> ShortestPaths::compute_shortest_path(size_t from, size_t to) const
{
    /// your result path
    std::vector<size_t> result;
    /// increment this for every node that you pop from the queue
    size_t num_visited = 0;

    std::priority_queue<std::pair<float, size_t>> queue;  // max-heap
    
    std::vector<float> distance(adjacency_matrix.size(), std::numeric_limits<float>::max());
    std::vector<bool> visited(adjacency_matrix.size(), false);

    std::vector<size_t> predecessor;  // to keep track of the path
    predecessor.resize(adjacency_matrix.size());
    
    queue.push(std::make_pair(0, from));
    distance[from] = 0;
    predecessor[from] = from;
    num_visited--;  // to avoid counting the first node

    while(!queue.empty()) {
        std::pair<float, size_t> current = queue.top();
        size_t current_node = current.second;
        queue.pop(); // pop the front element
        if (visited[current_node]) continue;

        num_visited++;
        visited[current_node] = true;
        if (visited[to]) break;
        
        for(size_t i=0; i<adjacency_matrix.size(); i++) { // iterate over all nodes
            if(adjacency_matrix[current_node][i] && !visited[i]) {  // if there is an edge and the node is not visited
                std::optional<float> temp = adjacency_matrix[current_node][i];
                if (!temp) continue; // this won't happen but just in case
                float dist = temp.value();  
                float new_dist = distance[current_node] + dist;
                if (new_dist < distance[i]) {  
                    distance[i] = new_dist;  // update the distance
                    queue.push(std::make_pair(-distance[i], i));  // multiply by -1 to make it a min-heap
                    predecessor[i] = current_node;
                }
            }
        }
    }

    result.push_back(to);
    while (predecessor[to] != to) {
        result.push_back(predecessor[to]);
        to = predecessor[to];
    }
    
    std::reverse(result.begin(), result.end());
    std::cout << "Nodes visited: " << num_visited << std::endl;
    return result;
}
*/


std::vector<size_t> ShortestPaths::compute_shortest_path(size_t from, size_t to) const {
    /// your result path
    std::vector<size_t> result;
    /// increment this for every node that you pop from the queue
    size_t num_visited = 0;

    // Max-heap based on f = g + h
    std::priority_queue<std::pair<float, size_t>> queue;

    std::vector<float> g_score(adjacency_matrix.size(), std::numeric_limits<float>::max());  // g(n): cost from start to current node
    std::vector<bool> visited(adjacency_matrix.size(), false);
    std::vector<size_t> predecessor(adjacency_matrix.size());

    g_score[from] = 0;
    predecessor[from] = from;
    
    // Push the start node with heuristic only (f = g + h, g = 0 for the start node)
    float dx = adjacency_matrix[from].pos_x - adjacency_matrix[to].pos_x;
    float dy = adjacency_matrix[from].pos_y - adjacency_matrix[to].pos_y;
    float h =  std::sqrt(dx * dx + dy * dy);
 
    queue.push(std::make_pair(-h, from));
    num_visited--;  // to avoid counting the first node

    while (!queue.empty()) {
        std::pair<float, size_t> current = queue.top();
        size_t current_node = current.second;
        queue.pop();
        if (visited[current_node]) continue;
        num_visited++;

        visited[current_node] = true;
        if (visited[to]) break;

        for (size_t i = 0; i < adjacency_matrix.size(); i++) {
            if (adjacency_matrix[current_node][i] && !visited[i]) { // if there is an edge and node is not visited
                std::optional<float> temp = adjacency_matrix[current_node][i];
               
                float dist = temp.value();  // distance between current_node and i
                float temp_g_score = g_score[current_node] + dist;

                if (temp_g_score < g_score[i]) {
                    g_score[i] = temp_g_score;
                    dx = adjacency_matrix[i].pos_x - adjacency_matrix[to].pos_x;
                    dy = adjacency_matrix[i].pos_y - adjacency_matrix[to].pos_y;
                    float h_score =  std::sqrt(dx * dx + dy * dy);
                    float f_score = g_score[i] + h_score;  // f(n) = g(n) + h(n)

                    queue.push(std::make_pair(-f_score, i));  // Push with updated f(n)
                    predecessor[i] = current_node;  // Keep track of the path
                }
            }
        }
    }

    if (!visited[to]) return result; // return empty vector

    result.push_back(to);
    while (predecessor[to] != to) {
        result.push_back(predecessor[to]);
        to = predecessor[to];
    }
    
    std::reverse(result.begin(), result.end());
    std::cout << "Nodes visited: " << num_visited << std::endl;
    return result;
}
