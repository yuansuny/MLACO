#ifndef GRAPH_H
#define GRAPH_H

#include <vector>
#include <iostream>
#include <string>
#include <algorithm>
#include <sstream>
#include <fstream>

namespace OP {
  class Graph {
    // Number of nodes in the graph.
    // Nodes go from 0 to n_nodes - 1.
    std::uint32_t n_nodes;
    const std::string file_name;
    const std::string input_dir;
    const int dist_type = 0; // 0: Euclidean distance; 1: Geographic distance
    std::vector<std::vector<double> > costs;
    std::vector<std::vector<double> > value_per_cost;
    std::vector<double> max_out_vpc, max_in_vpc;
    std::vector<double> values;
    std::vector<int> depot;
    double Tmax;
    std::uint32_t P;
    std::vector<std::vector<bool> > optimal_solution;
    std::vector<std::uint32_t> opt_tour;
    double opt_obj;

    void read_graph();
//    void read_tsp_graph();
    void read_optimal_solution();

    bool checkKeyword(std::string keyword, std::string value);
    std::string trim(std::string s);

    
  public:
    
    // Created a new (random) graph with n_nodes nodes
    explicit Graph(std::string file_name, std::string input_dir);

    // Size of the graph (i.e. number of nodes).
    std::uint32_t size() const { return n_nodes; }

    double get_Tmax() const { return Tmax; }
    
    // Cost of arc (i,j).
    double cost(std::uint32_t i, std::uint32_t j) const { return costs[i][j]; }

    double get_value_per_cost(std::uint32_t i, std::uint32_t j) const { return value_per_cost[i][j]; }

    double get_value(std::uint32_t i) const { return values[i]; }

    double get_max_out_vpc(std::uint32_t i) const { return max_out_vpc[i]; }

    double get_max_in_vpc(std::uint32_t i) const { return max_in_vpc[i]; }

    // Optimal solution of edge x[i][j]
    bool get_optimal_value(std::uint32_t i, std::uint32_t j) const { return optimal_solution[i][j]; }

    std::string get_file_name() const { return file_name; }
    
    // Prints out the cost matrix of the graph.
    friend std::ostream& operator<<(std::ostream& out, const Graph& g);
  };
}

#endif
