#include "graph.h"
#include <cmath>
#include <random>
#include <iterator>
#include <algorithm>

namespace OP {
    Graph::Graph (std::string file_name, std::string input_dir) : file_name{file_name}, input_dir{input_dir} {
        std::cout << file_name << "\n";

        read_graph();
        read_optimal_solution();
    }

    void Graph::read_optimal_solution(){
        std::string opt_file_name = input_dir + file_name + ".opt.tour";
        std::ifstream opt_file(opt_file_name);
        if (!opt_file){
            std::cout << "optimal solution is not provided \n";
            return;
        }
        std::string line;
        bool READ_Point = 0;
        std::uint32_t node;
        while(!opt_file.eof()) {
            getline(opt_file, line);
//            std::cout << line << "\n";
            if (line == "EOF" || line == "-1") {
                break;
            }
            if (READ_Point){
                std::stringstream stream(line);
                while (stream >> node) {
                    opt_tour.push_back(node - 1);
//                    std::cout << node << ", ";
                }
//                std::cout << "\n";
            }
            if (line == "TOUR_SECTION"){
                READ_Point = 1;
            }
        }
        opt_obj = 0;
        optimal_solution = std::vector<std::vector<bool> >(n_nodes, std::vector<bool>(n_nodes, 0));
        for (auto i = 0u; i < opt_tour.size() - 1; ++i){
            opt_obj += values[opt_tour[i]];
            optimal_solution[opt_tour[i]][opt_tour[i+1]] = 1;
        }
        std::cout << "optimal objective value is " << opt_obj << "\n";
    }

    void Graph::read_graph(){
        std::string input_file = input_dir  + file_name + ".op";
        std::ifstream file(input_file);
        std::string line;
        double xc, yc, val;
        std::vector<double> x_coord, y_coord;

        getline(file, line);
        std::stringstream stream(line);
        stream >> Tmax >> P;
//        std::cout << line << std::endl;

        n_nodes = 0u;
        while(!file.eof()) {
            getline(file, line);
//            std::cout << line << std::endl;
            if (line == "EOF" || line == "-1") {
                break;
            }
            std::stringstream stream(line);
            stream >> xc >> yc >> val;
            x_coord.push_back(xc);
            y_coord.push_back(yc);
            values.push_back(val);
//            std::cout << x_coord[n_nodes] << ", " << y_coord[n_nodes] << values[n_nodes] << std::endl;
            n_nodes++;
        }
        std::cout << "number of nodes is " << n_nodes << std::endl;
        costs = std::vector<std::vector<double> >(n_nodes, std::vector<double>(n_nodes, 999999999));
        if (dist_type == 0){
            //Euclidean distance
            for(auto i = 0u; i < n_nodes; ++i) {
                for(auto j = i+1; j < n_nodes; ++j) {
                    costs[i][j] = std::sqrt(std::pow(x_coord[i] - x_coord[j], 2.0) + std::pow(y_coord[i] - y_coord[j], 2.0));
                    costs[j][i] = costs[i][j];
                }
            }
        } else{
            //Geographic distance
            for (auto i = 2u; i < n_nodes; ++i){
                values[i] = log2(values[i]+1);
            }
            for (auto i = 0u; i < n_nodes; ++i){
                x_coord[i] = 3.141592*((int)x_coord[i] + 5.0*(x_coord[i]-(int)x_coord[i])/3.0) / 180.0;
                y_coord[i] = 3.141592*((int)y_coord[i] + 5.0*(y_coord[i]-(int)y_coord[i])/3.0) / 180.0;
            }
            for(auto i = 0u; i < n_nodes; ++i) {
                for(auto j = i+1; j < n_nodes; ++j) {
                    costs[i][j] = (6378.388 * acos(0.5*((1.0+cos(y_coord[i]-y_coord[j]))*cos(x_coord[i]-x_coord[j])-(1.0-cos(y_coord[i]-y_coord[j]))*cos(x_coord[i]+x_coord[j]))) + 1.0);
                    costs[j][i] = costs[i][j];
                }
            }
        }


        value_per_cost = std::vector<std::vector<double> >(n_nodes, std::vector<double>(n_nodes, 0));
        max_out_vpc = std::vector<double>(n_nodes, 0);
        max_in_vpc = std::vector<double>(n_nodes, 0);
        for(auto i = 0u; i < n_nodes; ++i) {
            for(auto j = 0u; j < n_nodes; ++j) {
                value_per_cost[i][j] = values[j]/costs[i][j];
                if (max_out_vpc[i] < value_per_cost[i][j]){
                    max_out_vpc[i] =  value_per_cost[i][j];
                }
                if (max_in_vpc[j] < value_per_cost[i][j]){
                    max_in_vpc[j] = value_per_cost[i][j];
                }
            }
        }


    }

}