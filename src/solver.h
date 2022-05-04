#ifndef SOLVER_H
#define SOLVER_H

#include "graph.h"
#include "reduce.h"
#include <iostream>
#include <vector>
#include <cstring>
//#include <omp.h>

// Magic tricks to have CPLEX behave well:
//#ifndef IL_STD
//#define IL_STD
//#endif
//#include <cstring>
//ILOSTLBEGIN
// End magic tricks

namespace OP {
  class Solver {
    // The graph on which we are solving the TSP.
    const Graph& g;
    const Reduce& r;
    const double cutoff;
    double objVal;
    std::vector<std::uint32_t> opt_sol;

    std::vector<double> time_series_data;

    const double alpha = 1;
    const double beta = 1;
    const double rho = 0.05;
    const double delta = 0.5;
//    const double pbest = 0.005;

    std::uint32_t C = 1;
    std::uint32_t nPop = 0;
    std::uint32_t restart = 100u;
    const std::uint32_t nCycle = 10000;

//    std::uint32_t n_thread = 4;
    
    // Prints the integer solution obtained by CPLEX to stdout.
//    void print_solution(const IloCplex& cplex, const IloArray<IloNumVarArray>& x);
    
  public:
    
    // Builds a solver for graph g.
    explicit Solver(const Graph& g, const Reduce& r, const float cutoff) : g{g}, r{r}, cutoff{cutoff} {}
    
    // Solves the TSP with CPLEX and prints the result.
//    void solve_op_cplex();

//    void solve_tsp_concorde();

    void solver_op_ACO();

    void solver_op_MMAS();

    double get_objective_value() const { return objVal; }

    std::vector<std::uint32_t> get_optimal_solution() const { return opt_sol; }
    std::vector<double> get_time_series_data() const { return time_series_data; }

  };
}

#endif
