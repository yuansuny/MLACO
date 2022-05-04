#include "solver.h"
#include <cmath>
#include <limits>

//extern "C" {
//#include <concorde.h>
//}

namespace OP {
    void Solver::solver_op_MMAS(){

        std::cout << "solving OP using MMAS" << std::endl;

        const auto n = g.size();

        nPop = C * n;

        std::vector<std::vector<double>> eta = std::vector<std::vector<double>>(n, std::vector<double>(n, 1.0));
        std::vector<std::vector<double>> tau = std::vector<std::vector<double>>(n, std::vector<double>(n, 1.0));
        std::vector<std::vector<std::uint32_t>> pop(nPop, std::vector<std::uint32_t>(n));
        std::vector<double> obj(nPop);

        time_series_data = std::vector<double>(nCycle);

        //initialize eta
        for (auto k = 2u; k < n; ++k){
            eta[0u][k] = r.get_predicted_value(0u,k) * g.get_value(k)/g.cost(0u,k);
        }
        for (auto j = 2u; j < n; ++j){
            for (auto k = 2u; k < n; ++k){
                if (j != k){
                    eta[j][k] = r.get_predicted_value(j,k) * g.get_value(k)/g.cost(j,k);
//                    eta[j][k] = g.get_value(k)/g.cost(j,k);
//                    tau[j][k] = r.get_predicted_value(j,k);
                }
            }
        }

        objVal = 0;
        double objVal_iter, objVal_temp = 0.0;
        std::uint32_t best_idx = 0u;
        std::uint32_t count = 0u;

        std::uint32_t start_v = 0u;
        std::uint32_t end_v = 1u;
        std::uint32_t v1, v2;
        double tau_max, tau_min;

        // compute the initial candidates that can be visited from start vertex
        std::vector<double> t_max(n);
        std::vector<std::uint32_t> ini_candidates;
        for (auto k = 0u; k < n; ++k){
            t_max[k] = g.get_Tmax() - g.cost(k, end_v);
            if (k != start_v && k != end_v && g.cost(start_v, k) <= t_max[k]){
                ini_candidates.push_back(k);
            }
        }

        std::random_device rd;
        std::mt19937 gen(rd());
        srand (time(NULL));

        std::vector<std::uint32_t> candidates(n);
        std::vector<double> prob(n);
        std::uint32_t curr_v, next_v;
        double curr_t, sum_prob, r;
        std::uint32_t nb_candidates, nb, idx;


        // evolve nCycle number of generations
        for (auto i = 0u; i < nCycle; ++i){

            objVal_iter = 0.0;

            // generate each individual sample
            for (auto j = 0u; j < nPop; ++j){
                pop[j][0] = start_v;
                curr_v = start_v;
                curr_t = 0.0;
                obj[j] = 0.0;
                nb = 1u;

                std::copy(ini_candidates.begin(), ini_candidates.end(), candidates.begin());
                nb_candidates = ini_candidates.size();


                while(nb_candidates != 0){
                    // compute probability of visiting each candidate vertex from the current vertex
                    sum_prob = 0.0;
                    for (auto k = 0u; k < nb_candidates; ++k){
                        prob[k] = pow(tau[curr_v][candidates[k]], alpha) * pow(eta[curr_v][candidates[k]], beta);
                        sum_prob += prob[k];
                    }
                    // select the next vertex to visit based on the probability
                    if (sum_prob == 0.0){
                        next_v = candidates[rand() % nb_candidates];
                    } else{
                        r = (double) rand() / RAND_MAX;
                        for (auto k = 0u; k < nb_candidates; ++k){
                            prob[k] /= sum_prob;
                            if (r <= prob[k]){
                                next_v = candidates[k];
                                break;
                            }else{
                                r -= prob[k];
                            }
                        }
                    }
                    pop[j][nb++] = next_v;
                    curr_t += g.cost(curr_v, next_v);
                    obj[j] += g.get_value(next_v);
                    curr_v = next_v;
                    // update the candidate vertex that can be visited in the next step
                    idx = 0u;
                    for (auto k = 0u; k < nb_candidates; ++k){
                        if (candidates[k] != curr_v && curr_t + g.cost(curr_v, candidates[k]) <= t_max[candidates[k]]){
                            candidates[idx++] = candidates[k];
                        }
                    }
                    nb_candidates = idx;
                }
                pop[j][nb++] = end_v;

                if (objVal_iter < obj[j]){
                    objVal_iter = obj[j];
                    best_idx = j;
                }
            }

            if (objVal_temp < objVal_iter){
                objVal_temp = objVal_iter;
                count = 0u;
            }else{
                count++;
            }

            if (objVal < objVal_temp){
                objVal = objVal_temp;
            }

            tau_max = 1/rho/objVal;
            tau_min = tau_max/2/n;

            // re-scale tau after the first iteration
            if (i == 0){
                for (auto j = 0u; j < n; ++j){
                    for (auto k = 0u; k < n; ++k){
                        tau[j][k] = tau[j][k]*(tau_max - tau_min) + tau_min;
                    }
                }
            }
            // update tau
            idx = 0u;
            while(pop[best_idx][idx] != end_v){
                v1 = pop[best_idx][idx];
                v2 = pop[best_idx][idx+1];
                tau[v1][v2] = tau[v1][v2] + 1/objVal/(1-rho);
                idx++;
            }
            for (auto j = 0u; j < n; ++j){
                for (auto k = 0u; k < n; ++k){
                    tau[j][k] = (1 - rho) * tau[j][k];
                    if (tau[j][k] < tau_min){
                        tau[j][k] = tau_min;
                    }else if (tau[j][k] > tau_max){
                        tau[j][k] = tau_max;
                    }
                    if (count == restart){
                        tau[j][k] = tau[j][k] + delta * (tau_max - tau[j][k]);
                    }
                }
            }
            if (count == restart){
                count = 0u;
                objVal_temp = 0.0;
                std::cout << "restart" << std::endl;
            }
            time_series_data[i] = objVal;
            std::cout << "Iteration: " << i+1 << ", " << "best objective value: " << objVal << std::endl;
        }
    }


    void Solver::solver_op_ACO(){

        std::cout << "solving OP using ACO" << std::endl;

        const auto n = g.size();

        nPop = C * n;

        std::vector<std::vector<double>> eta = std::vector<std::vector<double>>(n, std::vector<double>(n, 1.0));
        std::vector<std::vector<double>> tau = std::vector<std::vector<double>>(n, std::vector<double>(n, 1.0));
        std::vector<std::vector<double>> delta_tau(n, std::vector<double>(n));
        std::vector<std::vector<std::uint32_t>> pop(nPop, std::vector<std::uint32_t>(n));
        std::vector<double> obj(nPop);

        time_series_data = std::vector<double>(nCycle);

        //initialize eta
        for (auto k = 1u; k < n; ++k){
            eta[0u][k] = g.get_value(k)/g.cost(0u,k);
        }
        for (auto j = 2u; j < n; ++j){
            for (auto k = 2u; k < n; ++k){
                if (j != k){
//                    eta[j][k] = r.get_predicted_value(j,k);
                    eta[j][k] = g.get_value(k)/g.cost(j,k);
                    tau[j][k] = r.get_predicted_value(j,k);
//                    if (tau[j][k] < 0.00001){
//                       tau[j][k] = 0.00001;
//                    }
                }
            }
        }

        objVal = 0;

        std::uint32_t start_v = 0u;
        std::uint32_t end_v = 1u;

        // compute the initial candidates that can be visited from start vertex
        std::vector<double> t_max(n);
        std::vector<std::uint32_t> ini_candidates;
        for (auto k = 0u; k < n; ++k){
            t_max[k] = g.get_Tmax() - g.cost(k, end_v);
            if (k != start_v && k != end_v && g.cost(start_v, k) <= t_max[k]){
                ini_candidates.push_back(k);
            }
        }

        std::random_device rd;
        std::mt19937 gen(rd());
        srand (time(NULL));

        std::vector<std::uint32_t> candidates(n);
        std::vector<double> prob(n);
        std::uint32_t curr_v, next_v;
        double curr_t, sum_prob, r;
        std::uint32_t nb_candidates, nb, idx;

        // evolve nCycle number of generations
        for (auto i = 0u; i < nCycle; ++i){

            for (auto j = 0u; j < n; ++j){
                for (auto k = 0u; k < n; ++k){
                    delta_tau[j][k] = 0.0;
                }
            }

            // generate each individual sample
            for (auto j = 0u; j < nPop; ++j){
                pop[j][0] = start_v;
                curr_v = start_v;
                curr_t = 0.0;
                obj[j] = 0.0;
                nb = 1u;

                std::copy(ini_candidates.begin(), ini_candidates.end(), candidates.begin());
                nb_candidates = ini_candidates.size();


                while(nb_candidates != 0){
                    // compute probability of visiting each candidate vertex from the current vertex
                    sum_prob = 0.0;
                    for (auto k = 0u; k < nb_candidates; ++k){
                        prob[k] = pow(tau[curr_v][candidates[k]], alpha) * pow(eta[curr_v][candidates[k]], beta);
                        sum_prob += prob[k];
                    }
                    // select the next vertex to visit based on the probability
                    if (sum_prob == 0.0){
                        next_v = candidates[rand() % nb_candidates];
                    } else{
                        r = (double) rand() / RAND_MAX;
                        for (auto k = 0u; k < nb_candidates; ++k){
                            prob[k] /= sum_prob;
                            if (r <= prob[k]){
                                next_v = candidates[k];
                                break;
                            }else{
                                r -= prob[k];
                            }
                        }
                    }
                    pop[j][nb++] = next_v;
                    curr_t += g.cost(curr_v, next_v);
                    obj[j] += g.get_value(next_v);
                    curr_v = next_v;
                    // update the candidate vertex that can be visited in the next step
                    idx = 0u;
                    for (auto k = 0u; k < nb_candidates; ++k){
                        if (candidates[k] != curr_v && curr_t + g.cost(curr_v, candidates[k]) <= t_max[candidates[k]]){
                            candidates[idx++] = candidates[k];
                        }
                    }
                    nb_candidates = idx;
                }
                pop[j][nb++] = end_v;

                if (objVal < obj[j]){
                    objVal = obj[j];
                }

                // update delta_tau_local
                for (auto k = 0u; k < nb-1; ++k){
//                    delta_tau[pop[j][k]][pop[j][k+1]] += obj[j]/C;
                    delta_tau[pop[j][k]][pop[j][k+1]] += obj[j]/obj[0]/C;
                }
            }

            // update tau
            for (auto j = 0u; j < n; ++j){
                for (auto k = 0u; k < n; ++k){
                    tau[j][k] = (1 - rho) * tau[j][k] + delta_tau[j][k];
                }
            }
            time_series_data[i] = objVal;
            std::cout << "Iteration: " << i+1 << ", " << "best objective value: " << objVal << std::endl;
        }
    }

//    void Solver::solve_op_cplex() {
//        const auto n = g.size();
//
//        // CPLEX environment. Takes care of everything, including memory management for CPLEX objects.
//        IloEnv env;
//
//        // CPLEX model. We put variables and constraints in it!
//        IloModel model(env);
//
//        // Model:
//        //
//        // BINARY VARIABLE x[i][j]    For all i,j = 0, ..., n - 1
//        //    x[i][j] == 1            If arc (i,j) is selected
//        //    x[i][j] == 0            Otherwise
//        //
//        // INTEGER VARIABLE t[i]      For all i = 0, ..., n - 1
//        //    t[i] == k               Iff node i is the k-th node in the tour
//        //    t[0] == 1
//        //    t[i] in [2, ..., n]     For all i = 1, ... n - 1
//        //
//        // OBJECTIVE FUNCTION
//        //    MIN sum((i,j), c[i][j] * x[i][j])
//        //
//        // CONSTRAINTS
//        //    1) sum(j, x[j][i]) == 1                    For all i
//        //    2) sum(j, x[i][j]) == 1                    For all i
//        //    3) t[i] - t[j] + 1 <= n * (1 - x[i][j])    For all i,j = 1, ..., n - 1
//        //       Can be written as:
//        //       t[i] - t[j] + n * x[i][j] <= n - 1
//
//        // Variables
//        IloArray<IloNumVarArray> x(env, n);
//        IloNumVarArray t(env, n);
//
//        // Constraints
//        IloRangeArray inbound_arcs(env, n-1);  // Constraints 1)
//        IloRangeArray outbound_arcs(env, n-1); // Constraints 2)
//        IloArray<IloRangeArray> mtz(env, n); // Constraints 3)
//        IloRangeArray resources(env, 1); // Constraints 4)
//
//        // We use this stringstream to create variable and constraint names
//        std::stringstream name;
//
//        // Create variable t[0] and fix it to value 1
//        // This breaks symmetry, because it fixes node 0 as the starting node of the tour
//        t[0] = IloNumVar(env, 1, 1, IloNumVar::Int, "t_0");
//
//        // Create variables t[1], ..., t[n]
//        for(auto i = 1u; i < n; ++i) {
//            name << "t_" << i;
//            t[i] = IloNumVar(env, 2, n, IloNumVar::Int, name.str().c_str());
//            name.str(""); // Clean name
//        }
//
//        // Create variables x
//        for(auto i = 0u; i < n; ++i) {
//            x[i] = IloNumVarArray(env, n);
//            for(auto j = 0u; j < n; ++j) {
//            name << "x_" << i << "_" << j;
//            x[i][j] = IloNumVar(env, 0, 1, IloNumVar::Bool, name.str().c_str());
//            name.str(""); // Clean name
//            }
//        }
//
//        IloExpr expr(env);
//
//        // removing variables by setting upper bound = lower bound = 0
//        for (auto i = 2u; i < n; ++i){
//            for (auto j = 2u; j < n; ++j){
//                if (i == j || r.get_predicted_value(i,j) == 0){
//                    x[i][j].setUB(0);
//                }
//            }
//        }
//
//
//        // Create constraints 1)
//        for(auto i = 1u; i < n; ++i) {
//            for(auto j = 0u; j < n; ++j) {
//                expr += x[j][i];
//            }
//            name << "inbound_" << i;
//            if (i == 1u){
//                inbound_arcs[i-1] = IloRange(env, 1, expr, 1, name.str().c_str());
//            } else{
//                inbound_arcs[i-1] = IloRange(env, 0, expr, 1, name.str().c_str());
//            }
//            name.str(""); // Clean name
//            expr.clear(); // Clean expr
//        }
//        // Add constraints 1) to the model
//        model.add(inbound_arcs);
//
//        // Create constraints 2)
//        for(auto i = 0u; i < n; ++i) {
//            name << "outbound_" << i;
//            if (i == 0u){
//                for(auto j = 0u; j < n; ++j) {
//                    expr += x[i][j];
//                }
//                outbound_arcs[i] = IloRange(env, 1, expr, 1, name.str().c_str());
//            }else if (i >= 2u){
//                for(auto j = 0u; j < n; ++j) {
//                    expr += x[i][j] - x[j][i];
//                }
//                outbound_arcs[i-1] = IloRange(env, 0, expr, 0, name.str().c_str());
//            }
//            name.str(""); // Clean name
//            expr.clear(); // Clean expr
//        }
//        // Add constraints 2) to the model
//        model.add(outbound_arcs);
//
//        // Create constraints 3)
//        // The constraint is for i = 1,...,n and therefore we add empty constraints for i == 0
//        mtz[0] = IloRangeArray(env);
//        // We then continue normally for all other i > 0
//        for(auto i = 1u; i < n; ++i) {
//            mtz[i] = IloRangeArray(env, n);
//            for(auto j = 1u; j < n; ++j) {
//                expr = t[i] - t[j] + 1 - (static_cast<int>(n) - 1) * (1 - x[i][j]);
//                name << "mtz_" << i << "_" << j;
//                mtz[i][j] = IloRange(env, -IloInfinity, expr, 0, name.str().c_str());
//                name.str(""); // Clean name
//                expr.clear(); // Clean expr
//            }
//            // Add constraints 3)[i] to the model
//            model.add(mtz[i]);
//        }
//
//        // create resources constraint
//        name << "resources";
//        for(auto i = 0u; i < n; ++i) {
//            for(auto j = 0u; j < n; ++j) {
//                expr += g.cost(i, j) * x[i][j];
//            }
//        }
//        resources[0] = IloRange(env, 0, expr, static_cast<double>(g.get_Tmax()), name.str().c_str());
//        name.str(""); // Clean name
//        expr.clear(); // Clean expr
//        // Add constraints 4) to the model
//        model.add(resources[0]);
//
//
//        // Create objective function
//        for(auto i = 0u; i < n; ++i) {
//            for(auto j = 0u; j < n; ++j) {
//                expr += g.get_value(i) * x[i][j];
//            }
//        }
//        IloObjective obj(env, expr, IloObjective::Maximize);
//
//        // Add the objective function to the model
//        model.add(obj);
//
//        // Free the memory used by expr
//        expr.end();
//
//        // Create the solver object
//        IloCplex cplex(model);
//
//        // Export model to file (useful for debugging!)
//        cplex.exportModel("model.lp");
//
//        //Set cutoff time
//        cplex.setParam(IloCplex::TiLim, cutoff);
//
//        // Set number of threads
//        cplex.setParam(IloCplex::Threads, omp_get_max_threads());
//
////        // Set MIP emphasis
////        cplex.setParam(IloCplex::MIPEmphasis, CPX_MIPEMPHASIS_HIDDENFEAS);
//
////        //Set MIP search method
////        cplex.setParam(IloCplex::MIPSearch, CPX_MIPSEARCH_TRADITIONAL);
//
//
//        // warm start
//        IloNumVarArray startVar(env);
//        IloNumArray startVal(env);
//        std::vector<std::uint32_t> sol = r.get_best_sol_sampling();
//        std::vector<std::vector<std::uint32_t>> start(n, std::vector<std::uint32_t>(n, 0));
//        for (auto i = 0u; i < sol.size()-1; ++i){
//            start[sol[i]][sol[i+1]] = 1;
//        }
//        for(auto i = 0u; i < n; ++i){
//            for (auto j = 0u; j < n; ++j) {
//                startVar.add(x[i][j]);
//                startVal.add(start[i][j]);
//            }
//        }
//        cplex.addMIPStart(startVar, startVal);
//        startVal.end();
//        startVar.end();
//
//
//        bool solved = false;
//        try {
//            // Try to solve with CPLEX (and hope it does not raise an exception!)
//            solved = cplex.solve();
//        } catch(const IloException& e) {
//            std::cerr << "\n\nCPLEX Raised an exception:\n";
//            std::cerr << e << "\n";
//            env.end();
//            throw;
//        }
//
//        if(solved) {
//            // If CPLEX successfully solved the model, print the results
//            std::cout << "\n\nCplex success!\n";
//            std::cout << "\tStatus: " << cplex.getStatus() << "\n";
//            objVal = cplex.getObjValue();
//            std::cout << "\tObjective value: " << cplex.getObjValue() << "\n";
//            print_solution(cplex, x);
//        } else {
//            std::cerr << "\n\nCplex error!\n";
//            std::cerr << "\tStatus: " << cplex.getStatus() << "\n";
//            std::cerr << "\tSolver status: " << cplex.getCplexStatus() << "\n";
//        }
//        env.end();
//    }

//    void Solver::print_solution(const IloCplex& cplex, const IloArray<IloNumVarArray>& x) {
//        const auto n = g.size();
//        assert(x.getSize() == n);
////        opt_sol = std::vector<std::uint32_t>(n);
//        std::cout << "\n\nTour: ";
//        const auto starting_vertex = 0u;
//        const auto ending_vertex = 1u;
//        auto current_vertex = starting_vertex;
//        do {
//            opt_sol.push_back(current_vertex);
//            std::cout << current_vertex << " ";
//            for(auto i = 0u; i < n; ++i) {
//                if(cplex.getValue(x[current_vertex][i]) > .5) {
//                    current_vertex = i;
//                    break;
//                }
//            }
//        } while(current_vertex != ending_vertex);
//        opt_sol.push_back(current_vertex);
//        std::cout << current_vertex << "\n";
//    }
}
