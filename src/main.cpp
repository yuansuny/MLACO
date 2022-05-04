#include "graph.h"
#include "solver.h"
#include "reduce.h"
#include "training.h"
#include <iostream>
#include <time.h>
#include <sys/time.h>
#include <iomanip>
#include <sstream>

std::string ToString(int value, int digitsCount){
    std::ostringstream os;
    os << std::setfill('0') << std::setw(digitsCount) << value;
    return os.str();
}

double get_cpu_time(){
    return (double)clock() / CLOCKS_PER_SEC;
}

double get_wall_time(){
    struct timeval time;
    if (gettimeofday(&time,NULL)){
        return 0;
    }
    return (double)time.tv_sec + (double)time.tv_usec * .000001;
}

int main(int argc, char* argv[]) {
    using namespace OP;

    const std::string input_dir = "../../Datasets/random100/";
    const std::string output_dir = "../results/random100/";

    double cutoff = 1000;

    std::uint32_t runs = 1u;
    std::uint32_t d;
    sscanf(argv[1], "%d", &d);

//    std::string input_file_name = file_name[d];

    std::string input_file_name = ToString(d, 3);

    const int num_training_files = 20;
    std::vector<std::string> training_file_name;
    for (int i = 0; i < num_training_files; ++i){
        if (i != 0 && i != 19 && i != 45 && i != 48 && i != 50 && i != 62 && i != 77 && i != 86 && i != 89 && i != 93){
            training_file_name.push_back( ToString(i, 3) );
        }
//    for (int i = 1; i < num_training_files; ++i){
    }

    // uncomment the following two lines if training if required.
//    auto training = Training(training_file_name, "../../Datasets/random50/");
//    training.generate_training_model_svm();

//    training.generate_training_model_svm_linear();

    std::string output_obj_filename, output_time_filename, output_time_series_filename;
    output_obj_filename = output_dir + input_file_name + "_res_obj.csv";
    output_time_filename = output_dir + input_file_name + "_res_time.csv";
    output_time_series_filename = output_dir + input_file_name + "_res_time_series.csv";
    std::ofstream output_file_obj (output_obj_filename, std::ios::trunc);
    std::ofstream output_file_time (output_time_filename, std::ios::trunc);
    std::ofstream output_file_time_series (output_time_series_filename, std::ios::trunc);

    if (output_file_obj.is_open()){
        output_file_obj << "|E|, " << "Y_s, " << "|E_s|, " << "Y" << "\n";
    } else{
        std::cout << "Cannot open the output file " + output_obj_filename << "\n";
        return 0;
    }
    output_file_obj.close();

    if (output_file_time.is_open()){
        output_file_time << "t_read, " << "t_sample, " << "t_reduce, " << "t_total" << "\n";
    } else{
        std::cout << "Cannot open the output file " + output_time_filename << "\n";
        return 0;
    }
    output_file_time.close();

    if (output_file_time_series.is_open()){
        output_file_time_series << "time series data" << "\n";
    }
    output_file_time_series.close();

    for (auto i = 0u; i < runs; ++i){
        double w0 = get_wall_time();
        const auto graph = Graph(input_file_name, input_dir);
        auto num_edges = graph.size()*(graph.size()-1);
        std::cout << "Original number of edges is " << num_edges << "\n";
        output_file_obj.open(output_obj_filename, std::ios::app);
        output_file_obj << num_edges <<", ";

        double w1 = get_wall_time();
        double c1 = get_cpu_time();
        std::cout << "Reading time is " << w1 - w0 << "s\n";
        output_file_time.open(output_time_filename, std::ios::app);
        output_file_time << w1 - w0 <<", ";

        auto reduce = Reduce(graph);
        double w2 = get_wall_time();
        double c2 = get_cpu_time();
        std::cout << "Sampling wall time is  " << w2 - w1 << "s\n";
        std::cout << "Sampling CPU time is  " << c2 - c1 << "s\n";
        output_file_time << w2 - w1 <<", ";

//        reduce.removing_variables_cbm();
//        reduce.removing_variables_rbm();
        reduce.removing_variables_svm();
//        reduce.removing_variables_svm_linear();

        double w3 = get_wall_time();
        double c3 = get_cpu_time();
        std::cout << "removing variables wall time is  " << w3 - w2 << "s\n";
        std::cout << "removing variables CPU time is  " << c3 - c2 << "s\n";
//        //reduce.greedyTour();
    //    double w4 = get_wall_time();
        output_file_obj << reduce.get_objective_value_sampling() <<", ";
        std::cout << "Best objective value found from sampling is  " << reduce.get_objective_value_sampling() << "\n";

        output_file_obj << reduce.get_reduced_problem_size() <<", ";
//        std::cout << "Number of edges left is " << reduce.get_reduced_problem_size() << "\n" << std::endl;

        double w5 = get_wall_time();
        std::cout << "Reduction time is  " << w5 - w1 << "s\n";
        output_file_time << w5 - w1 <<", ";

        auto solver = Solver(graph, reduce, cutoff-(w5-w0));
//        solver.solve_tsp_concorde();
//        solver.solve_op_cplex();
//        solver.solver_op_ACO();
        solver.solver_op_MMAS();

        output_file_obj << solver.get_objective_value() << "\n";
        std::cout << "Best objective value found is  " << solver.get_objective_value() << "\n";
        output_file_obj.close();


        std::vector<double> time_series;
        time_series = solver.get_time_series_data();
        output_file_time_series.open(output_time_series_filename, std::ios::app);
        for (auto j = 0u; j < time_series.size(); ++j){
//            std::cout << time_series[j] << ", ";
            output_file_time_series << time_series[j] << ", ";
        }
        output_file_time_series << "\n";
        output_file_time_series.close();


        auto w6 = get_wall_time();
        std::cout << "Total time is  " << w6 - w1 << "s\n";
        output_file_time << w6 - w1 << "\n";
        output_file_time.close();
    }

    return 0;
}
