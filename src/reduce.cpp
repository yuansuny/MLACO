#include "reduce.h"
#include <cmath>
#include <limits>

namespace OP {
    Reduce::Reduce(const Graph& g) : g{g} {
        multi_thread_sampling();
    }

    void Reduce::multi_thread_sampling() {
        num_thread = omp_get_max_threads();
        std::cout << "threads used: " << num_thread << std::endl;
        const auto n = g.size();
        sample_size = sample_factor * n;
        std::cout << "sample size is " << sample_size << std::endl;
        samples.resize(sample_size);
        objs = std::vector<double>(sample_size, 0.0);

        #pragma omp parallel
        {
            int threadnum = omp_get_thread_num();
            std::uint32_t low = sample_size * threadnum / num_thread;
            std::uint32_t high = sample_size * (threadnum+1) / num_thread;
            std::random_device rd;
            std::mt19937 gen(rd());
            srand (time(NULL));
            std::vector<std::uint32_t> sequence(n-2);
            std::iota(sequence.begin(), sequence.end(), 2);
            std::uint32_t idx;
            double Tcrt;
            std::vector<double> Tmax(n);
            for (std::uint32_t j = 0u; j < n; ++j){
                Tmax[j] = g.get_Tmax() - g.cost(j,1u);
            }
            for(std::uint32_t i = low; i < high; ++i) {
                std::shuffle(sequence.begin(), sequence.end(), gen);
                samples[i].push_back(0u);
                idx = 0u;
                Tcrt = 0;
                for(auto j = 0u; j < n-2; ++j){
                    if ( Tcrt + g.cost(idx,sequence[j] ) <= Tmax[sequence[j]] ){
                        Tcrt += g.cost(idx,sequence[j]);
                        idx = sequence[j];
                        samples[i].push_back(idx);
                        objs[i] += g.get_value(idx);
                    }
                }
                samples[i].push_back(1u);
            }
        }

        std::uint32_t id = 0;
        double max_obj = objs[0];
        for (auto i = 1u; i < sample_size; ++i){
            if (objs[i] > max_obj){
                max_obj = objs[i];
                id = i;
            }
        }
        std::cout << "best obj in sampling is " << objs[id] << "\n";
        best_sol_sampling = samples[id];
        best_obj_sampling = objs[id];
    }

    void Reduce::removing_variables_rbm() {
        std::cout << "problem reduction using ranking-based measure with threshold " << threshold_r << std::endl;
        compute_ranking_based_measure();
        const auto n = g.size();
        predicted_value = std::vector<std::vector<float>>(n, std::vector<float>(n, 0));
        for (auto i = 0u; i < n; ++i){
            for (auto j = 0u; j < n; ++j){
                if (ranking_scores[i][j] < threshold_r){
                    predicted_value[i][j] = 0;
                } else{
                    predicted_value[i][j] = 1;
                }
            }
        }
//        repair();
//        compute_reduced_problem_size();
    }

    void Reduce::compute_ranking_based_measure() {
        std::cout << "computing ranking based measure " << std::endl;
        const auto n = g.size();
        std::vector<std::uint32_t> sort_idx(sample_size);
        std::iota(sort_idx.begin(), sort_idx.end(), 0);
        std::vector<double> objs_copy(objs);
        std::sort(sort_idx.begin(), sort_idx.end(), [&objs_copy](std::uint32_t i1, std::uint32_t i2) {return objs_copy[i1] > objs_copy[i2];});
        ranking_scores = std::vector<std::vector<float>>(n, std::vector<float>(n, 0.0));
        rank = std::vector<std::uint32_t>(sample_size);
        for (auto i = 0u; i < sample_size; i++){
            rank[sort_idx[i]] = i+1;
        }

        #pragma omp parallel
        {
//            std::cout << "computing ranking scores from thread " << omp_get_thread_num() << std::endl;
            std::vector<std::vector<float>> ranking_scores_local = std::vector<std::vector<float>>(n, std::vector<float>(n, 0.0));
            int threadnum = omp_get_thread_num();
            int low = sample_size * threadnum / num_thread;
            int high = sample_size * (threadnum+1) / num_thread;
            int ind1, ind2;
            for (int i = low; i < high; ++i){
                for (auto j = 0u; j < samples[i].size()-1; ++j){
                    ind1 = samples[i][j];
                    ind2 = samples[i][j+1];
                    ranking_scores_local[ind1][ind2] += 1.0/rank[i];
                }
            }
            #pragma omp critical
            for (auto i = 0u; i < n; ++i){
                for (auto j = 0u; j < n; ++j){
                    ranking_scores[i][j] += ranking_scores_local[i][j];
                }
            }
        }

        max_rbm = 0;
        for (auto i = 0u; i < n; ++i){
            for (auto j = 0u; j < n; ++j){
                if (ranking_scores[i][j] > max_rbm){
                    max_rbm = ranking_scores[i][j];
                }
            }
        }
    }


    void Reduce::removing_variables_cbm(){
        std::cout << "problem reduction using correlation-based measure with threshold " << threshold_c << std::endl;
        compute_correlation_based_measure();
        const auto n = g.size();
        predicted_value = std::vector<std::vector<float>>(n, std::vector<float>(n, 0));
        for (auto i = 0u; i < n; ++i){
            for (auto j = 0u; j < n; ++j){
                if (corr_xy[i][j] > threshold_c){
                    predicted_value[i][j] = 1;
                } else{
                    predicted_value[i][j] = 0;
                }
            }
        }
//        repair();
//        compute_reduced_problem_size();
    }

    void Reduce::compute_correlation_based_measure(){
        std::cout << "computing correlation based measure " << std::endl;
        const auto n = g.size();
        double mean_y = 0.0;
        for (auto i = 0u; i < sample_size; ++i){
            mean_y += objs[i];
        }
        mean_y = mean_y/sample_size;
        diff_y = std::vector<double>(sample_size);
        double variance_y = 0.0, sum_diff_y = 0.0;
        for (auto i = 0u; i < sample_size; ++i){
            diff_y[i] = objs[i] - mean_y;
            variance_y += diff_y[i]*diff_y[i];
            sum_diff_y += diff_y[i];
        }

        mean_x = std::vector<std::vector<double>> (n, std::vector<double>(n, 0.0));
        S1 = std::vector<std::vector<double>>(n, std::vector<double>(n,0.0));

        #pragma omp parallel
        {
//            std::cout << "computing correlation measure from thread " << omp_get_thread_num() << std::endl;
            std::vector<std::vector<double>> mean_x_local = std::vector<std::vector<double>>(n, std::vector<double>(n, 0.0));
            std::vector<std::vector<double>> S1_local = std::vector<std::vector<double>>(n, std::vector<double>(n, 0.0));

            int threadnum = omp_get_thread_num();
            int low = sample_size * threadnum / num_thread;
            int high = sample_size * (threadnum+1) / num_thread;
            double ratio = 1.0/sample_size;
            int ind1, ind2;
            for (int i = low; i < high; ++i){
                for (auto j = 0u; j < samples[i].size()-1; ++j){
                    ind1 = samples[i][j];
                    ind2 = samples[i][j+1];
                    mean_x_local[ind1][ind2] += ratio;
                    S1_local[ind1][ind2] += diff_y[i];
                }
            }

            #pragma omp critical
            for (auto i = 0u; i < n; ++i){
                for (auto j = 0u; j < n; ++j){
                    mean_x[i][j] += mean_x_local[i][j];
                    S1[i][j] += S1_local[i][j];
                }
            }
        }

        std::vector<std::vector<double>> variance_x(n, std::vector<double>(n, 0.0));
        std::vector<std::vector<double>> variance_xy(n, std::vector<double>(n,0.0));
        corr_xy = std::vector<std::vector<float>>(n, std::vector<float>(n,0.0));
        max_cbm = -1;
        min_cbm = 1;
        for (auto i = 0u; i < n; ++i){
            for (auto j = 0u; j < n; ++j){
                variance_x[i][j] = mean_x[i][j]*(1-mean_x[i][j])*sample_size;
                variance_xy[i][j] = (1-mean_x[i][j])*S1[i][j] - mean_x[i][j]*(sum_diff_y - S1[i][j]);
                if (variance_x[i][j] > 0){
                    corr_xy[i][j] = variance_xy[i][j]/sqrt(variance_x[i][j]*variance_y);
                }
                if (corr_xy[i][j] > max_cbm){
                    max_cbm = corr_xy[i][j];
                }
                if (corr_xy[i][j] < min_cbm){
                    min_cbm = corr_xy[i][j];
                }
            }
        }
    }

    void Reduce::constructing_test_data(){
        compute_correlation_based_measure();
        compute_ranking_based_measure();

        #pragma omp parallel
        {
            int threadnum = omp_get_thread_num();
//            std::cout << "constructing_test_data from thread " << threadnum << std::endl;
            std::string test_s = test_data_dir + g.get_file_name() + test_data_name + std::to_string(threadnum) + ".csv";

            char test_data[test_s.size()+1];
            strcpy(test_data, test_s.c_str());
            std::ofstream test_file(test_data, std::ios::trunc);
            if (! test_file.is_open()){
                std::cout << "Cannot open the output file " <<  test_data << "\n";
            }

            int n = g.size();
            int low = n * threadnum / num_thread;
            int high = n * (threadnum + 1) / num_thread;

            for (int i = low; i < high; ++i){
                for (int j = 2; j < n; ++j){
                    if (i != j){
                        test_file << 0 << " ";
                        test_file << "1:" << std::fixed << std::setprecision(6) << corr_xy[i][j]/max_cbm << " ";
                        test_file << "2:" << std::fixed << std::setprecision(6) << ranking_scores[i][j]/max_rbm << " ";
                        test_file << "3:" << std::fixed << std::setprecision(6) << g.cost(i,j) / g.get_Tmax() << " ";
                        test_file << "4:" << std::fixed << std::setprecision(6) << g.get_value_per_cost(i,j) / g.get_max_out_vpc(i) << " ";
                        test_file << "5:" << std::fixed << std::setprecision(6) << g.get_value_per_cost(i,j) / g.get_max_in_vpc(j) << " " <<"\n";
                    }

                }
            }
            test_file.close();
        }
    }

    void Reduce::loading_output_data(){
        int n = g.size();
        predicted_value = std::vector<std::vector<float>>(n, std::vector<float>(n, 0));
        #pragma omp parallel
        {
            int threadnum = omp_get_thread_num();
//            std::cout << "loading predicted value from thread " << threadnum << std::endl;

            std::string output_s = test_data_dir + g.get_file_name() + output_file_name + std::to_string(threadnum) + ".csv";
            char output_file[output_s.size()+1];
            strcpy(output_file, output_s.c_str());
            std::ifstream predicted_file(output_file);
            if (! predicted_file){
                std::cout << "fail to read the predicted file \n";
            }

            int low = n * threadnum / num_thread;
            int high = n * (threadnum + 1) / num_thread;
            float p0, p1, lb;
            std::string s1;
            predicted_file >> s1 >> p0 >> p1;
             for (int i = low; i < high; ++i){
                for (int j = 2; j < n; ++j){
                    if (i != j){
                        predicted_file >> lb >> p0 >> p1;
                        predicted_value[i][j] = p1;
                    }
                }
            }
            predicted_file.close();
            remove(output_file);
//            const int rem_result = remove(output_file);
//            if(rem_result == 0){
//                std::cout << "Successfully remove predicted data file from thread " << threadnum << std::endl;
//            } else {
//                std::cout << "No such predicted data file from thread " << threadnum << std::endl;
//            }
        }
    }

    // Removing variables using SVM
    void Reduce::removing_variables_svm(){
        std::cout << "problem reduction using SVM" << std::endl;

        constructing_test_data();

        #pragma omp parallel
        {
            int threadnum = omp_get_thread_num();
//            std::cout << "SVM prediction from thread " << threadnum << std::endl;

            std::string test_s = test_data_dir + g.get_file_name() + test_data_name + std::to_string(threadnum) + ".csv";
            std::string output_s = test_data_dir + g.get_file_name() + output_file_name + std::to_string(threadnum) + ".csv";
            std::string model_s = training_data_dir + training_model_name;
            char test_data[test_s.size()+1];
            char output_file[output_s.size()+1];
            char model_file[model_s.size()+1];
            strcpy(test_data, test_s.c_str());
            strcpy(output_file, output_s.c_str());
            strcpy(model_file, model_s.c_str());

            #pragma omp critical
            svm_predict_model(test_data, model_file, output_file, probability);

            remove(test_data);

//            std::cout << "SVM prediction from thread " << threadnum << " done" << std::endl;
        }

        loading_output_data();
//        repair();
//        compute_reduced_problem_size();
//        std::cout << "problem reduction using SVM done" << std::endl;
    }


    // Removing variables using Linear SVM (fast)
    void Reduce::removing_variables_svm_linear(){
        std::cout << "problem reduction using linear SVM" << std::endl;
        constructing_test_data();

        #pragma omp parallel
        {
            int threadnum = omp_get_thread_num();
//            std::cout << "SVM prediction from thread " << threadnum << std::endl;

            std::string test_s = test_data_dir + g.get_file_name() + test_data_name + std::to_string(threadnum) + ".csv";
            std::string output_s = test_data_dir + g.get_file_name() + output_file_name + std::to_string(threadnum) + ".csv";
            std::string model_s = training_data_dir + training_model_name;
            char test_data[test_s.size()+1];
            char output_file[output_s.size()+1];
            char model_file[model_s.size()+1];
            strcpy(test_data, test_s.c_str());
            strcpy(output_file, output_s.c_str());
            strcpy(model_file, model_s.c_str());

            #pragma omp critical
            linear_svm_predict_model(test_data, model_file, output_file, probability);
//            std::cout << "Linear SVM prediction from thread " << threadnum << " done" << std::endl;

            remove(test_data);

//            const int rem_result = remove(test_data);
//            if(rem_result == 0){
//                std::cout << "Successfully remove test data file from thread " << threadnum << std::endl;
//            } else {
//                std::cout << "No such test data file from thread " << threadnum << std::endl;
//            }
        }

        loading_output_data();
//        repair();
//        compute_reduced_problem_size();
//        std::cout << "Linear SVM prediction done" << std::endl;
    }

    //adding the best solution found in sampling to make sure the graph is still connected
    void Reduce::repair(){
        for (auto j = 0u; j < best_sol_sampling.size()-1; ++j){
            predicted_value[best_sol_sampling[j]][best_sol_sampling[j+1]] = 1;
        }
    }

    void Reduce::compute_reduced_problem_size(){
        const auto n = g.size();
        std::uint64_t count = 0u;
        for (auto i = 2u; i < n; ++i){
            for (auto j = 2u; j < n; ++j){
                if(i != j && predicted_value[i][j] == 1)
                    count += 1;
            }
        }
        num_edge_left = count;
        std::cout << "proportion of remaining problem size is " << (double)(count+2*n)/(n*(n-1)) << std::endl;
    }

}
