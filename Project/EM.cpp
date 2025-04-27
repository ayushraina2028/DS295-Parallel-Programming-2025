#include <bits/stdc++.h>
#include <chrono>
#include <filesystem>
#include "../../eigen-3.4.0/Eigen/Dense"
#include <sys/stat.h>
#include <omp.h>

using namespace std;
using namespace Eigen;
using namespace chrono;

high_resolution_clock::time_point getTime() {
    return high_resolution_clock::now();
}

double getDuration(high_resolution_clock::time_point start, high_resolution_clock::time_point end) {
    return duration_cast<duration<double>>(end - start).count();
}

class GaussianMixtureModel {

    private:
        int num_components;
        int maxIterations;
        double tolerance;
    
        /* Model Parameters */
        VectorXd weights;
        vector<VectorXd> means;
        vector<MatrixXd> covariances;
        vector<MatrixXd> precisions;  // Inverse of covariances
        vector<double> normalizers; 

        /* Best Model Tracking */
        double bestLogLikelihood;
        
        /* Timing Information */
        map<string, vector<double>> timing;
        int iterationCount;
        
        double multivariateGaussianPDF(const VectorXd& x, const VectorXd& mean, const MatrixXd& covariance) {
            // Pre-compute these outside the main loop if possible
            static double log_2pi = log(2.0 * M_PI);
            int d = x.size();
            
            // Cache these values for each component
            double logdet = log(covariance.determinant());
            MatrixXd precision = covariance.inverse();
            
            VectorXd diff = x - mean;
            double exponent = -0.5 * diff.transpose() * precision * diff;
            
            return exp(exponent - 0.5 * (d * log_2pi + logdet));
        }

        // In initialization or after covariance updates:
        void precomputePDFTerms() {
            int d = means[0].size();
            precisions.resize(num_components);
            normalizers.resize(num_components);
            
            for(int k = 0; k < num_components; k++) {
                precisions[k] = covariances[k].inverse();
                double logdet = log(covariances[k].determinant());
                normalizers[k] = -0.5 * (d * log(2.0 * M_PI) + logdet);
            }
        }

        // Then in PDF calculation:
        double fastMultivariateGaussianPDF(const VectorXd& x, int k) {
            VectorXd diff = x - means[k];
            double exponent = -0.5 * diff.transpose() * precisions[k] * diff;
            return exp(exponent + normalizers[k]);
        }
    
    public:
    
        GaussianMixtureModel(int num_components = 3, int max_iterations = 1000, double tolerance = 1e-4) {
            this->num_components = num_components;
            this->maxIterations = max_iterations;
            this->tolerance = tolerance;
            this->bestLogLikelihood = -numeric_limits<double>::infinity();
            this->iterationCount = 0;
            
            // Initialize timing map
            timing["initialization"] = vector<double>();
            timing["e_step"] = vector<double>();
            timing["m_step"] = vector<double>();
            timing["per_iteration"] = vector<double>();
            timing["total_fit"] = vector<double>(1, 0.0); // Will hold a single value
            timing["prediction"] = vector<double>();
        }
    
        void randomInitialization(const MatrixXd& X) {
            auto start_time = getTime();
            
            int n_samples = X.rows();
            int n_features = X.cols();
            
            // K-means++ style initialization for means
            random_device RD;
            mt19937 gen(RD());
            uniform_int_distribution<int> init_dist(0, n_samples - 1);
            uniform_real_distribution<double> prob_dist(0.0, 1.0);
            
            // Choose first center uniformly at random
            int first_idx = init_dist(gen);
            means.clear();
            means.push_back(X.row(first_idx).transpose());
            
            // Choose remaining centers with probability proportional to distance
            vector<double> distances(n_samples, numeric_limits<double>::max());
            
            for(int k = 1; k < num_components; k++) {
                double sum_distances = 0.0;
                
                // Compute distance to closest center for each point
                for(int i = 0; i < n_samples; i++) {
                    for(int j = 0; j < k; j++) {
                        double dist = (X.row(i).transpose() - means[j]).squaredNorm();
                        distances[i] = min(distances[i], dist);
                    }
                    sum_distances += distances[i];
                }
                
                // Choose next center with probability proportional to distance
                double threshold = prob_dist(gen) * sum_distances;
                int next_idx = 0;
                double cumulative = distances[0];
                
                while(cumulative < threshold && next_idx < n_samples - 1) {
                    next_idx++;
                    cumulative += distances[next_idx];
                }
                
                means.push_back(X.row(next_idx).transpose());
            }
            
            // Initialize covariances based on data spread
            covariances.clear();
            MatrixXd dataCentered = X.rowwise() - X.colwise().mean();
            MatrixXd dataCovariance = (dataCentered.transpose() * dataCentered) / (n_samples - 1);
            
            for(int i = 0; i < num_components; i++) {
                // Add small regularization for numerical stability
                MatrixXd covariance = dataCovariance + MatrixXd::Identity(n_features, n_features) * 1e-6;
                covariances.push_back(covariance);
            }
            
            // Initialize weights uniformly
            weights = VectorXd::Ones(num_components) / num_components;
            
            // Record timing
            auto end_time = getTime();
            timing["initialization"].push_back(getDuration(start_time, end_time));
        }
    
        // In GaussianMixtureModel::E_STEP method
        pair<MatrixXd, double> E_STEP(const MatrixXd& X) {
            auto start_time = getTime();
            
            int n_samples = X.rows();
            MatrixXd responsibilities = MatrixXd::Zero(n_samples, num_components);
            
            // Calculate responsibilities in parallel
            #pragma omp parallel for
            for(int i = 0; i < n_samples; i++) {
                VectorXd sample = X.row(i);
                VectorXd weighted_likelihoods(num_components);
                
                for(int k = 0; k < num_components; k++) {
                    weighted_likelihoods(k) = weights(k) * multivariateGaussianPDF(sample, means[k], covariances[k]);
                }
                
                double sum_weighted_likelihood = weighted_likelihoods.sum();
                for(int k = 0; k < num_components; k++) {
                    responsibilities(i, k) = weighted_likelihoods(k) / sum_weighted_likelihood;
                }
            }
            
            // Compute log-likelihood
            double log_likelihood = 0.0;
            #pragma omp parallel for reduction(+:log_likelihood)
            for(int i = 0; i < n_samples; i++) {
                double row_sum = 0.0;
                for(int k = 0; k < num_components; k++) {
                    row_sum += weights(k) * multivariateGaussianPDF(X.row(i), means[k], covariances[k]);
                }
                log_likelihood += log(row_sum);
            }
            
            // Record E-step time
            auto end_time = getTime();
            timing["e_step"].push_back(getDuration(start_time, end_time));
            
            return make_pair(responsibilities, log_likelihood);
        }
    
        void M_STEP(const MatrixXd& X, const MatrixXd& responsibilities) {
            auto start_time = getTime();
            
            int n_samples = X.rows();
            int n_features = X.cols();
        
            /* Soft Cluster Counts */
            VectorXd N_k = responsibilities.colwise().sum();
        
            /* Updating Weights */
            for(int k = 0; k < num_components; k++) {
                weights(k) = N_k(k) / n_samples;
            }
        
            /* Updating Means */
            means.resize(num_components);
            
            #pragma omp parallel for
            for(int k = 0; k < num_components; k++) {
                if(N_k(k) > 0) {
                    VectorXd weightedSum = VectorXd::Zero(n_features);
                    
                    // Compute weighted sum for this component
                    for(int i = 0; i < n_samples; i++) {
                        weightedSum += responsibilities(i, k) * X.row(i).transpose();
                    }
                    
                    means[k] = weightedSum / N_k(k);
                }
                else {
                    means[k] = VectorXd::Zero(n_features);
                }
            }
        
            /* Updating Covariances */
            covariances.resize(num_components);
            
            #pragma omp parallel for
            for(int k = 0; k < num_components; k++) {
                if(N_k(k) > 0) {
                    MatrixXd covariance = MatrixXd::Zero(n_features, n_features);
                    
                    // Compute covariance for this component
                    for(int i = 0; i < n_samples; i++) {
                        VectorXd diff = X.row(i).transpose() - means[k];
                        covariance += responsibilities(i, k) * diff * diff.transpose();
                    }
                    
                    covariance /= N_k(k);
                    covariances[k] = covariance + MatrixXd::Identity(n_features, n_features) * 1e-6;
                }
                else {
                    covariances[k] = MatrixXd::Identity(n_features, n_features) * 1e-6;
                }
            }
            
            // Precompute precision matrices and normalizers for PDF calculation
            precomputePDFTerms();
            
            // Record timing
            auto end_time = getTime();
            timing["m_step"].push_back(getDuration(start_time, end_time));
        }
    
        void fit(const MatrixXd& X) {
            auto total_start_time = getTime();
            iterationCount = 0;
            
            /* Random Initialization */
            randomInitialization(X);
            
            double prevLogLikelihood = -numeric_limits<double>::infinity();
            double currentLogLikelihood = -numeric_limits<double>::infinity();
            int noImprovementCount = 0;
    
            for(int iteration = 0; iteration < maxIterations; iteration++) {
                iterationCount++;
                auto iter_start_time = getTime();
                
                /* E-Step */
                auto RL = E_STEP(X);
                MatrixXd responsibilities = RL.first;
                double logLikelihood = RL.second;
                
                /* M-Step */
                M_STEP(X, responsibilities);
                
                /* Record iteration time */
                auto iter_end_time = getTime();
                timing["per_iteration"].push_back(getDuration(iter_start_time, iter_end_time));
                
                /* Checking Convergence */
                if(iteration > 0) {
                    double improvement = logLikelihood - currentLogLikelihood;
                    if(improvement < tolerance) {
                        noImprovementCount++;
                        if(noImprovementCount >= 3) {  // Require 3 consecutive small improvements
                            cout << "Converged at iteration " << iteration << endl;
                            break;
                        }
                    } else {
                        noImprovementCount = 0;
                    }
                }
                
                // Print progress every 10 iterations
                // if(iteration % 10 == 0) {
                cout << "Iteration " << iteration << ", Log Likelihood: " << logLikelihood << endl;
                // }
                
                prevLogLikelihood = currentLogLikelihood;
                currentLogLikelihood = logLikelihood;
                bestLogLikelihood = currentLogLikelihood;
            }
            
            // Record total fit time
            auto total_end_time = getTime();
            timing["total_fit"][0] = getDuration(total_start_time, total_end_time);
        }
    
        VectorXi predict(const MatrixXd& X) {
            auto start_time = getTime();
            
            auto L = E_STEP(X);
            MatrixXd responsibilities = L.first;
    
            VectorXi predictions(X.rows());
            for(int i = 0;i < X.rows(); i++) {
                int maxIndex = 0;
                double maxValue = responsibilities(i, 0);
    
                for(int j = 1;j < num_components; j++) {
                    if(responsibilities(i, j) > maxValue) {
                        maxValue = responsibilities(i, j);
                        maxIndex = j;
                    }
                }
    
                predictions(i) = maxIndex;
            }
            
            // Record prediction time
            auto end_time = getTime();
            timing["prediction"].push_back(getDuration(start_time, end_time));
    
            return predictions;
        }
        
        // Print timing statistics similar to Python version
        void printTimingStats() {
            cout << "=== GMM Timing Statistics ===" << endl;
            cout << "Total fit time: " << timing["total_fit"][0] << " seconds" << endl;
            
            if (!timing["initialization"].empty()) {
                double avg_init = 0;
                for (auto t : timing["initialization"]) avg_init += t;
                avg_init /= timing["initialization"].size();
                cout << "Average initialization time: " << avg_init << " seconds" << endl;
            }
            
            if (!timing["e_step"].empty()) {
                double avg_e = 0;
                for (auto t : timing["e_step"]) avg_e += t;
                avg_e /= timing["e_step"].size();
                cout << "Average E-step time: " << avg_e << " seconds" << endl;
            }
            
            if (!timing["m_step"].empty()) {
                double avg_m = 0;
                for (auto t : timing["m_step"]) avg_m += t;
                avg_m /= timing["m_step"].size();
                cout << "Average M-step time: " << avg_m << " seconds" << endl;
            }
            
            if (!timing["per_iteration"].empty()) {
                double avg_iter = 0;
                for (auto t : timing["per_iteration"]) avg_iter += t;
                avg_iter /= timing["per_iteration"].size();
                cout << "Average iteration time: " << avg_iter << " seconds" << endl;
                cout << "Total iterations: " << iterationCount << endl;
            }
            
            if (!timing["prediction"].empty()) {
                double avg_pred = 0;
                for (auto t : timing["prediction"]) avg_pred += t;
                avg_pred /= timing["prediction"].size();
                cout << "Average prediction time: " << avg_pred << " seconds" << endl;
            }
        }
        
        // Save timing information to file
        void saveTimingToFile(const string& filename, int n_samples, int n_features) {
            // Create timing directory if it doesn't exist - fixed filesystem approach
            #if defined(__cplusplus) && __cplusplus >= 201703L
                // C++17 approach with filesystem
                if (!filesystem::exists("timing_results")) {
                    filesystem::create_directory("timing_results");
                }
            #else
                // Fallback approach for older compilers
                struct stat info;
                if (stat("timing_results", &info) != 0) {
                    #ifdef _WIN32
                        system("mkdir timing_results");
                    #else
                        system("mkdir -p timing_results");
                    #endif
                }
            #endif
            
            ofstream file(filename);
            if (!file.is_open()) {
                cerr << "Error: Could not open file " << filename << endl;
                return;
            }
            
            // Get current date and time
            time_t now = time(0);
            char date_buf[80];
            struct tm tstruct = *localtime(&now);
            strftime(date_buf, sizeof(date_buf), "%Y-%m-%d %H:%M:%S", &tstruct);
            
            // Calculate total E-step and M-step time
            double e_step_total = 0;
            for (auto t : timing["e_step"]) e_step_total += t;
            
            double m_step_total = 0;
            for (auto t : timing["m_step"]) m_step_total += t;
            
            double init_total = 0;
            for (auto t : timing["initialization"]) init_total += t;
            
            double avg_iter = 0;
            for (auto t : timing["per_iteration"]) avg_iter += t;
            avg_iter /= timing["per_iteration"].size();
            
            double avg_e_step = e_step_total / timing["e_step"].size();
            double avg_m_step = m_step_total / timing["m_step"].size();
            
            double total_runtime = timing["total_fit"][0];
            double e_step_pct = (e_step_total / total_runtime) * 100;
            double m_step_pct = (m_step_total / total_runtime) * 100;
            
            double prediction_time = 0;
            if (!timing["prediction"].empty()) {
                prediction_time = timing["prediction"][0];
            }
            
            file << "=== C++ GMM Timing Summary ===" << endl;
            // file << "Date: " << date_buf << endl;
            file << "Dataset size: " << n_samples << " points, " << n_features << " dimensions" << endl;
            file << "Components: " << num_components << endl << endl;
            
            file << "--- Overall Performance ---" << endl;
            file << "Total runtime: " << total_runtime << " seconds" << endl;
            file << "Total iterations: " << iterationCount << endl;
            file << "Average iteration time: " << avg_iter << " seconds" << endl;
            file << "Prediction time: " << prediction_time << " seconds" << endl << endl;
            
            file << "--- Component Breakdown ---" << endl;
            file << "Initialization: " << init_total << " seconds" << endl;
            file << "E-step total: " << e_step_total << " seconds (" << e_step_pct << "%)" << endl;
            file << "M-step total: " << m_step_total << " seconds (" << m_step_pct << "%)" << endl << endl;
            
            file << "--- Per Iteration Statistics ---" << endl;
            file << "Average E-step time: " << avg_e_step << " seconds" << endl;
            file << "Average M-step time: " << avg_m_step << " seconds" << endl;
            
            file.close();
        }
    
        VectorXd getWeights() {
            return weights;
        }
    
        vector<VectorXd> getMeans() {
            return means;
        }
    
        vector<MatrixXd> getCovariances() {
            return covariances;
        }
    
        double getBestLogLikelihood() {
            return bestLogLikelihood;
        }
    };

// Function to read CSV data
MatrixXd readCSV(const string& filename, bool hasHeader = true) {
    ifstream file(filename);
    if (!file.is_open()) {
        cerr << "Error: Could not open file " << filename << endl;
        exit(1);
    }
    
    vector<vector<double>> data;
    string line;
    
    // Skip header if present
    if (hasHeader && getline(file, line)) {
        // Header skipped
    }
    
    while (getline(file, line)) {
        stringstream ss(line);
        vector<double> row;
        string value;
        
        while (getline(ss, value, ',')) {
            row.push_back(stod(value));
        }
        
        data.push_back(row);
    }
    
    // Convert to Eigen matrix
    int rows = data.size();
    int cols = rows > 0 ? data[0].size() : 0;
    
    MatrixXd matrix(rows, cols);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            matrix(i, j) = data[i][j];
        }
    }
    
    return matrix;
}

// Function to read labels
VectorXi readLabels(const string& filename) {
    ifstream file(filename);
    if (!file.is_open()) {
        cerr << "Error: Could not open file " << filename << endl;
        exit(1);
    }
    
    vector<int> labels;
    string line;
    
    // Skip header
    getline(file, line);
    
    while (getline(file, line)) {
        labels.push_back(stoi(line));
    }
    
    // Convert to Eigen vector
    VectorXi labelVector(labels.size());
    for (size_t i = 0; i < labels.size(); i++) {
        labelVector(i) = labels[i];
    }
    
    return labelVector;
}

// Function to calculate accuracy using a simple matching approach
double calculateAccuracy(const VectorXi& true_labels, const VectorXi& predicted_labels, int num_clusters) {
    // Create a mapping table for predicted to true labels
    MatrixXi contingency_table = MatrixXi::Zero(num_clusters, num_clusters);
    
    // Fill the contingency table
    for (int i = 0; i < true_labels.size(); i++) {
        contingency_table(predicted_labels(i), true_labels(i))++;
    }
    
    // Find the best match for each predicted cluster
    int matched_samples = 0;
    
    // For each predicted cluster, find the true cluster with most matching samples
    for (int i = 0; i < num_clusters; i++) {
        int max_match = 0;
        for (int j = 0; j < num_clusters; j++) {
            max_match = max(max_match, contingency_table(i, j));
        }
        matched_samples += max_match;
    }
    
    return static_cast<double>(matched_samples) / true_labels.size();
}

int main() {
    // Read data from CSV
    cout << "Reading data from CSV files..." << endl;
    MatrixXd X = readCSV("datasets/gmm_data.csv");
    VectorXi true_labels;
    
    bool has_labels = true;
    try {
        true_labels = readLabels("datasets/gmm_labels.csv");
    } catch (...) {
        cout << "No labels file found or it couldn't be read. Proceeding without ground truth." << endl;
        has_labels = false;
    }
    
    // Get dimensions
    int n_samples = X.rows();
    int n_features = X.cols();
    
    // Count unique labels to determine number of components
    int max_label = 0;
    if (has_labels) {
        for (int i = 0; i < true_labels.size(); i++) {
            max_label = max(max_label, true_labels(i));
        }
    }
    int true_components = has_labels ? max_label + 1 : 3; // Default to 3 if no labels
    
    cout << "Data loaded: " << n_samples << " samples with " << n_features << " features" << endl;
    cout << "Number of clusters: " << true_components << endl;
    
    // Fit GMM model
    cout << "Fitting GMM model..." << endl;
    GaussianMixtureModel gmm(true_components, 100, 1e-3);
    
    gmm.fit(X);
    cout << "Fitting completed" << endl;
    
    // Print timing statistics
    gmm.printTimingStats();
    
    // Save timing information to file
    gmm.saveTimingToFile("timing_results/cpp_gmm_timing_summary.txt", n_samples, n_features);
    cout << "Timing information saved to timing_results/cpp_gmm_timing_summary.txt" << endl;
    
    // Predict cluster assignments
    cout << "\nPredicting cluster assignments..." << endl;
    VectorXi predictions = gmm.predict(X);
    cout << "Prediction completed" << endl;
    
    // Get model parameters
    VectorXd weights = gmm.getWeights();
    vector<VectorXd> means = gmm.getMeans();
    vector<MatrixXd> covs = gmm.getCovariances();
    
    // Print model parameters
    cout << "\nFitted GMM Parameters:" << endl;
    cout << "Log-likelihood: " << gmm.getBestLogLikelihood() << endl;
    
    cout << "\nWeights:" << endl;
    for (int i = 0; i < true_components; i++) {
        cout << "Component " << i << ": " << weights(i) << endl;
    }
    
    cout << "\nMeans:" << endl;
    for (int i = 0; i < true_components; i++) {
        cout << "Component " << i << ": " << means[i].transpose() << endl;
    }
    
    cout << "\nCovariances:" << endl;
    for (int i = 0; i < true_components; i++) {
        cout << "Component " << i << ":\n" << covs[i] << endl;
    }

    // Calculate accuracy if we have true labels
    if (has_labels) {
        double accuracy = calculateAccuracy(true_labels, predictions, true_components);
        cout << "\nClustering accuracy: " << accuracy * 100 << "%" << endl;
    }
    
    // Count samples per predicted cluster
    VectorXi cluster_counts = VectorXi::Zero(true_components);
    for (int i = 0; i < n_samples; i++) {
        cluster_counts(predictions(i))++;
    }
    
    cout << "\nSamples per cluster:" << endl;
    for (int i = 0; i < true_components; i++) {
        cout << "Cluster " << i << ": " << cluster_counts(i) << " samples" << endl;
    }
    
    // Save cluster assignments to a CSV file
    ofstream outfile("gmm_predictions_cpp.csv");
    outfile << "index,predicted_cluster" << endl;
    for (int i = 0; i < n_samples; i++) {
        outfile << i << "," << predictions(i) << endl;
    }
    outfile.close();

    // Save model parameters to a CSV file
    ofstream model_file("gmm_model_parameters_cpp.csv");
    model_file << "Component,Weight,Mean,Covariance" << endl;
    for (int i = 0; i < true_components; i++) {
        model_file << i << "," << weights(i) << ",";
        model_file << means[i].transpose() << ",";
        
        /* Put Covariance components in one line in row major order */
        for (int j = 0; j < n_features; j++) {
            for (int k = 0; k < n_features; k++) {
                model_file << covs[i](j, k);
                if (j != n_features - 1 || k != n_features - 1) {
                    model_file << " ";
                }
            }
        }
        model_file << endl;
    }
    model_file.close();
    cout << "\nModel parameters saved to gmm_model_parameters_cpp.csv" << endl;
    cout << "Cluster assignments saved to gmm_predictions_cpp.csv" << endl;
    
    cout << "GMM Testing completed successfully!" << endl;
    
    return 0;
}