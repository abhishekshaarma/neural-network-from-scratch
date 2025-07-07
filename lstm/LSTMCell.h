#pragma once
#include <Eigen/Dense>
#include <random>
#include <cmath>

using namespace Eigen;
class LSTMCell
{
private:
    //weight 
    MatrixXd W_forget, W_input, W_candidate, W_output;
    // Bias vectors
    VectorXd b_forget, b_input, b_candidate, b_output;
    
    int input_size, hidden_size;
    //random number
    std::mt19937 gen;
    
    // Activation function
    
    MatrixXd sigmoid(const MatrixXd& x)
    {
        return 1.0 / (1.0 + (-x.array()).exp());
    }
    
    MatrixXd tanh_activation(const MatrixXd& x)
    {
        return x.array().tanh();
    }
    void initialize_weights()
    {
        std::uniform_real_distribution<double> dis(-1.0, 1.0);
        
        int combined_size = input_size + hidden_size;
        double limit = std::sqrt(6.0 / (combined_size + hidden_size));
        
        // Initialize weight matrices
        W_forget = Eigen::MatrixXd::Random(hidden_size, combined_size) * limit;
        W_input = Eigen::MatrixXd::Random(hidden_size, combined_size) * limit;
        W_candidate = Eigen::MatrixXd::Random(hidden_size, combined_size) * limit;
        W_output = Eigen::MatrixXd::Random(hidden_size, combined_size) * limit;
        
        // Initialize biases
        b_forget = Eigen::VectorXd::Ones(hidden_size);  // Forget gate bias = 1
        b_input = Eigen::VectorXd::Zero(hidden_size);
        b_candidate = Eigen::VectorXd::Zero(hidden_size);
        b_output = Eigen::VectorXd::Zero(hidden_size);
    }
 VectorXd hidden_state, cell_state;
    
public:
    LSTMCell(int input_size, int hidden_size) 
        : input_size(input_size), hidden_size(hidden_size), gen(std::random_device{}())
    {
        initialize_weights();
        // Initialize states
        hidden_state = VectorXd::Zero(hidden_size);
        cell_state = VectorXd::Zero(hidden_size);
    }
    
    // Main forward pass function
    std::pair<VectorXd, VectorXd> forward(const VectorXd& input) {
        // Concatenate input and previous hidden state
        VectorXd combined(input_size + hidden_size);
        combined.head(input_size) = input;
        combined.tail(hidden_size) = hidden_state;
        
        // Gate computations
        VectorXd forget_gate = sigmoid(W_forget * combined + b_forget);
        VectorXd input_gate = sigmoid(W_input * combined + b_input);
        VectorXd candidate_values = tanh_activation(W_candidate * combined + b_candidate);
        VectorXd output_gate = sigmoid(W_output * combined + b_output);
        
        // Update cell state
        cell_state = forget_gate.cwiseProduct(cell_state) + 
                     input_gate.cwiseProduct(candidate_values);
        
        // Update hidden state
        hidden_state = output_gate.cwiseProduct(tanh_activation(cell_state));
        
        return {hidden_state, cell_state};
    }
    
    void reset_states()
    {
        hidden_state = VectorXd::Zero(hidden_size);
        cell_state = VectorXd::Zero(hidden_size);
    }
    
    VectorXd get_hidden_state() const { return hidden_state; }
    VectorXd get_cell_state() const { return cell_state; }
};
