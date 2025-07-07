#pragma once
#include "LSTMCell.h"
#include <vector>

class LSTMNetwork {
private:
    std::vector<LSTMCell> cells;  // One cell per time step (unrolled)
    int input_size;
    int hidden_size;
    int output_size;
    bool return_sequences;  // Whether to return outputs for all time steps or just last
    
    // Output layer weights and bias
    MatrixXd W_out;
    VectorXd b_out;
    
public:
    LSTMNetwork(int input_size, int hidden_size, int output_size, bool return_sequences = false)
        : input_size(input_size), 
          hidden_size(hidden_size), 
          output_size(output_size),
          return_sequences(return_sequences) 
    {
        // Initialize a single cell (we'll reuse it for each time step)
        cells.emplace_back(input_size, hidden_size);
        
        // Initialize output layer weights
        W_out = MatrixXd::Random(output_size, hidden_size) * 0.01;
        b_out = VectorXd::Zero(output_size);
    }
    
    // Forward pass for a single sequence
    MatrixXd forward(const MatrixXd& inputs) {
        int sequence_length = inputs.rows();
        MatrixXd outputs;
        
        if (return_sequences) {
            outputs.resize(sequence_length, output_size);
        } else {
            outputs.resize(1, output_size);
        }
        
        // Process each time step
        for (int t = 0; t < sequence_length; t++) {
            VectorXd input = inputs.row(t);
            
            // Forward pass through the LSTM cell
            auto [hidden_state, cell_state] = cells[0].forward(input);
            
            // Compute output for this time step
            VectorXd output = W_out * hidden_state + b_out;
            
            if (return_sequences) {
                outputs.row(t) = output;
            } else if (t == sequence_length - 1) {
                outputs.row(0) = output;
            }
        }
        
        return outputs;
    }
    
    // Forward pass for a batch of sequences
    std::vector<MatrixXd> forward_batch(const std::vector<MatrixXd>& batch_inputs) {
        std::vector<MatrixXd> batch_outputs;
        batch_outputs.reserve(batch_inputs.size());
        
        for (const auto& inputs : batch_inputs) {
            batch_outputs.push_back(forward(inputs));
        }
        
        return batch_outputs;
    }
    
    void reset_states() {
        for (auto& cell : cells) {
            cell.reset_states();
        }
    }
    
    // Getter for hidden state
    VectorXd get_hidden_state() const {
        return cells[0].get_hidden_state();
    }
    
    // Getter for cell state
    VectorXd get_cell_state() const {
        return cells[0].get_cell_state();
    }
};
