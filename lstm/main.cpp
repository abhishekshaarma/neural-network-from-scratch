#include "LSTMNetwork.h"
#include <iostream>
int main() {
    // Example usage
    int input_size = 10;
    int hidden_size = 20;
    int output_size = 5;
    int sequence_length = 15;
    int batch_size = 8;
    
    // Create network that returns only final output
    LSTMNetwork lstm(input_size, hidden_size, output_size, false);
    
    // Create network that returns outputs for all time steps
    LSTMNetwork lstm_seq(input_size, hidden_size, output_size, true);
    
    // Create a random input sequence
    MatrixXd input_sequence = MatrixXd::Random(sequence_length, input_size);
    
    // Process the sequence
    MatrixXd output = lstm.forward(input_sequence);
    std::cout << "Final output shape: " << output.rows() << "x" << output.cols() << std::endl;
    
    // Process with return_sequences=true
    MatrixXd seq_output = lstm_seq.forward(input_sequence);
    std::cout << "Sequence output shape: " << seq_output.rows() << "x" << seq_output.cols() << std::endl;
    
    // Process a batch of sequences
    std::vector<MatrixXd> batch(batch_size, input_sequence);
    auto batch_outputs = lstm.forward_batch(batch);
    std::cout << "Batch size: " << batch_outputs.size() << std::endl;
    
    return 0;
}
