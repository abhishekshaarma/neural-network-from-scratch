#include <iostream>
#include <Eigen>
#include "layer.h"
#include "namespace.h"
#include "NeuralNetwork.h"
#include "utils.h"
using namespace Eigen;
using namespace functions;
const std::string mnist_train_data_path = "train-images.idx3-ubyte";
const std::string mnist_train_label_path = "train-labels.idx1-ubyte";
const std::string mnist_test_data_path = "t10k-images.idx3-ubyte";
const std::string mnist_test_label_path = "t10k-labels.idx1-ubyte";
int main()
{

    //  declare the variables
    std::vector<VectorXd> train_dataset;
    std::vector<VectorXd> label_train_dataset;
    std::vector<VectorXd> test_dataset;
    std::vector<VectorXd> label_test_dataset;

    //  load the data once
    try
    {
        utils::read_mnist_train_data(mnist_train_data_path, train_dataset);
        utils::read_mnist_train_label(mnist_train_label_path, label_train_dataset);
        utils::read_mnist_test_data(mnist_test_data_path, test_dataset);
        utils::read_mnist_test_label(mnist_test_label_path, label_test_dataset);
    }
    catch (const std::exception& e)
    {
        std::cerr << "Error loading MNIST files: " << e.what() << std::endl;
        return 1;
    }

    //create the smaller datasets
    const size_t train_samples_to_use = 10000;
    const size_t test_samples_to_use = 2000;
    std::vector<VectorXd> train_dataset_small(train_dataset.begin(), train_dataset.begin() + train_samples_to_use);
    std::vector<VectorXd> label_train_dataset_small(label_train_dataset.begin(), label_train_dataset.begin() + train_samples_to_use);
    std::vector<VectorXd> test_dataset_small(test_dataset.begin(), test_dataset.begin() + test_samples_to_use);
    std::vector<VectorXd> label_test_dataset_small(label_test_dataset.begin(), label_test_dataset.begin() + test_samples_to_use);
    std::cout << "Train dataset size: " << train_dataset.size() << std::endl;
    std::cout << "Train labels size: " << label_train_dataset.size() << std::endl;
    std::cout << "Test dataset size: " << test_dataset.size() << std::endl;
    std::cout << "Test labels size: " << label_test_dataset.size() << std::endl;
    
   
    
    // Optionally, display the image (pseudo-code, implement based on your needs)
    // utils::display_image(train_dataset[sample_idx], 28, 28);

    for(auto& image : train_dataset_small)
    {
        image /= 255.0;
    }
    for(auto& image : test_dataset_small)
    {
        image /= 255.0;
    }
    Layer hidden_layer(784, 64, sigmoid, sigmoid_derivative);
    Layer output_layer(64, 10, sigmoid, sigmoid_derivative);
    
    NeuralNetwork nn({hidden_layer, output_layer});

    nn.train(train_dataset_small, label_train_dataset_small, 0.1, 3);
    nn.test(test_dataset_small, label_test_dataset_small);
    return 0;
}



    /*
    
    
   
    
    // Expected sizes (MNIST standard)
    const size_t expected_train_size = 60000;
    const size_t expected_test_size = 10000;
    
    if (train_dataset.size() != expected_train_size || 
        label_train_dataset.size() != expected_train_size) {
        std::cerr << "Warning: Training data size mismatch! Expected " 
                 << expected_train_size << " samples." << std::endl;
    }
    
    if (test_dataset.size() != expected_test_size || 
        label_test_dataset.size() != expected_test_size) {
        std::cerr << "Warning: Test data size mismatch! Expected " 
                 << expected_test_size << " samples." << std::endl;
    }
    
    // Check vector dimensions
    if (!train_dataset.empty()) {
        std::cout << "Image vector size: " << train_dataset[0].size() << std::endl;
        if (train_dataset[0].size() != 784) { // 28x28 images
            std::cerr << "Warning: Unexpected image dimension!" << std::endl;
        }
    }
    
    if (!label_train_dataset.empty()) {
        std::cout << "Label vector size: " << label_train_dataset[0].size() << std::endl;
        if (label_train_dataset[0].size() != 10) { // One-hot encoded labels
            std::cerr << "Warning: Unexpected label dimension!" << std::endl;
        }
    }
    
    // Display a sample image (if you have a display function)
    if (!train_dataset.empty() && !label_train_dataset.empty()) {
        int sample_idx = 0; // First image
        
        // Find the label (assuming one-hot encoding)
        int label = -1;
        for (int i = 0; i < label_train_dataset[sample_idx].size(); i++) {
            if (label_train_dataset[sample_idx](i) > 0.5) {
                label = i;
                break;
            }
            utils::display_ascii_image(train_dataset[sample_idx], 28, 28);
                
        }
        
        std::cout << "Sample image #" << sample_idx << " has label: " << label << std::endl;
        
        
    }
    
 
    

    */
