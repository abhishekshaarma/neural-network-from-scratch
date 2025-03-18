#include <iostream>
#include <Eigen>
#include "layer.h"
#include "namespace.h"
#include "NeuralNetwork.h"
#include "utils.h"

using namespace Eigen;
using namespace functions;

const std::string mnist_train_data_path = "train-images-idx3-ubyte";
const std::string mnist_train_label_path = "train-labels-idx1-ubyte";
const std::string mnist_test_data_path = "t10k-images-idx3-ubyte";
const std::string mnist_test_label_path = "t10k-labels-idx1-ubyte";

int main()
{
    srand(time(0));

    
    std::vector<VectorXd> train_dataset;
    std::vector<VectorXd> label_train_dataset;

    std::vector<VectorXd> test_dataset;
    std::vector<VectorXd> label_test_dataset;

    utils::read_mnist_train_data(mnist_train_data_path, train_dataset);
    utils::read_mnist_train_label(mnist_train_label_path, label_train_dataset);

    utils::read_mnist_test_data(mnist_test_data_path, test_dataset);
    utils::read_mnist_test_label(mnist_test_label_path, label_test_dataset);
    for(auto& image : train_dataset)
    {
        image /= 255.0;  // Normalize pixel values to [0,1] range
    }
    for(auto& image : test_dataset)
    {
        image /= 255.0;
    }
    Layer hidden_layer(784, 64, sigmoid, sigmoid_derivative);
    Layer output_layer(64, 10, sigmoid, sigmoid_derivative);

    NeuralNetwork nn({hidden_layer, output_layer});

nn.train(train_dataset, label_train_dataset, 0.01, 3);  // 


    nn.test(test_dataset, label_test_dataset);

    return 0;
}
