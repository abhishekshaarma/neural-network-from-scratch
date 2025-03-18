#include <Eigen>

using Eigen::MatrixXd;
using Eigen::VectorXd;

class Layer
{
private:
    MatrixXd weights;  //matrix for the weights
    VectorXd biases;   //matrix for the biases 
    VectorXd neurons_values;  //matrix for the each neuron biases 
    VectorXd neurons_values_activate;
    std::function<VectorXd (const VectorXd &)> activation_function; // activation function
    std::function<VectorXd (const VectorXd &)> activation_function_derivative; // activation function
    
    VectorXd delta;
public:

    Layer(int input_size_neurons, int neurons,
          std::function<VectorXd(const VectorXd &)> activation_function,  // Parameter named "activation_funtion"
          std::function<VectorXd(const VectorXd &)> activation_function_derivative)
        :activation_function(activation_function),  // But assigning to "activation_function"
         activation_function_derivative(activation_function_derivative)    {
//        weights = MatrixXd::Random(input_size_neurons, neurons);
        weights = MatrixXd::Random(neurons, input_size_neurons) * sqrt(1.0 / input_size_neurons);



        biases = VectorXd::Zero(neurons);
    }
    
void forward(const VectorXd &input)
    {
        VectorXd Y = weights.transpose() * input + biases;
        neurons_values = Y;
        neurons_values_activate = activation_function(Y);
        // In Layer::forward
        if (input.array().isNaN().any()) std::cout << "NaN in forward input" << std::endl;
        if (neurons_values.array().isNaN().any()) std::cout << "NaN in neurons_values" << std::endl;
        if (neurons_values_activate.array().isNaN().any()) std::cout << "NaN in activate" << std::endl;


    }
    
    VectorXd get_neurons_values_activate()
    {
        return neurons_values_activate;
    }
    
    void set_delta(const VectorXd& delta)
    {
        this->delta = delta;
    }
    VectorXd get_delta()
    {
        return delta;
    }
   MatrixXd get_weights()
    {
        return weights;  // Return the weights matrix
    }
   
    void update_weight(const VectorXd& input, double learning_rate)
    {
        weights = weights - learning_rate * (delta * input.transpose());
        
        biases  = biases - learning_rate * delta;
        // In Layer::update_weight 
        if (delta.array().isNaN().any()) std::cout << "NaN in delta" << std::endl;
        if (weights.array().isNaN().any()) std::cout << "NaN in weights after update" << std::endl;
    }

    VectorXd derivative_of_activation_function()
    {
        return activation_function_derivative(neurons_values);
    }
};
