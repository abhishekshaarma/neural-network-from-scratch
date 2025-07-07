#ifndef UTILS_H
#define UTILS_H

#include <iostream>
#include <fstream>
#include <Eigen>
#include <arpa/inet.h>


namespace utils
{
    void read_mnist_train_data(const std::string& path, 
                                std::vector<Eigen::VectorXd>& data)
    {
        std::ifstream file(path, std::ios::binary);
        if (file.is_open()) {
            int magic_number = 0;
            int number_of_images = 0;
            int rows = 0;
            int cols = 0;

            file.read((char*)&magic_number, sizeof(magic_number));
            magic_number = ntohl(magic_number);
            file.read((char*)&number_of_images, sizeof(number_of_images));
            number_of_images = ntohl(number_of_images);
            file.read((char*)&rows, sizeof(rows));
            rows = ntohl(rows);
            file.read((char*)&cols, sizeof(cols));
            cols = ntohl(cols);

            for (int i = 0; i < number_of_images; i++)
            {
                Eigen::VectorXd vec(rows * cols);
                for (int r = 0; r < rows; r++)
                {
                    for (int c = 0; c < cols; c++)
                    {
                        unsigned char temp = 0;
                        file.read((char*)&temp, sizeof(temp));
                        vec(r * cols + c) = (double)temp / 255.0;

                    }
                }
                data.push_back(vec);
            }
        }
    }

    void read_mnist_train_label(const std::string& path, 
                                std::vector<Eigen::VectorXd>& data)
    {
        std::ifstream file(path, std::ios::binary);
        if (file.is_open())
        {
            int magic_number = 0;
            int number_of_items = 0;

            file.read((char*)&magic_number, sizeof(magic_number));
            magic_number = ntohl(magic_number);
            file.read((char*)&number_of_items, sizeof(number_of_items));
            number_of_items = ntohl(number_of_items);

            for (int i = 0; i < number_of_items; i++)
            {
                unsigned char temp = 0;
                file.read((char*)&temp, sizeof(temp));
                Eigen::VectorXd vec(10);
                vec.setZero();
                vec((int)temp) = 1.0;
                data.push_back(vec);
            }
        }
    }
    
    void read_mnist_test_data(const std::string& path, 
                                std::vector<Eigen::VectorXd>& data)
    {
        std::ifstream file(path, std::ios::binary);
        if (file.is_open()) {
            int magic_number = 0;
            int number_of_images = 0;
            int rows = 0;
            int cols = 0;

            file.read((char*)&magic_number, sizeof(magic_number));
            magic_number = ntohl(magic_number);
            file.read((char*)&number_of_images, sizeof(number_of_images));
            number_of_images = ntohl(number_of_images);
            file.read((char*)&rows, sizeof(rows));
            rows = ntohl(rows);
            file.read((char*)&cols, sizeof(cols));
            cols = ntohl(cols);

            for (int i = 0; i < number_of_images; i++)
            {
                Eigen::VectorXd vec(rows * cols);
                for (int r = 0; r < rows; r++)
                {
                    for (int c = 0; c < cols; c++)
                    {
                        unsigned char temp = 0;
                        file.read((char*)&temp, sizeof(temp));
                        vec(r * cols + c) = (double)temp / 255.0;
                    }
                }
                data.push_back(vec);
            }
        }    
    }

    void read_mnist_test_label(const std::string& path, 
                                std::vector<Eigen::VectorXd>& data)
    {
        std::ifstream file(path, std::ios::binary);
        if (file.is_open())
        {
            int magic_number = 0;
            int number_of_items = 0;

            file.read((char*)&magic_number, sizeof(magic_number));
            magic_number = ntohl(magic_number);
            file.read((char*)&number_of_items, sizeof(number_of_items));
            number_of_items = ntohl(number_of_items);

            for (int i = 0; i < number_of_items; i++)
            {
                unsigned char temp = 0;
                file.read((char*)&temp, sizeof(temp));
                Eigen::VectorXd vec(10);
                vec.setZero();
                vec((int)temp) = 1.0;
                data.push_back(vec);
            }
        }
    }
    
        // Display an MNIST image as ASCII art
        void display_ascii_image(const VectorXd& image, int width, int height) {
            for (int y = 0; y < height; y++) {
                for (int x = 0; x < width; x++) {
                    double pixel = image(y * width + x);
                    // Map pixel values to ASCII characters based on intensity
                    char c = ' ';
                    if (pixel > 0.9) c = '#';
                    else if (pixel > 0.7) c = '+';
                    else if (pixel > 0.5) c = '*';
                    else if (pixel > 0.3) c = '-';
                    else if (pixel > 0.1) c = '.';
                    std::cout << c << c; // Print each pixel as two chars for better aspect ratio
                }
                std::cout << std::endl;
            }
        }
    
}

#endif 
