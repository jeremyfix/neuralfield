#include <neuralfield.hpp>
#include <iostream>

#include "tests.hpp"



void test_heaviside(void) {
    {
        // Test 1D   Non toric even
        const unsigned int N = 10;
        const double weight = 0.65;
        const double radius = 0.5;
        const bool toric = false;

        std::vector<int> shape = {N};
        auto constant_input = neuralfield::function::constant(1.0, shape, "h0"); 
        auto heaviside = neuralfield::link::heaviside(weight, radius, toric, shape, "h1");
        heaviside->connect(constant_input);
        heaviside->update();

        ASSERT_EQUAL_LAYERS((*constant_input), 
                (std::array<double,N>({{1,1,1,1,1,1,1,1,1,1}})));
        ASSERT_EQUAL_LAYERS((*heaviside), 
                (std::array<double,N>({{int(N * radius + 1)*weight/double(N),
                                        int(N * radius + 2)*weight/double(N),
                                        int(N * radius + 3)*weight/double(N),
                                        int(N * radius + 4)*weight/double(N),
                                        int(N * radius + 5)*weight/double(N),
                                        int(N * radius + 5)*weight/double(N),
                                        int(N * radius + 4)*weight/double(N),
                                        int(N * radius + 3)*weight/double(N),
                                        int(N * radius + 2)*weight/double(N),
                                        int(N * radius + 1)*weight/double(N)}})));

        neuralfield::clear_current_network();
    }
    {
        // Test 1D   Non toric odd
        const unsigned int N = 11;
        const double weight = 1.0;
        const double radius = 0.5;
        const bool toric = false;

        std::vector<int> shape = {N};
        auto constant_input = neuralfield::function::constant(1.0, shape, "h0"); 
        auto heaviside = neuralfield::link::heaviside(weight, radius, toric, shape, "h1");
        heaviside->connect(constant_input);
        heaviside->update();

        ASSERT_EQUAL_LAYERS((*constant_input), 
                (std::array<double,N>({{1,1,1,1,1,1,1,1,1,1,1}})));
        ASSERT_EQUAL_LAYERS((*heaviside), 
                (std::array<double,N>({{int(N * radius + 1)*weight/double(N),
                                        int(N * radius + 2)*weight/double(N),
                                        int(N * radius + 3)*weight/double(N),
                                        int(N * radius + 4)*weight/double(N),
                                        int(N * radius + 5)*weight/double(N),
                                        int(N * radius + 6)*weight/double(N),
                                        int(N * radius + 5)*weight/double(N),
                                        int(N * radius + 4)*weight/double(N),
                                        int(N * radius + 3)*weight/double(N),
                                        int(N * radius + 2)*weight/double(N),
                                        int(N * radius + 1)*weight/double(N)}})));
        
        neuralfield::clear_current_network();
    }
    {
        // Test 1D   Toric even
        const unsigned int N = 10;
        const double weight = 0.65;
        const double radius = 0.4;
        const bool toric = true;

        std::vector<int> shape = {N};
        auto constant_input = neuralfield::function::constant(1.0, shape, "h0"); 
        auto heaviside = neuralfield::link::heaviside(weight, radius, toric, shape, "h1");
        heaviside->connect(constant_input);
        heaviside->update();

        ASSERT_EQUAL_LAYERS((*constant_input), 
                (std::array<double,N>({{1,1,1,1,1,1,1,1,1,1}})));
        double value = int(2 * N * radius + 1)*weight/double(N);
        ASSERT_EQUAL_LAYERS((*heaviside), 
                (std::array<double,N>({{value, value, value,
                                        value, value, value,
                                        value, value, value,
                                        value}})));

        neuralfield::clear_current_network();
    }
}


int main(int argc, char* argv[]) {
    test_heaviside();
}
