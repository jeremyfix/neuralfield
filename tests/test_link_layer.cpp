#include <neuralfield.hpp>
#include <iostream>

#include "tests.hpp"



void test_heaviside(void) {
    // Test 1D
    std::vector<int> shape = {10};
    auto constant_input = neuralfield::function::constant(1.0, shape, "h0"); 
    auto heaviside = neuralfield::link::heaviside(1.0, 0.5, shape, "h1");
    heaviside->connect(constant_input);
    heaviside->update();

	ASSERT_EQUAL_LAYERS((*constant_input), 
						(std::array<double,10>({{1,1,1,1,1,1,1,1,1,1}})));
   	ASSERT_EQUAL_LAYERS((*heaviside), 
						(std::array<double,10>({{.6,.7,.8,.9,1.,1.,.9,.8,.7,.6}})));
}


int main(int argc, char* argv[]) {
    test_heaviside();
}
