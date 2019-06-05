#include <neuralfield.hpp>
#include <iostream>

void test_heaviside(void) {
    // Test 1D
    std::vector<int> shape = {11};
    auto constant_input = neuralfield::function::constant(1.0, shape, "h0"); 
    auto heaviside = neuralfield::link::heaviside(1.0, 0.5, shape, "h1");
    heaviside->connect(constant_input);
    heaviside->update();

    std::cout << "h0 :" << std::endl;
    for(auto& v: *constant_input)
        std::cout << v << " " ;
    std::cout << std::endl;
    std::cout << "h1 :" << std::endl;
    for(auto& v: *heaviside)
        std::cout << v << " " ;
    std::cout << std::endl;
}


int main(int argc, char* argv[]) {
    test_heaviside();

}
