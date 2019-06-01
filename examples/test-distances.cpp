
#include "neuralfield.hpp"
#include <fstream>

int main(int argc, char* argv[]) {

    if(argc != 2) {
        std::cerr << "Usage : " << argv[0] << " N " << std::endl;
        ::exit(-1);
    }

    int N = std::atoi(argv[1]);
    std::array<int, 1> shape = {N};
    auto dist1D = neuralfield::distances::make_euclidean_1D(shape, true);

    std::ofstream outfile("distances.data");
    for(int i = 0 ; i < N ; ++i) {
        double d = dist1D(i, N/2);
        outfile << i << "\t" << d << std::endl;
    }
    outfile.close();
}
