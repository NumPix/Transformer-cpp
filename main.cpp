#include "Tensor.hpp"
#include <iostream>

int main(int argc, char** argv) {
    auto T = Tensor<2, 3>({
        {1, 2, 3},
        {4, 5, 6}
    });

    for (std::size_t i = 0; i < 2; ++i) {
        for (std::size_t j = 0; j < 3; ++j) {
            std::cout << T[i, j] << " ";
        }

        std::cout << std::endl;
    }

    return 0;
}