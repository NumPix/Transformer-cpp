#include "tensor/tensor.hpp"
#include "tensor/tensor_io.hpp"
#include <iostream>
#include <iomanip>

int main(int argc, char** argv) {
    auto T = tensor::Tensor<float, 1, 2, 3>::full(2);
    std::cout << std::scientific << std::setprecision(8) << T;

    return 0;
}