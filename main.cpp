#include "tensor/tensor"
#include "tensor/tensor_io"
#include <iostream>
#include <iomanip>

int main(int argc, char** argv) {
    auto T = tensor::Tensor<float, 4096, 4096>::full(2);
    T = tensor::exp(T);

    std::cout << std::scientific << std::setprecision(8) << T;

    return 0;
}