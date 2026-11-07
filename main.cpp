#include "tensor/tensor"
#include "tensor/tensor_io"
#include <iostream>
#include <iomanip>

int main(int argc, char** argv) {
    auto T1 = tensor::Tensor<float, 2, 2>::full(2);
    auto T2 = tensor::Tensor<float, 2, 2>::full(4);

    T1 /= T2;

    T1[1, 1] += 2;

    std::cout << T1 << '\n' << tensor::max(T1);

    return 0;
}