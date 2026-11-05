#ifndef TENSOR_MATH_HPP
#define TENSOR_MATH_HPP

#include "tensor.hpp"

#include <cmath>
#include <utility>

namespace tensor::detail {
    template<std::floating_point T, std::invocable<T> F, std::size_t... Dimensions>
    Tensor<T, Dimensions...> elementwiseUnary(const Tensor<T, Dimensions...>& t, F func) {
        if (t.empty()) {
            throw std::logic_error("can't apply function to empty tensor");
        }

        Tensor<T, Dimensions...> tOut;
        std::transform(t.data(), t.data() + t.size, tOut.data(), func);

        return tOut;
    }

    template<std::floating_point T, std::invocable<T> F, std::size_t... Dimensions>
    Tensor<T, Dimensions...> elementwiseUnary(Tensor<T, Dimensions...>&& t, F func) {
        if (t.empty()) {
            throw std::logic_error("can't apply function to empty tensor");
        }

        std::transform(t.data(), t.data() + t.size, t.data(), func);

        return std::move(t);
    }
}

namespace tensor {

#define TENSOR_FUNCTION_UNARY(func)                                                         \
    template<std::floating_point T, std::size_t... Dimensions>                              \
    inline Tensor<T, Dimensions...> func(const Tensor<T, Dimensions...>& t) {               \
        return detail::elementwiseUnary(t, [](T t) { return std::func(t); });                \
    }                                                                                       \
                                                                                            \
    template<std::floating_point T, std::size_t... Dimensions>                              \
    inline Tensor<T, Dimensions...> func(Tensor<T, Dimensions...>&& t) {                    \
        return detail::elementwiseUnary(std::move(t), [](T t) { return std::func(t); });     \
    }                                                                           

    TENSOR_FUNCTION_UNARY(abs)
    TENSOR_FUNCTION_UNARY(sqrt)
    TENSOR_FUNCTION_UNARY(cbrt)

    TENSOR_FUNCTION_UNARY(exp)
    TENSOR_FUNCTION_UNARY(exp2)
    TENSOR_FUNCTION_UNARY(log)
    TENSOR_FUNCTION_UNARY(log2)
    TENSOR_FUNCTION_UNARY(log10)

    TENSOR_FUNCTION_UNARY(sin)
    TENSOR_FUNCTION_UNARY(cos)
    TENSOR_FUNCTION_UNARY(tan)
    TENSOR_FUNCTION_UNARY(asin)
    TENSOR_FUNCTION_UNARY(acos)
    TENSOR_FUNCTION_UNARY(atan)

    TENSOR_FUNCTION_UNARY(sinh)
    TENSOR_FUNCTION_UNARY(cosh)
    TENSOR_FUNCTION_UNARY(tanh)
    TENSOR_FUNCTION_UNARY(asinh)
    TENSOR_FUNCTION_UNARY(acosh)
    TENSOR_FUNCTION_UNARY(atanh)

    TENSOR_FUNCTION_UNARY(floor)
    TENSOR_FUNCTION_UNARY(ceil)
    TENSOR_FUNCTION_UNARY(round)
    TENSOR_FUNCTION_UNARY(trunc)
    

#undef TENSOR_FUNCTION_UNARY

}

#endif // TENSOR_MATH_HPP