#ifndef TENSOR_MATH_HPP
#define TENSOR_MATH_HPP

#include "tensor.hpp"

#include <cmath>
#include <concepts>
#include <type_traits>
#include <utility>
#include <algorithm>
#include <stdexcept>

namespace tensor::detail {
    template<std::floating_point T, std::invocable<T> F, std::size_t... Dimensions>
    void unary(const Tensor<T, Dimensions...>& t, T* out, F func) {
        checkNotEmpty("can't apply function to empty tensor", t);

        std::transform(t.data(), t.data() + t.size, out, func);
    }

    template<std::floating_point T, std::invocable<T> F, std::size_t... Dimensions>
    Tensor<T, Dimensions...> elementwiseUnary(const Tensor<T, Dimensions...>& t, F func) {
        Tensor<T, Dimensions...> tOut;
        
        unary(t, tOut.data(), func);

        return tOut;
    }

    template<std::floating_point T, std::invocable<T> F, std::size_t... Dimensions>
    Tensor<T, Dimensions...> elementwiseUnary(Tensor<T, Dimensions...>&& t, F func) {
        unary(t, t.data(), func);
        return std::move(t);
    }


    template<std::floating_point T, std::invocable<T, T> F, std::size_t... Dimensions>
    void binary(const Tensor<T, Dimensions...>& t1, const Tensor<T, Dimensions...>& t2, T* out, F func) {
        checkNotEmpty("can't apply function to empty tensor", t1, t2);

        std::transform(t1.data(), t1.data() + t1.size, t2.data(), out, func);
    }

    template<std::floating_point T, std::invocable<T, T> F, std::size_t... Dimensions>
    Tensor<T, Dimensions...> elementwiseBinary(const Tensor<T, Dimensions...>& t1, const Tensor<T, Dimensions...>& t2, F func) {
        Tensor<T, Dimensions...> tOut;

        binary(t1, t2, tOut.data(), func);

        return tOut;
    }

    template<std::floating_point T, std::invocable<T, T> F, std::size_t... Dimensions>
    Tensor<T, Dimensions...> elementwiseBinary(Tensor<T, Dimensions...>&& t1, const Tensor<T, Dimensions...>& t2, F func) {
        binary(t1, t2, t1.data(), func);
        return std::move(t1);
    }

    template<std::floating_point T, std::invocable<T, T> F, std::size_t... Dimensions>
    Tensor<T, Dimensions...> elementwiseBinary(const Tensor<T, Dimensions...>& t1, Tensor<T, Dimensions...>&& t2, F func) {
        binary(t1, t2, t2.data(), func);
        return std::move(t2);
    }

    template<std::floating_point T, std::invocable<T, T> F, std::size_t... Dimensions>
    Tensor<T, Dimensions...> elementwiseBinary(Tensor<T, Dimensions...>&& t1, Tensor<T, Dimensions...>&& t2, F func) {
        binary(t1, t2, t1.data(), func);
        return std::move(t1);
    }
}

namespace tensor {

#define TENSOR_FUNCTION_UNARY(func)                                                                                         \
    template<std::floating_point T, std::size_t... Dimensions>                                                              \
    inline Tensor<T, Dimensions...> func(const Tensor<T, Dimensions...>& t) {                                               \
        return detail::elementwiseUnary(t, [](T x) { return std::func(x); });                                               \
    }                                                                                                                       \
                                                                                                                            \
    template<std::floating_point T, std::size_t... Dimensions>                                                              \
    inline Tensor<T, Dimensions...> func(Tensor<T, Dimensions...>&& t) {                                                    \
        return detail::elementwiseUnary(std::move(t), [](T x) { return std::func(x); });                                    \
    }            
    
#define TENSOR_OPERATOR_UNARY(op)                                                                                           \
    template<std::floating_point T, std::size_t... Dimensions>                                                              \
    Tensor<T, Dimensions...> operator op(const Tensor<T, Dimensions...>& t) {                                               \
        return detail::elementwiseUnary(t, [](T x) { return (op x); });                                                     \
    }                                                                                                                       \
                                                                                                                            \
    template<std::floating_point T, std::size_t... Dimensions>                                                              \
    Tensor<T, Dimensions...> operator op(Tensor<T, Dimensions...>&& t) {                                                    \
        return detail::elementwiseUnary(std::move(t), [](T x) { return (op x); });                                          \
    }                                                                                                                       
                                                                                                                        
#define TENSOR_OPERATOR_BINARY(op)                                                                                          \
    template<std::floating_point T, std::size_t... Dimensions>                                                              \
    Tensor<T, Dimensions...> operator op(const Tensor<T, Dimensions...>& t1, const Tensor<T, Dimensions...>& t2) {          \
        return detail::elementwiseBinary(t1, t2, [](T x1, T x2) { return x1 op x2; });                                      \
    }                                                                                                                       \
                                                                                                                            \
    template<std::floating_point T, std::size_t... Dimensions>                                                              \
    Tensor<T, Dimensions...> operator op(Tensor<T, Dimensions...>&& t1, const Tensor<T, Dimensions...>& t2) {               \
        return detail::elementwiseBinary(std::move(t1), t2, [](T x1, T x2) { return x1 op x2; });                           \
    }                                                                                                                       \
                                                                                                                            \
    template<std::floating_point T, std::size_t... Dimensions>                                                              \
    Tensor<T, Dimensions...> operator op(const Tensor<T, Dimensions...>& t1, Tensor<T, Dimensions...>&& t2) {               \
        return detail::elementwiseBinary(t1, std::move(t2), [](T x1, T x2) { return x1 op x2; });                           \
    }                                                                                                                       \
                                                                                                                            \
    template<std::floating_point T, std::size_t... Dimensions>                                                              \
    Tensor<T, Dimensions...> operator op(Tensor<T, Dimensions...>&& t1, Tensor<T, Dimensions...>&& t2) {                    \
        return detail::elementwiseBinary(std::move(t1), std::move(t2), [](T x1, T x2) { return x1 op x2; });                \
    }                                                                                                                       

#define TENSOR_OPERATOR_BINARY_SCALAR(op)                                                                                   \
    template<std::floating_point T, std::size_t... Dimensions>                                                              \
    Tensor<T, Dimensions...> operator op(const Tensor<T, Dimensions...>& t, std::type_identity_t<T> n) {                    \
        return detail::elementwiseUnary(t, [n](T x) { return x op n; });                                                    \
    }                                                                                                                       \
                                                                                                                            \
    template<std::floating_point T, std::size_t... Dimensions>                                                              \
    Tensor<T, Dimensions...> operator op(Tensor<T, Dimensions...>&& t, std::type_identity_t<T> n) {                         \
        return detail::elementwiseUnary(std::move(t), [n](T x) { return x op n; });                                         \
    }                                                                                                                       \
                                                                                                                            \
    template<std::floating_point T, std::size_t... Dimensions>                                                              \
    Tensor<T, Dimensions...> operator op(std::type_identity_t<T> n, const Tensor<T, Dimensions...>& t) {                    \
        return detail::elementwiseUnary(t, [n](T x) { return n op x; });                                                    \
    }                                                                                                                       \
                                                                                                                            \
    template<std::floating_point T, std::size_t... Dimensions>                                                              \
    Tensor<T, Dimensions...> operator op(std::type_identity_t<T> n, Tensor<T, Dimensions...>&& t) {                         \
        return detail::elementwiseUnary(std::move(t), [n](T x) { return n op x; });                                         \
    }             
    
#define TENSOR_OPERATOR_COMPOUND(op)                                                                                      \
    template<std::floating_point T, std::size_t... Dimensions>                                                              \
    Tensor<T, Dimensions...>& operator op##=(Tensor<T, Dimensions...>& t1, const Tensor<T, Dimensions...>& t2) {            \
        detail::binary(t1, t2, t1.data(), [](T x1, T x2) { return x1 op x2; });                                             \
        return t1;                                                                                                          \
    }       
    
#define TENSOR_OPERATOR_COMPOUND_SCALAR(op)                                                                               \
    template<std::floating_point T, std::size_t... Dimensions>                                                              \
    Tensor<T, Dimensions...>& operator op##=(Tensor<T, Dimensions...>& t, std::type_identity_t<T> n) {                      \
        detail::unary(t, t.data(), [n](T x) { return x op n; });                                                            \
        return t;                                                                                                           \
    }   


    TENSOR_OPERATOR_COMPOUND_SCALAR(+)
    TENSOR_OPERATOR_COMPOUND_SCALAR(-)
    TENSOR_OPERATOR_COMPOUND_SCALAR(*)
    TENSOR_OPERATOR_COMPOUND_SCALAR(/)


    TENSOR_OPERATOR_COMPOUND(+)
    TENSOR_OPERATOR_COMPOUND(-)
    TENSOR_OPERATOR_COMPOUND(*)
    TENSOR_OPERATOR_COMPOUND(/)


    TENSOR_OPERATOR_BINARY_SCALAR(+)
    TENSOR_OPERATOR_BINARY_SCALAR(-)
    TENSOR_OPERATOR_BINARY_SCALAR(*)
    TENSOR_OPERATOR_BINARY_SCALAR(/)


    TENSOR_OPERATOR_BINARY(+)
    TENSOR_OPERATOR_BINARY(-)
    TENSOR_OPERATOR_BINARY(*)
    TENSOR_OPERATOR_BINARY(/)

    
    TENSOR_OPERATOR_UNARY(+)
    TENSOR_OPERATOR_UNARY(-)


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


#undef TENSOR_OPERATOR_COMPOUND_SCALAR
#undef TENSOR_OPERATOR_COMPOUND
#undef TENSOR_OPERATOR_BINARY_SCALAR
#undef TENSOR_OPERATOR_BINARY
#undef TENSOR_OPERATOR_UNARY
#undef TENSOR_FUNCTION_UNARY


template<std::floating_point T, std::size_t... Dimensions>
T min(const Tensor<T, Dimensions...>& t) {
    detail::checkNotEmpty("can't take min of empty tensor", t);

    return *std::min_element(t.data(), t.data() + t.size);
}

template<std::floating_point T, std::size_t... Dimensions>
T max(const Tensor<T, Dimensions...>& t) {
    detail::checkNotEmpty("can't take min of empty tensor", t);

    return *std::max_element(t.data(), t.data() + t.size);
}

}

#endif // TENSOR_MATH_HPP