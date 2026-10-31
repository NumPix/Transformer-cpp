#ifndef TENSOR_HPP
#define TENSOR_HPP

#include <cstdint>
#include <cstddef>
#include <stdexcept>
#include <format>

template <std::size_t... Dimensions>
struct nested_array { 
    using type = double; 
};

template <std::size_t D, std::size_t... Dimensions>
struct nested_array<D, Dimensions...> {
    using type = typename nested_array<Dimensions...>::type[D];
};

template <std::size_t... Dimensions>
using nested_array_t = typename nested_array<Dimensions...>::type;

template <typename A>
static void flatten(const A& a, double*& out) {
    if constexpr (std::is_array_v<A>) {
        for (const auto& element : a) {
            flatten(element, out);
        }
    } else {
        *out++ = a;
    }
};

template <std::size_t... Dimensions>
class Tensor {
public:
    static constexpr std::size_t rank = sizeof...(Dimensions);
    static constexpr std::size_t size = (std::size_t{1} * ... * Dimensions);

private:
    double values[size]{}; 

public:
    constexpr Tensor() = default;

    constexpr Tensor(const nested_array_t<Dimensions...>& rawValues) {
        double* out = values;
        flatten(rawValues, out);
    }

    template<std::size_t I>
        requires (I < rank)
    static constexpr std::size_t dimension() {
        return Dimensions...[I];
    }

    static constexpr Tensor full(double value) {
        Tensor tensor{};

        for (std::size_t i = 0; i < tensor.size; ++i) {
            tensor.values[i] = value;
        }

        return tensor;
    }

    static constexpr Tensor ones() {
        return full(1.0);
    }

    static constexpr Tensor zeroes() {
        return full(0.0);
    }

    double operator[](decltype(Dimensions)... indices) {
        const std::size_t indicesArray[] = { indices... };
        const std::size_t dimensionsArray[] = { Dimensions... };

        std::size_t offset = 0;

        for (std::size_t n = 0; n < rank; ++n) {
            if (indicesArray[n] >= dimensionsArray[n]) {
                throw std::out_of_range(
                    std::format(
                        "Tensor index {} is {}, but dimension {} has size {}",
                        n, indicesArray[n], n, dimensionsArray[n]
                    )
                );
            }

            offset = offset * dimensionsArray[n] + indicesArray[n];
        }

        return values[offset];
    }
};

#endif // TENSOR_HPP