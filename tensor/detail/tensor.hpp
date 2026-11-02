#ifndef TENSOR_HPP
#define TENSOR_HPP

#include <concepts>
#include <cstddef>
#include <type_traits>
#include <limits>

#include <array>
#include <memory>
#include <algorithm>
#include <stdexcept>
#include <format>

namespace tensor::detail {

template <std::floating_point T, std::size_t... Dimensions>
struct NestedArray { 
    using type = T; 
};

template <std::floating_point T, std::size_t D, std::size_t... Dimensions>
struct NestedArray<T, D, Dimensions...> {
    using type = typename NestedArray<T, Dimensions...>::type[D];
};

template <std::floating_point T, std::size_t... Dimensions>
using nested_array_t = typename NestedArray<T, Dimensions...>::type;

template <std::floating_point T, typename A>
constexpr void flatten(const A& a, T*& out) {
    if constexpr (std::is_array_v<A>) {
        for (const auto& element : a) {
            flatten(element, out);
        }
    } else {
        *out++ = a;
    }
}

}

namespace tensor {

template <std::floating_point T, std::size_t... Dimensions>
class Tensor {
public:
    static constexpr std::size_t rank = sizeof...(Dimensions);
    static constexpr std::size_t size = (std::size_t{1} * ... * Dimensions);
    static constexpr std::array<std::size_t, rank> shape = { Dimensions... };

private:
    std::unique_ptr<T[]> values = std::make_unique<T[]>(size);

    void checkNotEmpty() const {
        if (empty()) {
            throw std::logic_error("can't access empty Tensor");
        }
    }

    static std::size_t calculateOffset(decltype(Dimensions)... indices) {
        const std::array<std::size_t, rank> indicesArray = { indices... };

        std::size_t offset = 0;

        for (std::size_t n = 0; n < rank; ++n) {
            if (indicesArray[n] >= shape[n]) {
                throw std::out_of_range(
                    std::format(
                        "Tensor index {} is {}, but dimension {} has size {}",
                        n, indicesArray[n], n, shape[n]
                    )
                );
            }

            offset = offset * shape[n] + indicesArray[n];
        }

        return offset;
    }

public:
    Tensor() = default;

    Tensor(const detail::nested_array_t<T, Dimensions...>& rawValues) {
        T* out = values.get();
        detail::flatten(rawValues, out);
    }

    Tensor(const Tensor& other) : values(other.empty() ? nullptr : std::make_unique_for_overwrite<T[]>(size)) {
        if (!other.empty()) {
            std::copy_n(other.values.get(), size, values.get());
        }
    }

    Tensor& operator=(const Tensor& other) {
        if (this == &other) {
            return *this;
        }

        if (other.empty()) {
            values.reset();

            return *this;
        }

        if (empty()) {
            values = std::make_unique_for_overwrite<T[]>(size);
        }

        std::copy_n(other.values.get(), size, values.get());

        return *this;
    }

    Tensor(Tensor&&) noexcept = default;
    Tensor& operator=(Tensor&&) noexcept = default;

    bool empty() const noexcept {
        return !values;
    }

    T* data() noexcept {
        return values.get();
    }

    const T* data() const noexcept {
        return values.get();
    }

    static Tensor full(T value) {
        Tensor tensor{};

        std::fill_n(tensor.values.get(), size, value);

        return tensor;
    }

    static Tensor ones() {
        return full(T{1});
    }

    static Tensor zeros() {
        return full(T{0});
    }

    T& operator[](decltype(Dimensions)... indices) {
        checkNotEmpty();

        std::size_t offset = calculateOffset(indices...);

        return values[offset];
    }

    const T& operator[](decltype(Dimensions)... indices) const {
        checkNotEmpty();

        std::size_t offset = calculateOffset(indices...);

        return values[offset];
    }

    T max() const {
        checkNotEmpty();

        T maxValue = -std::numeric_limits<T>::infinity();

        for (std::size_t i = 0; i < size; ++i) {
            maxValue = std::max(maxValue, values[i]);
        }

        return maxValue;
    }

    T min() const {
        checkNotEmpty();

        T minValue = std::numeric_limits<T>::infinity();

        for (std::size_t i = 0; i < size; ++i) {
            minValue = std::min(minValue, values[i]);
        }

        return minValue;
    }
};

}

#endif // TENSOR_HPP