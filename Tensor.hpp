#ifndef TENSOR_HPP
#define TENSOR_HPP

#include <concepts>
#include <cstddef>
#include <limits>
#include <type_traits>

#include <array>
#include <memory>
#include <string>

#include <algorithm>
#include <cmath>

#include <stdexcept>

#include <format>
#include <iomanip>
#include <ios>
#include <ostream>
#include <sstream>

template <std::floating_point T, std::size_t... Dimensions>
struct nested_array { 
    using type = T; 
};

template <std::floating_point T, std::size_t D, std::size_t... Dimensions>
struct nested_array<T, D, Dimensions...> {
    using type = typename nested_array<T, Dimensions...>::type[D];
};

template <std::floating_point T, std::size_t... Dimensions>
using nested_array_t = typename nested_array<T, Dimensions...>::type;

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

template <std::floating_point T, std::size_t... Dimensions>
class Tensor {
public:
    static constexpr std::size_t rank = sizeof...(Dimensions);
    static constexpr std::size_t size = (std::size_t{1} * ... * Dimensions);
    static constexpr std::array<std::size_t, rank> shape = { Dimensions... };

private:
    std::unique_ptr<T[]> values = std::make_unique<T[]>(size);

    bool empty() const noexcept {
        return !values;
    }

    void emptyAccessCheck() const {
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

    static constexpr std::size_t PRINT_SUMMARY_THRESHOLD = 1000;
    static constexpr std::size_t PRINT_SUMMARY_EDGEITEMS = 3;
    static constexpr std::size_t PRINT_DEFAULT_PRECISION = 6;
    static constexpr bool SUPPRESS_SCIENTIFIC = false;

    static std::string formatElement(T value, bool useScientific, std::size_t precision, bool trim) {
        if (std::isnan(value)) {
            return "nan";
        }

        if (std::isinf(value)) {
            return value > 0 ? "inf" : "-inf";
        }

        std::ostringstream buffer;
        buffer << std::showpoint << (useScientific ? std::scientific : std::fixed) << std::setprecision(precision) << value;

        std::string valueStr = buffer.str();

        std::size_t exponentPos = std::min(valueStr.find('e'), valueStr.size());

        if (trim) {
            std::size_t lastMeaningful = valueStr.find_last_not_of('0', exponentPos - 1);
            valueStr.erase(lastMeaningful + 1, exponentPos - lastMeaningful - 1);
        }

        return valueStr;
    }

    void getPrintElementWidths(bool useScientific, std::size_t precision, bool trim, 
        std::size_t& integerWidth, 
        std::size_t& fractionWidth, 
        std::size_t& mantissaWidth, 
        std::size_t& exponentWidth) const {

        integerWidth = 0;
        fractionWidth = 0;
        mantissaWidth = 0;
        exponentWidth = 0;

        std::size_t specialWidth = 0;

        for (std::size_t i = 0; i < size; ++i) {
            std::string valueStr = formatElement(values[i], useScientific, precision, trim);
            std::size_t valueLength = valueStr.size();

            if (!std::isfinite(values[i])) {
                specialWidth = std::max(specialWidth, valueLength);
                continue;
            }

            std::size_t pointPos = valueStr.find('.');
            std::size_t exponentPos = valueStr.find('e');

            integerWidth = std::max(integerWidth, pointPos);

            if (useScientific) {
                mantissaWidth = std::max(mantissaWidth, exponentPos - pointPos - 1);
                exponentWidth = std::max(exponentWidth, valueLength - exponentPos - 2);
            } else {
                fractionWidth = std::max(fractionWidth, valueLength - pointPos);
            }
        }

        if (useScientific) {
            fractionWidth = 1 + mantissaWidth + 2 + exponentWidth;
        }

        if (integerWidth + fractionWidth < specialWidth) {
            integerWidth = specialWidth - fractionWidth;
        }
    }

    void print(std::ostream& os) const {
        T minAbs = std::numeric_limits<T>::infinity();
        T maxAbs = 0.0;

        for (std::size_t i = 0; i < size; ++i) {
            T valueAbs = std::abs(values[i]);

            if (valueAbs != 0.0 && std::isfinite(valueAbs)) {
                minAbs = std::min(minAbs, valueAbs);
                maxAbs = std::max(maxAbs, valueAbs);
            }
        }

        bool useScientific = maxAbs > 0.0 && (maxAbs >= 1e8 || (!SUPPRESS_SCIENTIFIC && (minAbs < 1e-4 || maxAbs / minAbs > 1000)));

        std::size_t integerWidth; 
        std::size_t fractionWidth;
        std::size_t mantissaWidth;
        std::size_t exponentWidth;
        std::size_t precision;

        std::streamsize streamPrecision = os.precision();

        precision = { streamPrecision >= 0 ? static_cast<std::size_t>(streamPrecision) : PRINT_DEFAULT_PRECISION };

        std::ios_base::fmtflags ff = os.flags() & std::ios::floatfield;
        bool trim = true;

        if (ff == std::ios::fixed) {
            useScientific = false;
            trim = false;
        } else if (ff == std::ios::scientific) {
            useScientific = true;
            trim = false;
        }   

        getPrintElementWidths(useScientific, precision, trim, integerWidth, fractionWidth, mantissaWidth, exponentWidth);

        printVerbose(os, useScientific, precision, trim, integerWidth, fractionWidth, mantissaWidth, exponentWidth, 0, 0, 0);
    }

    void printVerbose(std::ostream& os, bool useScientific, std::size_t precision, bool trim,
        std::size_t integerWidth, 
        std::size_t fractionWidth, 
        std::size_t mantissaWidth, 
        std::size_t exponentWidth, 
        std::size_t axis, std::size_t offset, std::size_t indent) const {

        if (rank == 0) {
            os << formatElement(values[0], useScientific, precision, trim);
            return;
        }

        os << '[';

        bool shouldSummarize = size > PRINT_SUMMARY_THRESHOLD;

        if (axis == rank - 1) {
            for (std::size_t i = 0; i < shape[axis]; ++i) {
                if (i > 0) {
                    os << ' ';
                }

                if (shouldSummarize && shape[axis] > PRINT_SUMMARY_EDGEITEMS * 2 && i == PRINT_SUMMARY_EDGEITEMS) {
                    os << "...";

                    i = shape[axis] - PRINT_SUMMARY_EDGEITEMS - 1;
                    continue;
                }

                std::string valueStr = formatElement(values[offset + i], useScientific, precision, trim);

                if (!std::isfinite(values[offset + i])) {
                    os << std::string(integerWidth + fractionWidth - valueStr.size(), ' ') << valueStr;
                    continue;
                }

                std::size_t pointPos = valueStr.find('.');
                std::size_t exponentPos = valueStr.find('e');

                if (useScientific) {
                    valueStr.insert(exponentPos + 2, exponentWidth - (valueStr.size() - exponentPos - 2), '0');
                    valueStr.insert(exponentPos, mantissaWidth - (exponentPos - pointPos - 1), '0');
                }

                os << std::string(integerWidth - pointPos, ' ') << valueStr << std::string(fractionWidth - (valueStr.size() - pointPos), ' ');
            }
        } else {
            std::size_t stride = 1;

            for (std::size_t i = axis + 1; i < rank; ++i) {
                stride *= shape[i];
            }

            for (std::size_t i = 0; i < shape[axis]; ++i) {
                if (i > 0) {
                    os << std::string(rank - indent - 1, '\n') << std::string(indent + 1, ' ');
                }

                if (shouldSummarize && shape[axis] > PRINT_SUMMARY_EDGEITEMS * 2 && i == PRINT_SUMMARY_EDGEITEMS) {
                    os << "...";

                    i = shape[axis] - PRINT_SUMMARY_EDGEITEMS - 1;
                    continue;
                }

                printVerbose(os, useScientific, precision, trim, integerWidth, fractionWidth, mantissaWidth, exponentWidth, axis + 1, offset + i * stride, indent + 1);
            }
        }

        os << ']';
    }

public:
    Tensor() = default;

    Tensor(const nested_array_t<T, Dimensions...>& rawValues) {
        T* out = values.get();
        flatten(rawValues, out);
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

    static Tensor full(T value) {
        Tensor tensor{};

        std::fill_n(tensor.values.get(), size, value);

        return tensor;
    }

    static Tensor ones() {
        return full(1.0);
    }

    static Tensor zeroes() {
        return full(0.0);
    }

    T& operator[](decltype(Dimensions)... indices) {
        emptyAccessCheck();

        std::size_t offset = calculateOffset(indices...);

        return values[offset];
    }

    const T& operator[](decltype(Dimensions)... indices) const {
        emptyAccessCheck();

        std::size_t offset = calculateOffset(indices...);

        return values[offset];
    }

    friend std::ostream& operator<<(std::ostream& os, const Tensor& tensor) {
        os << "Tensor<";

        for (std::size_t i = 0; i < tensor.rank; ++i) {
            if (i > 0) {
                os << ", ";
            }

            os << tensor.shape[i];
        }

        os << ">";

        if (tensor.empty()) {
            os << " (empty)";
        } else {
            os << "\n";
            tensor.print(os);
        }

        return os;
    }

    T max() const {
        emptyAccessCheck();

        T maxValue = -std::numeric_limits<T>::infinity();

        for (std::size_t i = 0; i < size; ++i) {
            maxValue = std::max(maxValue, values[i]);
        }

        return maxValue;
    }

    T min() const {
        emptyAccessCheck();

        T minValue = std::numeric_limits<T>::infinity();

        for (std::size_t i = 0; i < size; ++i) {
            minValue = std::min(minValue, values[i]);
        }

        return minValue;
    }
};

#endif // TENSOR_HPP