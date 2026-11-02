#ifndef TENSOR_IO_HPP
#define TENSOR_IO_HPP

#include "tensor.hpp"

#include <cmath>
#include <string>
#include <iomanip>
#include <ios>
#include <ostream>
#include <sstream>

namespace tensor::detail {

inline constexpr std::size_t printSummaryThreshold = 1000;
inline constexpr std::size_t printSummaryEdgeItems = 3;
inline constexpr std::size_t printDefaultPrecision = 6;
inline constexpr bool suppressScientific = false;

struct PrintFormat {
    bool useScientific = false;
    std::size_t precision = printDefaultPrecision;
    bool trim = true;

    std::size_t integerWidth = 0;
    std::size_t fractionWidth = 0;
    std::size_t mantissaWidth = 0;
    std::size_t exponentWidth = 0;
};

template<std::floating_point T>
std::string formatElement(T value, const PrintFormat& format) {
    if (std::isnan(value)) {
        return "nan";
    }

    if (std::isinf(value)) {
        return value > 0 ? "inf" : "-inf";
    }

    std::ostringstream buffer;
    buffer << std::showpoint << (format.useScientific ? std::scientific : std::fixed) << std::setprecision(format.precision) << value;

    std::string valueStr = buffer.str();

    std::size_t exponentPos = std::min(valueStr.find('e'), valueStr.size());

    if (format.trim) {
        std::size_t lastMeaningful = valueStr.find_last_not_of('0', exponentPos - 1);
        valueStr.erase(lastMeaningful + 1, exponentPos - lastMeaningful - 1);
    }

    return valueStr;
}

template<std::floating_point T>
void computeElementWidths(const T* values, std::size_t size, PrintFormat& format) {

    std::size_t specialWidth = 0;

    for (std::size_t i = 0; i < size; ++i) {
        std::string valueStr = formatElement(values[i], format);
        std::size_t valueLength = valueStr.size();

        if (!std::isfinite(values[i])) {
            specialWidth = std::max(specialWidth, valueLength);
            continue;
        }

        std::size_t pointPos = valueStr.find('.');
        std::size_t exponentPos = valueStr.find('e');

        format.integerWidth = std::max(format.integerWidth, pointPos);

        if (format.useScientific) {
            format.mantissaWidth = std::max(format.mantissaWidth, exponentPos - pointPos - 1);
            format.exponentWidth = std::max(format.exponentWidth, valueLength - exponentPos - 2);
        } else {
            format.fractionWidth = std::max(format.fractionWidth, valueLength - pointPos);
        }
    }

    if (format.useScientific) {
        format.fractionWidth = 1 + format.mantissaWidth + 2 + format.exponentWidth;
    }

    if (format.integerWidth + format.fractionWidth < specialWidth) {
        format.integerWidth = specialWidth - format.fractionWidth;
    }
}

template<std::floating_point T, std::size_t Rank>
void printVerbose(std::ostream& os, const T* values, std::array<std::size_t, Rank> shape, std::size_t size, const PrintFormat& format, std::size_t axis, std::size_t offset, std::size_t indent) {

    if (Rank == 0) {
        os << formatElement(values[0], format);
        return;
    }

    os << '[';

    bool shouldSummarize = size > printSummaryThreshold;

    if (axis == Rank - 1) {
        for (std::size_t i = 0; i < shape[axis]; ++i) {
            if (i > 0) {
                os << ' ';
            }

            if (shouldSummarize && shape[axis] > printSummaryEdgeItems * 2 && i == printSummaryEdgeItems) {
                os << "...";

                i = shape[axis] - printSummaryEdgeItems - 1;
                continue;
            }

            std::string valueStr = formatElement(values[offset + i], format);

            if (!std::isfinite(values[offset + i])) {
                os << std::string(format.integerWidth + format.fractionWidth - valueStr.size(), ' ') << valueStr;
                continue;
            }

            std::size_t pointPos = valueStr.find('.');
            std::size_t exponentPos = valueStr.find('e');

            if (format.useScientific) {
                valueStr.insert(exponentPos + 2, format.exponentWidth - (valueStr.size() - exponentPos - 2), '0');
                valueStr.insert(exponentPos, format.mantissaWidth - (exponentPos - pointPos - 1), '0');
            }

            os << std::string(format.integerWidth - pointPos, ' ') << valueStr << std::string(format.fractionWidth - (valueStr.size() - pointPos), ' ');
        }
    } else {
        std::size_t stride = 1;

        for (std::size_t i = axis + 1; i < Rank; ++i) {
            stride *= shape[i];
        }

        for (std::size_t i = 0; i < shape[axis]; ++i) {
            if (i > 0) {
                os << std::string(Rank - indent - 1, '\n') << std::string(indent + 1, ' ');
            }

            if (shouldSummarize && shape[axis] > printSummaryEdgeItems * 2 && i == printSummaryEdgeItems) {
                os << "...";

                i = shape[axis] - printSummaryEdgeItems - 1;
                continue;
            }

            printVerbose<T, Rank>(os, values, shape, size, format, axis + 1, offset + i * stride, indent + 1);
        }
    }

    os << ']';
}

template<std::floating_point T, std::size_t Rank>
void print(std::ostream& os, const T* values, std::array<std::size_t, Rank> shape, std::size_t size) {
    T minAbs = std::numeric_limits<T>::infinity();
    T maxAbs = 0.0;

    for (std::size_t i = 0; i < size; ++i) {
        T valueAbs = std::abs(values[i]);

        if (valueAbs != 0.0 && std::isfinite(valueAbs)) {
            minAbs = std::min(minAbs, valueAbs);
            maxAbs = std::max(maxAbs, valueAbs);
        }
    }

    bool useScientific = maxAbs > 0.0 && (maxAbs >= 1e8 || (!suppressScientific && (minAbs < 1e-4 || maxAbs / minAbs > 1000)));

    std::streamsize streamPrecision = os.precision();
    std::size_t precision = { streamPrecision >= 0 ? static_cast<std::size_t>(streamPrecision) : printDefaultPrecision };

    std::ios_base::fmtflags ff = os.flags() & std::ios::floatfield;
    bool trim = true;

    if (ff == std::ios::fixed) {
        useScientific = false;
        trim = false;
    } else if (ff == std::ios::scientific) {
        useScientific = true;
        trim = false;
    }   

    PrintFormat format{ useScientific, precision, trim };

    computeElementWidths(values, size, format);

    printVerbose<T, Rank>(os, values, shape, size, format, 0, 0, 0);
}

}

namespace tensor {

template<std::floating_point T, std::size_t... Dimensions>
std::ostream& operator<<(std::ostream& os, const Tensor<T, Dimensions...>& tensor) {
    using TensorType = Tensor<T, Dimensions...>;

    os << "Tensor<";

    for (std::size_t i = 0; i < TensorType::rank; ++i) {
        if (i > 0) {
            os << ", ";
        }

        os << TensorType::shape[i];
    }

    os << ">";

    if (tensor.empty()) {
        os << " (empty)";
    } else {
        os << "\n";
        detail::print<T, TensorType::rank>(os, tensor.data(), TensorType::shape, TensorType::size);
    }

    return os;
}

}

#endif // TENSOR_IO_HPP

