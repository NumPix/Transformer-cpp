#ifndef TENSOR_IO_HPP
#define TENSOR_IO_HPP

#include "tensor.hpp"

#include <limits>
#include <cmath>
#include <string>
#include <iomanip>
#include <ios>
#include <ostream>
#include <sstream>

namespace tensor::detail {

static constexpr std::size_t PRINT_SUMMARY_THRESHOLD = 1000;
static constexpr std::size_t PRINT_SUMMARY_EDGEITEMS = 3;
static constexpr std::size_t PRINT_DEFAULT_PRECISION = 6;
static constexpr bool SUPPRESS_SCIENTIFIC = false;

struct PrintFormat {
    bool useScientific = false;
    std::size_t precision = PRINT_DEFAULT_PRECISION;
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
void getPrintElementWidths(const T* values, std::size_t size, PrintFormat& format) {

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

template<std::floating_point T, std::size_t rank>
void printVerbose(std::ostream& os, const T* values, std::array<std::size_t, rank> shape, std::size_t size, const PrintFormat& format, std::size_t axis, std::size_t offset, std::size_t indent) {

    if (rank == 0) {
        os << formatElement(values[0], format);
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

            printVerbose<T, rank>(os, values, shape, size, format, axis + 1, offset + i * stride, indent + 1);
        }
    }

    os << ']';
}

template<std::floating_point T, std::size_t rank>
void print(std::ostream& os, const T* values, std::array<std::size_t, rank> shape, std::size_t size) {
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

    std::streamsize streamPrecision = os.precision();
    std::size_t precision = { streamPrecision >= 0 ? static_cast<std::size_t>(streamPrecision) : PRINT_DEFAULT_PRECISION };

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

    getPrintElementWidths(values, size, format);

    printVerbose<T, rank>(os, values, shape, size, format, 0, 0, 0);
}

}

namespace tensor {

template<std::floating_point T, std::size_t... Dimensions>
std::ostream& operator<<(std::ostream& os, const tensor::Tensor<T, Dimensions...>& tensor) {
    using tensor_t = tensor::Tensor<T, Dimensions...>;

    os << "Tensor<";

    for (std::size_t i = 0; i < tensor_t::rank; ++i) {
        if (i > 0) {
            os << ", ";
        }

        os << tensor_t::shape[i];
    }

    os << ">";

    if (tensor.empty()) {
        os << " (empty)";
    } else {
        os << "\n";
        tensor::detail::print<T, tensor_t::rank>(os, tensor.data(), tensor_t::shape, tensor_t::size);
    }

    return os;
}

}

#endif // TENSOR_IO_HPP

