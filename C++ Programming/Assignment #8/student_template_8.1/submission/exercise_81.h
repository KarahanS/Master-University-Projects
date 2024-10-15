#pragma once
#include <algorithm>
#include <vector>

/**
 * @brief Comutes the Mean Squared Error.
 * You can assume that ground_truth.size() == prediction.size()
 * 
 * @param ground_truth Vector of ground truth values.
 * @param prediction Vector or predictions.
 * @return float The MSE value.
 */
float MSE(const std::vector<float>& ground_truth,
          const std::vector<float>& prediction);

/**
 * @brief Comutes the Mean Absolute Error.
 * You can assume that ground_truth.size() == prediction.size()
 * 
 * @param ground_truth Vector of ground truth values.
 * @param prediction Vector or predictions.
 * @return float The MAE value.
 */
float MAE(const std::vector<float>& ground_truth,
          const std::vector<float>& prediction);


// TODO 8.1.c: implement zip here!
template<typename T, typename U> std::vector<std::pair<T, U>> zip(const std::vector<T>& a, const std::vector<U>& b) {
  std::vector<std::pair<T, U>> result;
  std::transform(a.begin(), a.end(), b.begin(), std::back_inserter(result), [](const T& x, const U& y) { return std::make_pair(x, y); });
  return result;
}

/**
 * @brief Generates a vector with integer values in the range [start, end[
 * Note: The list must be empty if start >= end
 * 
 * @param start The first value in the returned vector
 * @param end The upper limit of the range. This value is not included.
 * @return std::vector<int> The generated vector.
 */
std::vector<int> range(int start, int end);
