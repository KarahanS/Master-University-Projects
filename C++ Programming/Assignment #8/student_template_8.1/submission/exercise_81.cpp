#include "exercise_81.h"
#include <algorithm>
#include <vector>
#include <numeric>


float MSE(const std::vector<float>& ground_truth,
          const std::vector<float>& prediction) {
  
  std::vector<float> squared_diffs;
  std::transform(ground_truth.begin(), ground_truth.end(), prediction.begin(),
                 std::back_inserter(squared_diffs),
                 [](float x, float y) -> float { return (x - y) * (x - y); });

  return std::accumulate(squared_diffs.begin(), squared_diffs.end(), 0.f) / static_cast<float>(ground_truth.size());
}

float MAE(const std::vector<float>& ground_truth,
          const std::vector<float>& prediction) {
  
  std::vector<float> abs_diffs;
  std::transform(ground_truth.begin(), ground_truth.end(), prediction.begin(),
                 std::back_inserter(abs_diffs),
                 [](float x, float y) -> float { return std::abs(x - y); });
  return std::accumulate(abs_diffs.begin(), abs_diffs.end(), 0.f) / static_cast<float>(ground_truth.size());
}

std::vector<int> range(int start, int end) {
    if (start >= end) {
        return {};
    }

    std::vector<int> vec;
    vec.resize(end - start);
    
    // Use std::generate with a reference to start
    std::generate(vec.begin(), vec.end(), [start]() mutable { return start++; });
    
    return vec;
}

