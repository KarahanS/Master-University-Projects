#include "submission/exercise_81.h"
#include <algorithm>
#include <cstdlib>
#include <vector>
#include <iostream>
#include <string>


/**
 * @brief Example for how to use sdt::transform
 * For more information see https://en.cppreference.com/w/cpp/algorithm/transform
 */
void halve_example() {
  // Input data
  std::vector<int> input_int = {0, 1, 2, 3, 4, 5};

  // generate the result using std::transform (this is variant (1) from the documentation)
  std::vector<float> output_int;
  std::transform(input_int.begin(), input_int.end(), // input range
                 std::back_inserter(output_int),     // how to construct the output vector
                 [](int x) -> float { return static_cast<float>(x) / 2.f; });  // operator

  // print result
  for (std::cout << "halve example: "; float value : output_int) {
    std::cout << value << " ";
  }
  std::cout << "\n";
}


int main() {
  halve_example();

  std::vector<float> gt = {0.f, 1.f, 2.f, 3.f, 4.f, 5.f};
  std::vector<float> pred = {2.f, 3.f, 4.f, 5.0f, 2.f, 3.f};
  // a) MSE
  std::cout << "MSE: " << MSE(gt, pred) << "\n";
  // b) MAE
  std::cout << "MAE: " << MAE(gt, pred) << "\n";

  // c) zip
  std::vector<std::string> names = {"zero", "one", "two", "three", "four", "five"};
  auto zipped = zip<float, std::string>(gt, names);
  for (std::cout << "zipped: "; const auto &pair : zipped) {
    std::cout << pair.first << "->" << pair.second << ", ";
  }
  std::cout << "\n";
  

  // d) range
  for (std::cout << "range(-10, 10): "; int value : range(-10, 10)) {
    std::cout << value << " ";
  }
  std::cout << "\n";

  return EXIT_SUCCESS;
}
