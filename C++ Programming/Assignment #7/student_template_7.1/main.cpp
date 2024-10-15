#include "submission/exercise_07.h"
#include <cstdlib>
#include <iostream>
#include <vector>

int main() {
  ////// Square
  std::cout << "square<char>(5) = " << static_cast<int>(square<char>(5))
            << " type: " << typeid(square<char>(5)).name() << "\n";
  std::cout << "square<int>(5) = " << square<int>(5)
            << " type: " << typeid(square<int>(5)).name() << "\n";
  std::cout << "square<float>(5) = " << square<float>(5)
            << " type: " << typeid(square<float>(5)).name() << "\n";
  std::cout << "square<double>(5) = " << square<double>(5)
            << " type: " << typeid(square<double>(5)).name() << "\n";

  std::cout << "square<char>(5.5) = " << static_cast<int>(square<char>(5.5))
            << " type: " << typeid(square<char>(5.5)).name() << "\n";
  std::cout << "square<int>(5.5) = " << square<int>(5.5)
            << " type: " << typeid(square<int>(5.5)).name() << "\n";
  std::cout << "square<float>(5.5) = " << square<float>(5.5)
            << " type: " << typeid(square<float>(5.5)).name() << "\n";
  std::cout << "square<double>(5.5) = " << square<double>(5.5)
            << " type: " << typeid(square<double>(5.5)).name() << "\n\n";

  ////// Halve
  std::cout << "halve<char>(5) = " << halve<char>(5)
            << " type: " << typeid(halve<char>(5)).name() << "\n";
  std::cout << "halve<int>(5) = " << halve<int>(5)
            << " type: " << typeid(halve<int>(5)).name() << "\n";
  std::cout << "halve<float>(5) = " << halve<float>(5)
            << " type: " << typeid(halve<float>(5)).name() << "\n";
  std::cout << "halve<double>(5) = " << halve<double>(5)
            << " type: " << typeid(halve<double>(5)).name() << "\n";

  std::cout << "halve<char>(5.5) = " << halve<char>(5.5)
            << " type: " << typeid(halve<char>(5.5)).name() << "\n";
  std::cout << "halve<int>(5.5) = " << halve<int>(5.5)
            << " type: " << typeid(halve<int>(5.5)).name() << "\n";
  std::cout << "halve<float>(5.5) = " << halve<float>(5.5)
            << " type: " << typeid(halve<float>(5.5)).name() << "\n";
  std::cout << "halve<double>(5.5) = " << halve<double>(5.5)
            << " type: " << typeid(halve<double>(5.5)).name() << "\n\n";

  ////// Add
  std::cout << "add<char>(40, 2) = " << static_cast<int>(add<char>(40, 2))
            << " type: " << typeid(add<char>(40, 2)).name() << "\n";
  std::cout << "add<int>(40, 2) = " << static_cast<int>(add<int>(40, 2))
            << " type: " << typeid(add<int>(40, 2)).name() << "\n";
  std::cout << "add<float>(40, 2) = " << static_cast<float>(add<float>(40, 2))
            << " type: " << typeid(add<float>(40, 2)).name() << "\n";
  std::cout << "add<double>(40, 2) = "
            << static_cast<double>(add<double>(40, 2))
            << " type: " << typeid(add<double>(40, 2)).name() << "\n";

  std::cout << "add<char>(39.5, 2.5) = "
            << static_cast<int>(add<char>(39.5, 2.5))
            << " type: " << typeid(add<char>(39.5, 2.5)).name() << "\n";
  std::cout << "add<int>(39.5, 2.5) = " << static_cast<int>(add<int>(39.5, 2.5))
            << " type: " << typeid(add<int>(39.5, 2.5)).name() << "\n";
  std::cout << "add<float>(39.5, 2.5) = "
            << static_cast<float>(add<float>(39.5, 2.5))
            << " type: " << typeid(add<float>(39.5, 2.5)).name() << "\n";
  std::cout << "add<double>(39.5, 2.5) = "
            << static_cast<double>(add<double>(39.5, 2.5))
            << " type: " << typeid(add<double>(39.5, 2.5)).name() << "\n\n";

  /////// Reduce
  // data to work with
  std::vector<int> input_data_int = {1, 2, 3, 4};
  std::vector<float> input_data_float = {1.23f, 23.45f, 33.3456f};
  // sum
  int sum_int = reduce<int>(add<int>, input_data_int, 0);
  float sum_float = reduce<float>(add<float>, input_data_float, 0.f);
  std::cout << "Sum reduction int: " << sum_int << "\n";
  std::cout << "Sum reduction float: " << sum_float << "\n\n";

  // product
  int prod_int = reduce<int>(multiply<int>, input_data_int, 1);
  float prod_float = reduce<float>(multiply<float>, input_data_float, 1.f);
  std::cout << "Product reduction int: " << prod_int << "\n";
  std::cout << "Product reduction float: " << prod_float << "\n\n";

  ////// Map
  // square
  auto map_sq_result_int = map<int>(square<int>, input_data_int);
  auto map_sq_result_float =
      map<float>(square<float>, input_data_float);
  std::cout << "Int map(square) result:\n";
  for (auto value : map_sq_result_int) {
    std::cout << value << ", ";
  }
  std::cout << "\n";
  std::cout << "Float map(square) result:\n";
  for (auto value : map_sq_result_float) {
    std::cout << value << ", ";
  }
  std::cout << "\n";

  // halve
  auto map_halve_result_int = map<int>(halve<int>, input_data_int);
  auto map_halve_result_float =
      map<float>(halve<float>, input_data_float);
  std::cout << "Int map(halve) result:\n";
  for (auto value : map_halve_result_int) {
    std::cout << value << ", ";
  }
  std::cout << "\n";
  std::cout << "Float map(halve) result:\n";
  for (auto value : map_halve_result_float) {
    std::cout << value << ", ";
  }
  std::cout << "\n";


  return EXIT_SUCCESS;
}
