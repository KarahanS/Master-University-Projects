#include "submission/exercise_82.h"
#include "submission/point.h"
#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

int main() {
  // Input data
  std::vector<Point> points = {Point(8, 0), Point(-1, 12), Point(-2, -1),
                               Point(1, 2), Point(2, 1),   Point(2, -4)};
  for (std::cout << "points: "; const auto &point : points) {
    std::cout << point << " ";
  }
  std::cout << "\n";

  // Sorting
  std::vector<Point> result_x = sort_x(points);
  for (std::cout << "sorted (x): "; const auto &point : result_x) {
    std::cout << point << " ";
  }
  std::cout << "\n";

  std::vector<Point> result_y = sort_y(points);
  for (std::cout << "sorted (y): "; const auto &point : result_y) {
    std::cout << point << " ";
  }
  std::cout << "\n";

  // median
  std::cout << "Median:" << median(points) << "\n";

  return EXIT_SUCCESS;
}
