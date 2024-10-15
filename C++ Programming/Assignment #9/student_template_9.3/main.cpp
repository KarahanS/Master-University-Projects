#include "submission/exercise_09_3.h"
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

using std::chrono::duration;
using std::chrono::high_resolution_clock;
using std::chrono::milliseconds;

using namespace std;

int main() {
  // generate vector of random numbers
  vector<double> v_serial_sort = get_vector_of_random_numbers();
  vector<double> v_parallel_sort = v_serial_sort;

  ////////////////////////////////////////////////////
  // serial and parallel sort
  auto timer_start = high_resolution_clock::now();
  sequential_sort(v_serial_sort);
  auto timer_stop = high_resolution_clock::now();

  duration<double, std::milli> duration_ms = (timer_stop - timer_start);
  double ms_sort_serial = duration_ms.count();
  std::cout << "Serial sort executed in: " << ms_sort_serial << std::endl;

  timer_start = high_resolution_clock::now();
  parallel_sort(v_parallel_sort);
  timer_stop = high_resolution_clock::now();

  duration_ms = (timer_stop - timer_start);
  double ms_sort_parallel = duration_ms.count();
  std::cout << "Parallel sort executed in: " << ms_sort_parallel << std::endl;

  double diff_sort = ms_sort_serial - ms_sort_parallel;
  std::cout << "Parallel sort was " << diff_sort << " ms faster." << std::endl;

  ////////////////////////////////////////////////////
  // serial and parallel execution with std::transform
  vector<double> v_serial_transform = get_vector_of_random_numbers();
  vector<double> v_parallel_transform = v_serial_transform;

  auto expensive_calculation = [](double v) {
    return std::sqrt(std::sin(v) * std::cos(v));
  };

  timer_start = high_resolution_clock::now();
  sequential_transform(v_serial_transform, expensive_calculation);
  timer_stop = high_resolution_clock::now();

  duration_ms = (timer_stop - timer_start);
  double ms_transform_serial = duration_ms.count();
  std::cout << "Serial transform executed in: " << ms_transform_serial
            << std::endl;

  timer_start = high_resolution_clock::now();
  parallel_transform(v_parallel_transform, expensive_calculation);
  timer_stop = high_resolution_clock::now();

  duration_ms = (timer_stop - timer_start);
  double ms_transform_parallel = duration_ms.count();
  std::cout << "Parallel transform executed in: " << ms_transform_parallel
            << std::endl;

  double diff_transform = ms_transform_serial - ms_transform_parallel;
  std::cout << "Parallel transform was " << diff_transform << " ms faster."
            << std::endl;

  return EXIT_SUCCESS;
}
