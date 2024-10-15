#include "submission/exercise_09_2.h"
#include <cstdlib>
#include <iostream>
#include <thread>
#include <map>

int main() {
  std::map<long unsigned int, size_t> diff_count;
  // start threads
  auto slow_producer_thread = std::thread(slow_producer);
  auto fast_producer_thread = std::thread(fast_producer);
  auto consumer_thread = std::thread(consumer, &diff_count);

  // wait until threads finished
  slow_producer_thread.join();
  fast_producer_thread.join();
  consumer_thread.join();

  // output results
  std::string result;
  for (auto& [diff, count] : diff_count) {
    result += std::to_string(diff) + " difference was observed " + std::to_string(count) + " times\n";
  }
  std::cout << result;
  return EXIT_SUCCESS;
}
