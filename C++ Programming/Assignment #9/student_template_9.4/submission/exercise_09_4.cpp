#include "exercise_09_4.h"

#include <chrono>
#include <cmath>
#include <fstream> // for ofstream
#include <iomanip>
#include <iostream>
#include <numeric>
#include <omp.h>
#include <random>
#include <set>
#include <vector>

/*
Oguz Ata Cal    - 6661014  
Karahan Saritas - 6661689
*/


using namespace std;

using std::chrono::duration_cast;
using std::chrono::high_resolution_clock;
using std::chrono::milliseconds;

// values near the mean are the most likely
// standard deviation affects the dispersion of generated values from the mean

/**
  Simple timer class which starts measuring when constructed and tells the
  elapsed time when destructed. Give it a name when constructing (so it can tell
  the user what was being timed). Call stop() if you want it to stop at a
  specific point.
*/
class Timer {
public:
  Timer(const std::string name = "unknown computation")
      : t_start_(std::chrono::high_resolution_clock::now()), t_sec(0) {
    std::cout << name.c_str() << " running ...\n";
  }

  double
  stop() { // get time in ms and divide by 1000 ( to get 3-digit precision)
    const auto t_end = high_resolution_clock::now();
    t_sec = static_cast<double>(duration_cast<milliseconds>(t_end - t_start_).count()) / 1e3;
    return t_sec;
  }

  ~Timer() {
    if (t_sec == 0)
      stop();
    std::cout << "Time: " << t_sec << " s\n\n";
  }

private:
  std::chrono::time_point<std::chrono::system_clock> t_start_; // start time
  double t_sec; // duration in seconds (three 3 digits precision)
};

int sample_from_normal_distribution() {
  static thread_local std::normal_distribution<> d{50, 10};
  static thread_local std::random_device rd{};
  static thread_local std::mt19937 gen{rd()};
  int bin = static_cast<int>(std::round(d(gen)));
  if (bin >= 100 || bin <= 0)
    return 0;
  return bin;
}

// print the histogram with ascii chars (for debugging)
void print_hist(const vector<int> &h) {
  for (size_t i = 0; i != h.size(); ++i) {
    std::cout << std::setw(2) << i << '\t'
              << std::string(static_cast<size_t>(h[i] / 10000), '*')
              << '\n';
  }
}

// the default implementation without any parallelization
std::vector<int> hist_serial(int N, int bins) {
  std::vector<int> hist(static_cast<size_t>(bins), 0);

  Timer t("Histogram calculation without parallelization.");
  for (int i = 0; i < N; ++i) {
    int bin = sample_from_normal_distribution();
    ++hist[static_cast<size_t>(bin)];
  }
  t.stop();
  return hist;
}

// use a critial section to synchronize access to the hist vector
std::vector<int> hist_critical(int N, int bins) {
  std::vector<int> hist(static_cast<size_t>(bins), 0);

  Timer t("Histogram calculation using a critical section.");
#pragma omp parallel for
  for (int i = 0; i < N; ++i) {
    int bin = sample_from_normal_distribution();
    #pragma omp critical   // single thread access to the histogram
    ++hist[static_cast<size_t>(bin)];
  }
  t.stop();
  return hist;
}

// use one lock/mutex for every bin to synchronize access to the hist vector
// hint: create a vector of locks and initialize them.
// Before accessing a bin you lock the corresponding lock and unlock it
// afterwards
std::vector<int> hist_element_lock(int N, int bins) {
  std::vector<int> hist(static_cast<size_t>(bins), 0);
  std::vector<omp_lock_t> locks(static_cast<size_t>(bins)); // vector of omp_lock_t

  Timer t("Histogram with bin-wise locks");
#pragma omp parallel for
  for (int i = 0; i < N; ++i) {
    int bin = sample_from_normal_distribution();
    size_t idx = static_cast<size_t>(bin);
    omp_set_lock(&locks[idx]); // lock the bin
    ++hist[static_cast<size_t>(bin)];
    omp_unset_lock(&locks[idx]); // unlock the bin
  }
  t.stop();

  return hist;
}

// Try to understand following lock-free version and add the last step of
// merging hists without any locking at all.
std::vector<int> hist_lockfree(int N, int bins) {
  std::vector<int> hist(static_cast<size_t>(bins), 0);

  int thread_count;
  std::vector<int> hist_large;
  Timer t("Histogram lock-free implementation.");
#pragma omp parallel
  {
#pragma omp single
    {
      thread_count = omp_get_num_threads();
      hist_large.resize(static_cast<size_t>(thread_count * bins)); // one section for each thread
    }
    int offset = omp_get_thread_num() * bins;
#pragma omp for
    for (int i = 0; i < N; ++i) {
      ++hist_large[static_cast<size_t>(offset + sample_from_normal_distribution())];
    }
    // find a lock-free way to merge bins (hint: iterate over bins in the outer loop and aggregate in the inner loop)
#pragma omp for // parallelize the merging of the bins
        for (int i = 0; i < bins; i++) {  // different threads will work on different bins
        
        // following for loop will be executed by all threads (pragma omp for parallelizes the outer loop)
            for (int t = 0; t < thread_count; t++) {   
                hist[i] += hist_large[static_cast<size_t>(t * bins + i)];
            }
        }
    }
  
  t.stop();
  return hist;
}

void run_exercise09() {
  constexpr int N = 1e7;
  constexpr int bins = 1e2;

  // burn in: starting the OpenMP thread pool has a slight overhead so do that
  // now. Otherwise the first method we benchmark would also include the startup
  // time.
  std::cout << "burn-in ...";
#pragma omp parallel
  {
    auto t_start_ = high_resolution_clock::now();
    auto t_end = high_resolution_clock::now();
    while (duration_cast<milliseconds>(t_end - t_start_).count() < 1000) {
      t_end = high_resolution_clock::now();
    }
  }
  std::cout << "done\n";

  // serial histogram computation
  auto serial_hist = hist_serial(N, bins);
  print_hist(serial_hist);
  int sum = std::accumulate(serial_hist.begin(), serial_hist.end(), 0);
  std::cout << "serial hist (sum): " << sum << std::endl;

  // critical histogram computation
  auto critical_hist = hist_critical(N, bins);
  print_hist(critical_hist);
  sum = std::accumulate(critical_hist.begin(), critical_hist.end(), 0);
  std::cout << "critical hist (sum): " << sum << std::endl;

  auto lock_hist = hist_element_lock(N, bins);
  print_hist(lock_hist);
  sum = std::accumulate(lock_hist.begin(), lock_hist.end(), 0);
  std::cout << "lock hist (sum): " << sum << std::endl;

  auto lockfree_hist = hist_lockfree(N, bins);
  print_hist(lockfree_hist);
  sum = std::accumulate(lockfree_hist.begin(), lockfree_hist.end(), 0);
  std::cout << "lockfree hist (sum): " << sum << std::endl;
}
