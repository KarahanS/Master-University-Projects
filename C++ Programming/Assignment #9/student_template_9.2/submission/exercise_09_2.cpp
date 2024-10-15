#include "exercise_09_2.h"

/*
Oguz Ata Cal    - 6661014  
Karahan Saritas - 6661689
*/


// IO
#include <iostream>

// random number generators
#include <random>

// threading
#include <thread>
#include <mutex>
#include <condition_variable>

// data structures
#include <map>
#include <queue>

using namespace std;

std::mt19937 delay_generator;
std::mt19937 slow_generator{}, fast_generator{};
std::mutex slow_mutex{}, fast_mutex{};
std::queue<long unsigned int> slow_queue{}, fast_queue{};
std::condition_variable fast_cond, slow_cond;

thread_local unsigned int execution_count = 0;

// every 0 to 200 ms, slow_producer generates a random number and notifies other threads via condition variable.
void slow_producer() {
  while (execution_count < 20) { // every thread generates 20 numbers
	{
		std::scoped_lock<std::mutex> ul(slow_mutex);
		slow_queue.push(slow_generator() % 100 + 1);
	} // end of scope releases lock
    execution_count++;
	slow_cond.notify_one(); // DONE: tell others that a new value is available in the queue
    std::this_thread::sleep_for(std::chrono::milliseconds(delay_generator() % 200));
  }
}

// every 0 to 20 ms, fast_producer generates a random number and notifies other threads via condition variable.
void fast_producer() {
  while (execution_count < 20) { // every thread generates 20 numbers
	{
		std::scoped_lock<std::mutex> ul(fast_mutex);
		fast_queue.push(fast_generator() % 100 + 1);
	} // end of scope releases lock
    execution_count++;
    fast_cond.notify_one(); // TODO: tell others that a new value is available in the queue
    std::this_thread::sleep_for(std::chrono::milliseconds(delay_generator() % 20));
  }
}

// consumer thread function processes input once it get's available
void consumer(std::map<unsigned long, size_t> *diff_count) {
  while (execution_count < 20) {
	// TODO student: Define a unique_lock and use one of the condition variables to wait until condition (non-empty queue) is fullfiled.
	unique_lock<mutex> ul(fast_mutex);
	slow_cond.wait(ul, [] { return !slow_queue.empty(); });
	// Note: the simpler scoped_lock doesn't work here as the condition variable needs to lock/unlock the mutex while waiting.
	long unsigned int slow_data = slow_queue.front();
	slow_queue.pop();
	// TODO student: don't forget to unlock
	ul.unlock();

	// TODO: similar for the fast queue
	unique_lock<mutex> ul2(fast_mutex);
	fast_cond.wait(ul2, [] { return !fast_queue.empty(); });
	long unsigned int fast_data = fast_queue.front();
	fast_queue.pop();
	// TODO student: don't forget to unlock
	ul2.unlock();

    // count how often differences between a and b occur
    (*diff_count)[slow_data - fast_data]++;
    execution_count++;
  }
}
