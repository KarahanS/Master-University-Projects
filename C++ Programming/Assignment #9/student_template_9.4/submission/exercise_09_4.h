#pragma once

#include <vector>

/**
 * @brief Runs the code of exercise09.
 *
 */
void run_exercise09();

std::vector<int> hist_critical(int N, int bins);

std::vector<int> hist_element_lock(int N, int bins);

std::vector<int> hist_lockfree(int N, int bins);

