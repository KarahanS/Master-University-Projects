#pragma once

#include <vector>

std::vector<double> get_vector_of_random_numbers();
std::vector<double> sequential_sort(std::vector<double> &data);
std::vector<double> parallel_sort(std::vector<double> &data);
std::vector<double> sequential_transform(std::vector<double> &data,
                                         double (*function)(double));
std::vector<double> parallel_transform(std::vector<double> &data,
                                       double (*function)(double));