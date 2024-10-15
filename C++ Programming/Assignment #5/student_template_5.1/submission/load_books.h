#pragma once

#include <string>
#include <vector>

/**
 * @brief Loads a list of words from a book.
 *
 * @param book_idx The index of the book. There are only 4 books available:
 * 0: Fundamentals of Buisnes Law
 * 1: Introduction to Contracts
 * 2: Python For Everyone
 * 3: Think C++
 * @return std::vector<std::string> The List of words.
 */
std::vector<std::string> load_book(const size_t book_idx);