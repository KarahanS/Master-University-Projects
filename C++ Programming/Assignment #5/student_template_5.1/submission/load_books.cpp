#include "load_books.h"
#include <cctype>
#include <fstream>
#include <string>
#include <vector>
#include <sstream>
#include <algorithm>
#include <locale>

std::vector<std::string> load_book(const size_t book_idx) {
  // The available book files
  std::string paths[] = {"FundamentalsOfBuisnesLaw.txt",
                         "IntroductionToContracts.txt", "PythonForEveryone.txt",
                         "ThinkC++.txt"};
  if (book_idx > 3) {
    throw std::runtime_error("Only 4 books are available, use book_idx 0-3");
  }
  // Open the text file
  std::ifstream file(paths[book_idx]);
  if (!file.is_open()) {
    throw std::runtime_error("Unable to open file");
  }
  // Split book into list of words
  std::vector<std::string> result;
  // For splitting around spaces
  std::string word; // for storing each word
  std::locale loc;

  while (file >> word) {
    // Remove some 
    for (const char remove_this : ",.;:(){}[]-+_|1234567890") {
      word.erase(std::remove(word.begin(), word.end(), remove_this), word.end());
    }
    for (char &character : word) {
      character = std::toupper(character, loc);
    }
    if (word.length() > 0) {
      result.push_back(word);
    }
  }


  return result;
}
