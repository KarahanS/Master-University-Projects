#pragma once

#include <map>
#include <string>
#include <vector>

/**
 * @brief A normalized histogram of words to represent a given text.
 *
 */
class Histogram {
public:
  /**
   * @brief Construct a new, normalized Histogram object.
   * The data is stored in the private std::map histogram.
   * @param words The list of words from which the histogram is built
   */
  Histogram(const std::vector<std::string> &words);

  /**
   * @brief Returns the size of the histogram, aka the number of unique words.
   *
   * @return size_t Number of words in the histogram.
   */
  size_t size() const;

  /**
   * @brief Checks if a word occours in the histogram
   *
   * @param word Word to check
   * @return true If word is in the histogram.
   * @return false If word is NOT in histogram.
   */
  bool contains(const std::string &word) const;

  /**
   * @brief Returns the probability for randomly selecting the given word in the text.
   *
   * @param word
   * @return double The probability for the given word.
   */
  double probability(const std::string &word) const;

  std::vector<std::pair<double, std::string>> most_common_words(unsigned int n_words) const;
  double dissimilarity(const Histogram &other) const;
  size_t closest(const std::vector<Histogram> &candidates) const;

  // TODO 5.1.c
  // TODO 5.1.d

private:
  // The histogram data is represented as map from word to probability (sorted)
  std::map<const std::string, double> histogram;
};