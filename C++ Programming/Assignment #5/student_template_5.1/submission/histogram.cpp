#include "histogram.h"

#include <limits>
#include <map>
#include <unordered_map>
#include <string>
#include <utility>
#include <vector>
#include <set>
#include <cfloat>

Histogram::Histogram(const std::vector<std::string> &words) {
  std::unordered_map<std::string, double> word_count;

  for (const std::string &word : words) word_count[word]++;

  // normalize word count to probabilities
  int total_words = int(words.size());
  for (auto &pair: word_count) pair.second = total_words == 0 ? 0 : (pair.second / total_words);
  for (const auto &pair: word_count) histogram[pair.first] = pair.second;
  
}

size_t Histogram::size() const {
  return histogram.size();
}

bool Histogram::contains(const std::string &word) const {
  return histogram.find(word) != histogram.end();
}

double Histogram::probability(const std::string &word) const {
  if (!contains(word)) return 0.0;
  else return histogram.at(word);
}

std::vector<std::pair<double, std::string>> Histogram::most_common_words(unsigned int n_words) const {
  std::multimap<double, std::string> prob_to_word;
  std::vector<std::pair<double, std::string>> common_words;

  for (const auto &pair : histogram) prob_to_word.insert({pair.second, pair.first});
  // iterate from end to beginning
  for (auto it = prob_to_word.rbegin(); it != prob_to_word.rend(); it++) {
    common_words.push_back(*it);
    if (common_words.size() == n_words) {
      break;
    }
  }

  return common_words;

}
double Histogram::dissimilarity(const Histogram &other) const {
    std::set<std::string> this_words;
    std::set<std::string> other_words;

    // Populate this_words and other_words with the keys of the histograms
    for (const auto& pair: histogram) this_words.insert(pair.first);
    for (const auto& pair: other.histogram) other_words.insert(pair.first);
    if (this_words.empty() || other_words.empty()) return 2;
  
    std::set<std::string> shared_words;
    for (const auto& word: this_words) {
        if (other_words.find(word) != other_words.end()) {
            shared_words.insert(word);
        }
    }

    double comp1 = 1.0 - (double(shared_words.size()) / this_words.size());
    double comp2 = 1.0 - (double(shared_words.size()) / other_words.size());
    double comp3 = 0.0;
    
    for (const auto& word: shared_words) {
      double count_diff = std::abs(probability(word) - other.probability(word));
      comp3 += count_diff;
    }
    return comp1 + comp2 + comp3;

}

size_t Histogram::closest(const std::vector<Histogram> &candidates) const {
    size_t closest_index = 0;
    double closest_distance = DBL_MAX;
    for (size_t i = 0; i < candidates.size(); i++) {
        double distance = dissimilarity(candidates[i]);
        if (distance < closest_distance) {
            closest_distance = distance;
            closest_index = i;
        }
    }
    return closest_index;
}