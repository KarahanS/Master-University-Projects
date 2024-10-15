#include <cstdlib>
#include "submission/histogram.h"
#include "submission/load_books.h"
#include <iostream>
#include <string>
#include <vector>

int main() {
  // Load the books (lists of words)
  std::vector<std::string> book0 = load_book(0);
  std::vector<std::string> book1 = load_book(1);
  std::vector<std::string> book2 = load_book(2);
  std::vector<std::string> book3 = load_book(3);

  // Number of words in each book
  std::cout << "Total words in each book:\n";
  std::cout << "Book0 has " << book0.size() << " words."
            << "\n";
  std::cout << "Book1 has " << book1.size() << " words."
            << "\n";
  std::cout << "Book2 has " << book2.size() << " words."
            << "\n";
  std::cout << "Book3 has " << book3.size() << " words."
            << "\n\n";

  // TODO: this will work after you finished 5.1.a)
  // Create the histograms
  Histogram hist0(book0);
  Histogram hist1(book1);
  Histogram hist2(book2);
  Histogram hist3(book3);

  // number of words in each histogram (unique words)
  std::cout << "Unique words in each book:\n";
  std::cout << "Book0 has " << hist0.size() << " unique words."
            << "\n";
  std::cout << "Book1 has " << hist1.size() << " unique words."
            << "\n";
  std::cout << "Book2 has " << hist2.size() << " unique words."
            << "\n";
  std::cout << "Book3 has " << hist3.size() << " unique words."
            << "\n\n";


  // most common words in each book
  std::cout << "Most common words: book: probability word\n";
  for (const auto &pair : hist0.most_common_words(5)) {
    std::cout << "Book0: " << pair.first << " " << pair.second << "\n";
  }
  std::cout << "\n";
  for (const auto &pair : hist1.most_common_words(5)) {
    std::cout << "Book1: " << pair.first << " " << pair.second << "\n";
  }
  std::cout << "\n";
  for (const auto &pair : hist2.most_common_words(5)) {
    std::cout << "Book2: " << pair.first << " " << pair.second << "\n";
  }
  std::cout << "\n";
  for (const auto &pair : hist3.most_common_words(5)) {
    std::cout << "Book3: " << pair.first << " " << pair.second << "\n";
  }
  std::cout << "\n";
  

  
  // similarity between book0 and book1
  std::cout << "Difference between book0 and book1: " << hist0.dissimilarity(hist1) << "\n";
  std::cout << "Difference between book0 and book3: " << hist0.dissimilarity(hist3) << "\n\n";
  

  
  // most similar histograms
  std::cout << "Similarity of books based on histograms:\n";
  const std::string book0_candidates[] = {"Book1", "Book2", "Book3"};
  std::cout << "Book0 ist closest to "
            << book0_candidates[hist0.closest({hist1, hist2, hist3})] << "\n";

  const std::string book1_candidates[] = {"Book0", "Book2", "Book3"};
  std::cout << "Book1 ist closest to "
            << book1_candidates[hist1.closest({hist0, hist2, hist3})] << "\n";

  const std::string book2_candidates[] = {"Book0", "Book1", "Book3"};
  std::cout << "Book2 ist closest to "
            << book2_candidates[hist2.closest({hist0, hist1, hist3})] << "\n";

  const std::string book3_candidates[] = {"Book0", "Book1", "Book2"};
  std::cout << "Book3 ist closest to "
            << book3_candidates[hist3.closest({hist0, hist1, hist2})] << "\n";

  std::cout << "Dissimilarity test " <<  Histogram({"IS", "IS", "IN"}) .dissimilarity(Histogram({"IN", "OR", "IS", "A"})) << std::endl;  

  return EXIT_SUCCESS;
}
