#include "exercise_09_1.h"

#include <cstdint>
#include <iostream>
#include <map>
#include <mutex>
#include <random>
#include <stdexcept>
#include <thread>

/*
Oguz Ata Cal    - 6661014  
Karahan Saritas - 6661689
*/


// This are the item types
enum class Item { BOOK = 0, TOY = 1 };

/**
 * @brief Makes a random choice between Book and Toy.
 * A Book has 5x the chance to be sold compared to a toy.
 *
 * @return Item The item that should be sold.
 */
Item choose_what_to_sell() {
  static thread_local std::default_random_engine generator;
  std::discrete_distribution<int> distribution({5, 1});
  int choice = distribution(generator); // either 0 or 1
  return static_cast<Item>(choice);     // either Book or Toy
}

// please use these three mutexes to protect access to shared variables
std::mutex item_mutex;
std::mutex book_mutex;
std::mutex toy_mutex;

/**
 * @brief Sell a book
 */
void sell_book(Shop &shop) {
  if (shop.books_on_stock <= 0) {
    throw std::runtime_error("Books are out of stock!");
  }
  book_mutex.lock(); // lock the book mutex for both sold_books and books_on_stock
  ++shop.sold_books;
  --shop.books_on_stock;
  book_mutex.unlock();

  ++shop.sold_items; // item_mutex is already locked in sell_stuff function
}

/**
 * @brief Sell a toy
 */
void sell_toy(Shop &shop) {
  if (shop.toys_on_stock <= 0) {
    throw std::runtime_error("Toys are out of stock!");
  }
  toy_mutex.lock(); // lock the toy mutex for both sold_toys and toys_on_stock
  ++shop.sold_toys;
  --shop.toys_on_stock;
  toy_mutex.unlock();
  

  ++shop.sold_items; // item_mutex is already locked in sell_stuff function

}

/**
 * @brief A salesperson keeps selling items.
 * When the store runs out of items she gets frustrated and quits.
 */
void sell_stuff(Shop &shop) {
  while (true) {
    // randomly select what to sell
    try {
      const std::lock_guard<std::mutex> lock(item_mutex);
      if (choose_what_to_sell() == Item::TOY) {
        sell_toy(shop);
      } else {
        sell_book(shop);
      }
    } catch (std::runtime_error &exception) {
      std::cout << exception.what() << " Salesperson is frustrated and quits!"
                << std::endl;
      break;
    }
  }
}
