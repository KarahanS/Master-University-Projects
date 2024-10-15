#include "submission/exercise_09_1.h"
#include <cstdlib>
#include <iostream>
#include <thread>

int main() {
  Shop shop;
  shop.books_on_stock = 1000;
  shop.toys_on_stock = 200;
  shop.sold_items = 0;
  shop.sold_books = 0;
  shop.sold_toys = 0;

  // Lets keep track of how many items the store had in the beginning
  int items_init = shop.books_on_stock + shop.toys_on_stock;
  int books_init = shop.books_on_stock;

  int toys_init = shop.toys_on_stock;
  // let some salespersons sell the items in parallel
  auto salesperson1 = std::thread(sell_stuff, std::ref(shop));
  auto salesperson2 = std::thread(sell_stuff, std::ref(shop));
  auto salesperson3 = std::thread(sell_stuff, std::ref(shop));

  // wait until all threads finished
  salesperson1.join();
  salesperson2.join();
  salesperson3.join();

  // lets see what was sold and how many items are left in stock
  std::cout << "Items: Start stock: " << items_init
            << "\tSold: " << shop.sold_items << "\n";
  std::cout << "Books: Start stock: " << books_init
            << "\tSold: " << shop.sold_books
            << "\tLeft on stock: " << shop.books_on_stock << "\n";
  std::cout << "Toys: Start stock: " << toys_init
            << "\tSold: " << shop.sold_toys
            << "\tLeft on stock: " << shop.toys_on_stock << "\n";
  return EXIT_SUCCESS;
}
