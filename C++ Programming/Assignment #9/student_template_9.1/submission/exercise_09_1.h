#pragma once

/**
 * @brief Representation of a simple shop that sells books and toys.
 * We use this to keep track of what has been sold and what is still on stock
 */
struct Shop {
  // This are the items that the store has on stock
  int books_on_stock;
  int toys_on_stock;

  // This is how many items were sold already
  int sold_items;
  int sold_books;
  int sold_toys;
};

/**
 * @brief This is what a salesperson does. She sells stuff from a shop.
 * 
 * @param shop The shop representation that is used to keep track of the items.
 */
void sell_stuff(Shop &shop);

/**
 * @brief Sell one book from a shop
 * @throws std::runtime_error If no book is left to sell in the shop a runtime
 * exception is thrown.
 * 
 * @param shop The shop representation that is updated when the book is sold.
 */
void sell_book(Shop &shop);

/**
 * @brief Sell one toy from a shop
 * @throws std::runtime_error If no toy is left to sell in the shop a runtime
 * exception is thrown.
 * 
 * @param shop The shop representation that is updated when the toy is sold.
 */
void sell_toy(Shop &shop);
