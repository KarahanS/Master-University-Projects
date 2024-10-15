#pragma once

#include <functional>
#include <iomanip>
#include <iostream>
#include <string>

class Food {
public:
  // constructor
  Food(double health_, double power_, double defence_, double health_gain_, double dna_gain_);

  // destructor
  virtual ~Food();

  virtual void attacked(std::function<void(double &, double)> attack_function);

  virtual Food *clone() const  = 0; // used to make polymorphic copies

  bool is_alive(); // alive if health > 0.0

  // print variable names and values
  virtual void print_header();
  virtual void print();

  // getter for class variables
  double get_power();
  double get_dna_gain();
  double get_health_gain();

protected:
  double health{};
  double power{};
  double defence{};
  double health_gain{};
  double dna_gain{};

  std::string name;
};
