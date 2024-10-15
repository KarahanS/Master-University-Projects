#include "Food.h"

//TODO: task b)
Food::Food(double health_, double power_, double defence_, double health_gain_, double dna_gain_) {
  health = health_;
  power = power_;
  defence = defence_;
  health_gain = health_gain_;
  dna_gain = dna_gain_;
}

Food::~Food() {}

void Food::attacked(std::function<void(double &, double)> attack_function) {
  if (attack_function) {
    attack_function(health, defence); // passes the (protected) variables health and defence to the function object
  }
}

//TODO: task a)
bool Food::is_alive() {
  return health > 0;
}

void Food::print_header() {
  std::cout << std::setw(10) << "name" << " | ";
  std::cout << std::setw(10) << "health" << " | ";
  std::cout << std::setw(10) << "power" << " | ";
  std::cout << std::setw(10) << "defence" << " | ";
  std::cout << std::setw(10) << "health_g" << " | ";
  std::cout << std::setw(10) << "dna_gain" << std::endl;
}

void Food::print() {
  std::cout << std::setw(10) << name << " | ";
  std::cout << std::setw(10) << health << " | ";
  std::cout << std::setw(10) << power << " | ";
  std::cout << std::setw(10) << defence << " | ";
  std::cout << std::setw(10) << health_gain << " | ";
  std::cout << std::setw(10) << dna_gain << std::endl;
}

double Food::get_power() { return power; }

double Food::get_dna_gain() { return dna_gain; }

double Food::get_health_gain() { return health_gain; }
