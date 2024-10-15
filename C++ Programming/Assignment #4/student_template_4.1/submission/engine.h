/*
Oguz Ata Cal    - 6661014  
Karahan Saritas - 6661689
*/

#pragma once

// own classes
#include "Amoeba.h"
#include "DeadCell.h"
#include "Bacterium.h"
#include "Virus.h"

// STL classes
#include <vector>

double random_value(double difficulty = 1.0) {
  return (double((std::rand() % 800) + 100) / 1000.0) * difficulty; // [0.1 .. 0.899] * difficulty
}

bool combat(Amoeba *player, Food *enemy) {
  std::cout << "\n+++++++++++++++++++++++++ new round ++++++++++++++++++++++++\n"
            << std::endl; 

  unsigned int turn = 0;
  
  bool player_won{false};

  if (player && enemy) {
    // print enemy and player information
    enemy->print_header();
    enemy->print();
    player->print_header();
    player->print();

    // fight until either player or enemy lost
    while (player->is_alive() && enemy->is_alive()) {
      turn++;
      
      auto player_attack = [&player](double &health, double defence) {
        health -= std::max(0.0, player->get_power() * random_value() - defence * random_value());
      };
      auto enemy_attack = [&enemy](double &health, double defence) {
        health -= std::max(0.0, enemy->get_power() * random_value() - defence * random_value());
      };
      // player attacks first
      enemy->attacked(player_attack); // lambda function

      // enemy attacks second
      player->attacked(enemy_attack); // Your Code here

      // print enemy and player information for this turn
      std::cout << "---------- turn: " << turn << " -----------" << std::endl;
      enemy->print_header();
      enemy->print();
      player->print_header();
      player->print();
    }

    player_won = (player->is_alive() && !enemy->is_alive());

    if (player_won)
    {
      player->eat(enemy->get_health_gain(), enemy->get_dna_gain());
    }
  }

  std::cout << "\n++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n"
            << std::endl; 

  return player_won;
}

void engine()
{
  std::cout << "Start game!" << std::endl;

  double difficulty{1.0};

  // create the player  
  Amoeba player; // default constructor

  // while player is still alive: fight!

  double DeadCell_default_health = 10.0;
  double DeadCell_default_power = 0.0;
  double DeadCell_default_defence = 0.0;

  double Bacterium_default_health = 200.0;
  double Bacterium_default_power = 10.0;
  double Bacterium_default_defence = 50.0;

  double Virus_default_health = 500.0;
  double Virus_default_power = 25.0;
  double Virus_default_defence = 25.0;
  while (player.is_alive()) {
    std::cout << "Difficulty: " << difficulty << std::endl;

    // define a vector with distinct enemies (DeadCell, Bacterium, ...) which are the competitors and available to fight in this round
    std::vector<Food *> all_enemies;
  
    all_enemies.push_back(new DeadCell(random_value(difficulty) * DeadCell_default_health, random_value(difficulty) * DeadCell_default_power, random_value(difficulty) * DeadCell_default_defence));
    all_enemies.push_back(new Bacterium(random_value(difficulty )* Bacterium_default_health, random_value(difficulty) * Bacterium_default_power, random_value(difficulty) * Bacterium_default_defence));
    all_enemies.push_back(new Virus(random_value(difficulty) * Virus_default_health, random_value(difficulty) * Virus_default_power, random_value(difficulty) * Virus_default_defence));

    if(all_enemies.empty()) return;

    player.print_header();
    player.print();
    std::cout << "\n";

    // deep copy (one to three) random competitor the player can fight in this round. Duplicates are allowed.
    std::vector<Food *> enemy;
    // take 3 candidate enemies (duplicates are allowed)
    enemy.push_back(all_enemies[std::rand() % int(all_enemies.size())]->clone());
    enemy.push_back(all_enemies[std::rand() % int(all_enemies.size())]->clone());
    enemy.push_back(all_enemies[std::rand() % int(all_enemies.size())]->clone());

    // print number of enemies selected for this round
    std::cout << std::setw(10) << "Select" << " : ";
    enemy[0]->print_header(); // only print header of first entry to reduce output in InfoMark
    for (size_t i = 0; i < enemy.size(); i++) {
      std::cout << std::setw(10) << i << " : ";
      enemy[i]->print();
    }
    std::cout << "Choose your enemy (0-" << enemy.size() - 1 << "): ";

    unsigned int selection;
    std::cin >> selection;

    selection = selection % enemy.size();

    combat(&player, enemy[selection]);

    difficulty += 1.0;
    
    // Your Code here
    // Hint: don't forget to free dynamically allocated objects!

    for (size_t i = 0; i < all_enemies.size(); i++) delete all_enemies[i];  
    for (size_t i = 0; i < enemy.size(); i++) delete enemy[i];       
    
    all_enemies.clear();   // clear the vector
    enemy.clear();         // clear the vector  
  }
	std::cout << "Player reached difficulty level (round): " << difficulty << std::endl;
}
