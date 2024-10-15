#pragma once

#include "Food.h"
class Bacterium : public Food {

public:
  Bacterium(double health_, double power_, double defence_);
  ~Bacterium();

  virtual Food *clone() const override;
};
