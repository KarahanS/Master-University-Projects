#pragma once

#include "Food.h"
class Virus : public Food {

public:
  Virus(double health_, double power_, double defence_);
  ~Virus();

  virtual Food *clone() const override;
};
