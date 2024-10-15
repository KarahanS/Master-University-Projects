#pragma once
#include <iostream>

/**
 * @brief Simple representation of a 2d-point
 * 
 */
struct Point {
    float x, y;

    Point(float x_, float y_) : x(x_), y(y_) {}

};

inline std::ostream &operator<<(std::ostream &os, Point const &m) {
    return os << "(" << m.x << ", " << m.y << ")";
}