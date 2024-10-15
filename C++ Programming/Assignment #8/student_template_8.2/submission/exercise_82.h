#pragma once
#include <vector>
#include "point.h"


/**
 * @brief Sorts the given list of points by the x-coordinate and returns the sorted copy.
 * Use std::sort() to implement this!

 * @param points The input list of points.
 * @return std::vector<Point> Sorted copy of the points.
 */
std::vector<Point> sort_x(const std::vector<Point> &points);

/**
 * @brief Sorts the given list of points by the y-coordinate and returns the sorted copy.
 * Use std::sort() to implement this!
 * 
 * @param points The input list of points.
 * @return std::vector<Point> Sorted copy of the points.
 */
std::vector<Point> sort_y(const std::vector<Point> &points);


/**
 * @brief Computes the median in x- and y-direction and returns both in one Point
 * where the x-variable is the x-median and the y-variable is the y-median.
 * 
 * @param points The list of points from which the medians are computed.
 * @return Point The medians in x and y direction.
 */
Point median(const std::vector<Point> &points);