#include "exercise_82.h"
#include "point.h"
#include <algorithm>
#include <vector>

std::vector<Point> sort_x(const std::vector<Point> &points) {
  std::vector<Point> sorted_points = points;
  std::sort(sorted_points.begin(), sorted_points.end(), [](const Point &a, const Point &b) {return a.x < b.x;});
  return sorted_points;
}

std::vector<Point> sort_y(const std::vector<Point> &points) {
  std::vector<Point> sorted_points = points;
  std::sort(sorted_points.begin(), sorted_points.end(), [](const Point &a, const Point &b) {return a.y < b.y;});
  return sorted_points;
}

Point median(const std::vector<Point> &points) {
  std::vector<Point> points_x = sort_x(points);
  float x_median = points_x.size() % 2 ? points_x[points_x.size() / 2].x : (points_x[points_x.size() / 2].x + points_x[points_x.size() / 2 - 1].x) / 2;
  std::vector<Point> points_y = sort_y(points);
  float y_median = points.size() % 2 ? points_y[points_y.size() / 2].y : (points_y[points_y.size() / 2].y + points_y[points_y.size() / 2 - 1].y) / 2;
  return Point(x_median, y_median);

}