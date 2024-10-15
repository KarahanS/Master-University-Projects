#include "transformations.h"
#include <cmath>
#include <vector>
#include <iostream>

#define PI 3.14159265

// helper function for 2D rotations
std::pair<float, float> rotate2D(float x, float y, float angle)
{
    const float sin = std::sin(angle);
    const float cos = std::cos(angle);

    return {cos*x-sin*y, sin*x+cos*y};
}

// initializer lists
Transformation::Transformation(const Shape& shape): sub_shape{shape.clone()} {}
Scaled::Scaled(const Shape& shape, const Point3D& factor): Transformation{shape}, factor{factor} {}
Translated::Translated(const Shape& shape, const Point3D& offset): Transformation{shape}, offset{offset} {}
Rotated::Rotated(const Shape& shape, const Axis& axis, float angle): Transformation{shape}, axis{axis}, angle{angle} {}

Shape Scaled::clone_impl() const { return {std::make_shared<Scaled>(sub_shape, factor)};}
AABB Scaled::getBounds_impl() const {
    // just scale the bounds of the sub_shape
    AABB bounds = sub_shape.getBounds();
    Point3D max_ = bounds.max * factor;
    Point3D min_ = bounds.min * factor;

    return AABB{min_, max_};
}

bool Scaled::isInside_impl(const Point3D& p) const {
    return sub_shape.isInside(p / factor);
}


Shape Translated::clone_impl() const { return {std::make_shared<Translated>(sub_shape, offset)}; }   
AABB Translated::getBounds_impl() const {
    // just translate the bounds of the sub_shape
    AABB bounds = sub_shape.getBounds();
    Point3D max_ = bounds.max + offset;
    Point3D min_ = bounds.min + offset;

    return AABB{min_, max_};
}

bool Translated::isInside_impl(const Point3D& p) const {
    return sub_shape.isInside(p - offset);
}

Shape Rotated::clone_impl() const {
    return {std::make_shared<Rotated>(sub_shape, axis, angle)};
}

Point3D Rotated::rotate(Point3D p) const {
    if (axis == Axis::X) std::tie(p.y, p.z) = rotate2D(p.y, p.z, angle);
    else if (axis == Axis::Y) std::tie(p.x, p.z) = rotate2D(p.x, p.z, angle);
    else std::tie(p.x, p.y) = rotate2D(p.x, p.y, angle);
    return p;
}
AABB Rotated::getBounds_impl() const {
    AABB bounds = sub_shape.getBounds();
    Point3D min = bounds.min, max = bounds.max;

    // lambda func to rotate a point
    auto rotated = rotate(min);
    float x_min = rotated.x, x_max = rotated.x;
    float y_min = rotated.y, y_max = rotated.y;
    float z_min = rotated.z, z_max = rotated.z;

    for (int i=0; i<2; i++) {  // iterate all 8 corners
        for (int j=0; j<2; j++) {
            for(int z=0; z<2; z++) {
                Point3D corner = {i ? max.x : min.x, j ? max.y : min.y, z ? max.z : min.z};
                Point3D rotated = rotate(corner);
                x_min = std::min(x_min, rotated.x);
                y_min = std::min(y_min, rotated.y);
                z_min = std::min(z_min, rotated.z);

                x_max = std::max(x_max, rotated.x);
                y_max = std::max(y_max, rotated.y);
                z_max = std::max(z_max, rotated.z);
            }
        }
    }
    Point3D min_(x_min, y_min, z_min);
    Point3D max_(x_max, y_max, z_max);

    return AABB{min_, max_};
}

bool Rotated::isInside_impl(const Point3D& p) const {
    Point3D rotated = rotate(p);
    return sub_shape.isInside(rotated);
}