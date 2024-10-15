#pragma once

#include "shapes.h"

#include <memory>

/// shared abstract transformation base class containing a nested shape to be transformed
class Transformation : public Shape {
protected:
    Transformation(const Shape& shape);
public:
    Shape sub_shape;
};

// of course, one could implement all these transformations jointly as a single transformation matrix.
// for simplicity, we don't do that here.

class Scaled: public Transformation {
public:
    Scaled(const Shape& shape, const Point3D& factor);
    Shape clone_impl() const override;
    AABB getBounds_impl() const override;
    bool isInside_impl(const Point3D& p) const override;

private:
    Point3D factor;
};

class Translated: public Transformation {
public:
    Translated(const Shape& shape, const Point3D& offset);
    Shape clone_impl() const override;
    AABB getBounds_impl() const override;
    bool isInside_impl(const Point3D& p) const override;

private:
    Point3D offset;
};

class Rotated: public Transformation {
public:
    Rotated(const Shape& shape, const Axis& axis, float angle);
    Shape clone_impl() const override;
    AABB getBounds_impl() const override;
    bool isInside_impl(const Point3D& p) const override;

private:
    Axis axis;
    float angle;

    Point3D rotate(Point3D p) const;
};
