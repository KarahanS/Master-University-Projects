#pragma once

#include "shapes.h"

#include <memory>

/// shared abstract operation base class containing two nested shapes
class Operation : public Shape {
protected:
    Operation(const Shape& shape_a, const Shape& shape_b);
public:
    Shape sub_shape_a, sub_shape_b;

private:
    /// by default, operations between two shapes have a result that is contained within their combined bounds
    AABB getBounds_impl() const override;
};


class And : public Operation {
public:
    And(const Shape& shape_a, const Shape& shape_b);
    Shape clone_impl() const override;
    bool isInside_impl(const Point3D& p) const override;
};
class Or: public Operation {
public:
    Or(const Shape& shape_a, const Shape& shape_b);
    Shape clone_impl() const override;
    bool isInside_impl(const Point3D& p) const override;
};

class Not: public Operation {
public:
    Not(const Shape& shape_a);
    Shape clone_impl() const override;
    bool isInside_impl(const Point3D& p) const override;
};

class Xor: public Operation {
public:
    Xor(const Shape& shape_a, const Shape& shape_b);
    Shape clone_impl() const override;
    bool isInside_impl(const Point3D& p) const override;
};