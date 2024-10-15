#include "operations.h"

Operation::Operation(const Shape& shape_a, const Shape& shape_b)
    : sub_shape_a{shape_a.clone()}, sub_shape_b{shape_b.clone()}
{

}

AABB Operation::getBounds_impl() const
{
    return sub_shape_a.getBounds()+sub_shape_b.getBounds();
}

And::And(const Shape& shape_a, const Shape& shape_b): Operation{shape_a, shape_b} {}

Shape And::clone_impl() const
{
    // this time we have shape_a and shape_b, so we need to clone "this" instance - it is not generic like shapes
    return {std::make_shared<And>(sub_shape_a, sub_shape_b)}; 
}

bool And::isInside_impl(const Point3D& p) const
{
    return sub_shape_a.isInside(p) && sub_shape_b.isInside(p);
}

Or::Or(const Shape& shape_a, const Shape& shape_b): Operation{shape_a, shape_b} {}

Shape Or::clone_impl() const {
    return {std::make_shared<Or>(sub_shape_a, sub_shape_b)};
}

bool Or::isInside_impl(const Point3D& p) const {
    return sub_shape_a.isInside(p) || sub_shape_b.isInside(p);
}

Not::Not(const Shape& shape_a): Operation{shape_a, shape_a} {}

Shape Not::clone_impl() const {
    return {std::make_shared<Not>(sub_shape_a)};
}

bool Not::isInside_impl(const Point3D& p) const {
    return !sub_shape_a.isInside(p);
}

Xor::Xor(const Shape& shape_a, const Shape& shape_b): Operation{shape_a, shape_b} {}


Shape Xor::clone_impl() const {
    return {std::make_shared<Xor>(sub_shape_a, sub_shape_b)};
}

bool Xor::isInside_impl(const Point3D& p) const {
    return sub_shape_a.isInside(p) != sub_shape_b.isInside(p);
}