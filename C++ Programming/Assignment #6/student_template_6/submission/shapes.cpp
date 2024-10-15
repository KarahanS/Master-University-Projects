#include "shapes.h"

#include "transformations.h"
#include "operations.h"

#include <stdexcept>

Shape::Shape(std::shared_ptr<Shape>&& shape) noexcept
    : instance{std::move(shape)} // take ownership
{

}

Shape Shape::operator&(const Shape& other) const {
    // return And of this and other instances
    // And is a Shape, so we can return it directly
    return {std::make_shared<And>(*this, other)};
}

Shape Shape::operator|(const Shape& other) const { return {std::make_shared<Or>(*this, other)}; }
Shape Shape::operator^(const Shape& other) const { return {std::make_shared<Xor>(*this, other)};}
Shape Shape::operator!() const { return {std::make_shared<Not>(*this)}; }
Shape Shape::operator+(const Shape& other) const { return operator|(other); }
Shape Shape::operator-(const Shape& other) const { return operator&(!other);}

Shape Shape::clone() const
{
    // if this shape just contains a pointer, we can return a simple copy
    if (instance)
        return *this;

    // otherwise, call the derived clone implementation
    return clone_impl();
}

AABB Shape::getBounds() const {
    // call the nested shape (if any)
    if (instance)
        return instance->getBounds();

    // otherwise, call the derived getBounds implementation
    return getBounds_impl();
}

bool Shape::isInside(const Point3D& p) const
{
    // call the nested shape (if any)
    if (instance)
        return instance->isInside(p);

    // otherwise, call the derived isInside implementation
    return isInside_impl(p);
}

Shape Shape::clone_impl() const
{
    // no default implementation available (but cannot set = 0, since we want to have instances of Shape)
    // if you get this error, you forgot to implement the override
    throw std::logic_error("clone called on an abstract shape");
}

AABB Shape::getBounds_impl() const
{
    // fallback default implementation
    return AABB{-1.0f, 1.0f};
}

bool Shape::isInside_impl(const Point3D&) const
{
    // no fallback implementation (but cannot set = 0, since we want to have instances of Shape)
    // if you get this error, you forgot to implement the override
    throw std::logic_error("isInside called on an abstract shape");
}

Shape Empty::clone_impl() const
{
    return {std::make_shared<Empty>()};
}

bool Empty::isInside_impl(const Point3D&) const {
    return false;
}

AABB Empty::getBounds_impl() const
{
    return AABB{};
}

Shape Cube::clone_impl() const
{
    return {std::make_shared<Cube>()};
}

bool Cube::isInside_impl(const Point3D& p) const {
    return getBounds().contains(p);
}

Shape Sphere::clone_impl() const  // this will be called from the clone function
{
    return {std::make_shared<Sphere>()}; // this will create a new instance of Sphere and return the shared pointer
}

bool Sphere::isInside_impl(const Point3D& p) const {
    return p.norm() <= 1.0f;
}


Shape Cylinder::clone_impl() const {
    return {std::make_shared<Cylinder>()};
}

bool Cylinder::isInside_impl(const Point3D& p) const {
    return p.x*p.x + p.y*p.y <= 1.0f && p.z <= 1.0f && p.z >= -1.0f;
}

Shape Octahedron::clone_impl() const {
    return {std::make_shared<Octahedron>()};
}

bool Octahedron::isInside_impl(const Point3D& p) const {
    return std::abs(p.x) + std::abs(p.y) + std::abs(p.z) <= 1.0f;
}

Shape Shape::scaled(const Point3D& factor) const {
    return {std::make_shared<Scaled>(*this, factor)};
}
Shape Shape::translated(const Point3D& offset) const {
    return {std::make_shared<Translated>(*this, offset)};
}
Shape Shape::rotated(const Axis& axis, float angle) const {
    return {std::make_shared<Rotated>(*this, axis, angle)};
}