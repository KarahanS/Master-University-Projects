#pragma once

#include "point3d.h"
#include "aabb.h"

#include <memory>

/// abstract shape base class
class Shape {
public:
  /// create a wrapper shape to encapsulate arbitrary derived shapes
  Shape(std::shared_ptr<Shape>&& shape) noexcept;
  // Make the destructor virtual to ensure that deleting shape deletes the derived type.
  virtual ~Shape() noexcept = default;

  /// create a clone of this Shape (wrapped in a base-type Shape)
  Shape clone() const;

  /// return the bounding box of the shape (by default [-1, -1, -1] to [1, 1, 1] for geometric shapes)
  AABB getBounds() const;
  /// check if the point is part of the shape's volume
  bool isInside(const Point3D& p) const;

  Shape operator&(const Shape& other) const;
  Shape operator|(const Shape& other) const;
  Shape operator^(const Shape& other) const;
  Shape operator!() const;
  Shape operator+(const Shape& other) const;
  Shape operator-(const Shape& other) const;

  Shape scaled(const Point3D& factor) const;
  Shape translated(const Point3D& offset) const;
  Shape rotated(const Axis& axis, float angle) const;

protected:
  // Protect default constructor to disallow instantiating abstract shapes.
  // If there was a pure virtual function, the compiler would prevent this already.
  // However, we would like to be able to always return a "Shape" rather than a pointer, so we need to allow instances of Shape
  Shape() = default;
  // these all should only be used on wrapped shapes
  Shape(const Shape&) = default;
  Shape& operator=(const Shape&) = default;
  Shape(Shape&&) noexcept = default;
  Shape& operator=(Shape&&) noexcept = default;

private:
  /// Pointer to derived shape instance when wrapped as a basic Shape instance, otherwise nullptr
  /// This instance is shared when the object is cloned, so better make it const. (Otherwise, modifying any copy would modify all copies.)
  std::shared_ptr<const Shape> instance {nullptr};

  // Since we want to create instances of Shape, we cannot make these pure virtual even if there is no sensible default.
  // The functions clone_impl and isInside_impl simply throw an exception when called.
  // Even though declared private, derived classes can still override these!

  /// return derived shape instance wrapped in a basic Shape instance or simply a copy of an existing basic Shape wrapper instance.
  virtual Shape clone_impl() const;
  /// get the bounding box of this shape (defaults to AABB{-1.0f, 1.0f}, override where necessary)
  virtual AABB getBounds_impl() const;
  /// check whether the given point is contained in this shape (override this in your shape implementation)
  virtual bool isInside_impl(const Point3D&) const;
};

class Empty final: public Shape {
private:
    Shape clone_impl() const override;
    bool isInside_impl(const Point3D&) const override;
    AABB getBounds_impl() const override;
};

class Cube final: public Shape {
private:
    Shape clone_impl() const override;
    bool isInside_impl(const Point3D& p) const override;
};

class Sphere final: public Shape {
private:
    Shape clone_impl() const override;
    bool isInside_impl(const Point3D& p) const override;
};

class Cylinder final: public Shape {
private:
    Shape clone_impl() const override;
    bool isInside_impl(const Point3D& p) const override;
};

class Octahedron final: public Shape {
private:
    Shape clone_impl() const override;
    bool isInside_impl(const Point3D& p) const override;
};