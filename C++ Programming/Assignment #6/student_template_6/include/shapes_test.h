#pragma once

#include "shapes.h"
#include "transformations.h"
#include "operations.h"

#if __cpp_lib_concepts >= 202002L

#include <concepts>

// concepts for checking the progress of your implementation

template <typename TShape>
concept ShapeImplemented = requires (TShape s, const Point3D p)
{
    {s.clone()}     -> std::same_as<Shape>;
    {s.isInside(p)} -> std::same_as<bool>;
    {s.getBounds()} -> std::same_as<AABB>;
};

template <typename TShape>
concept OperationsImplemented = requires (TShape s1, TShape s2)
{
    {s1&s2} -> std::same_as<Shape>;
    {s1|s2} -> std::same_as<Shape>;
    {s1^s2} -> std::same_as<Shape>;
    {!s1}   -> std::same_as<Shape>;
    {s1+s2} -> std::same_as<Shape>;
    {s1-s2} -> std::same_as<Shape>;
};

template <typename TShape>
concept TransformationsImplemented = requires (TShape s, Point3D p, Axis axis, float angle) {
    {s.translated(p)}        -> std::same_as<Shape>;
    {s.scaled(p)}            -> std::same_as<Shape>;
    {s.rotated(axis, angle)} -> std::same_as<Shape>;
};

template <typename TShape>
concept ShapeFullyImplemented = ShapeImplemented<TShape> && OperationsImplemented<TShape> && TransformationsImplemented<TShape>;


// these tests will check at compile time if the necessary classes, functions, and operations are implemented
// they only check for the type signatures, they do _not_ check if they are implemented correctly
inline void test_shape_implementation() {
#ifdef TEST_TASK_1
    ShapeImplemented auto cube = Cube{};
    ShapeImplemented auto sphere = Sphere{};
    ShapeImplemented auto cylinder = Cylinder{};
    ShapeImplemented auto octahedron = Octahedron{};
#endif

#ifdef TEST_TASK_2
    OperationsImplemented auto operations_test = Cube{};
#endif

#ifdef TEST_TASK_3
    TransformationsImplemented auto transformations_test = Cube{};
#endif

#ifdef TEST_TASK_4
    // there is nothing here - everything is already declared - you only need to fill in the functions
    // use the tests on Infomark or check the output on the console
#endif
}

#else
#warning Concepts Library requires installation of libstdc++-10 or libc++-13 - concept tests are disabled
#endif
