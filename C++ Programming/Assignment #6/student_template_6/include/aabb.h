#pragma once

#include "point3d.h"

#include <limits>

/// axis aligned bounding box
struct AABB {
    /// component-wise minimum of the positions contained in the bounding box
    Point3D min {std::numeric_limits<float>::infinity()};
    /// component-wise maximum of the positions contained in the bounding box
    Point3D max {-std::numeric_limits<float>::infinity()};

    /// extend the bounding box to contain the given point
    void extend(const Point3D& p) {
        min = ::min(min, p);
        max = ::max(max, p);
    }

    /// returns the size of the bounding box
    Point3D extents() const {
        return max-min;
    }

    /// checks whether the point is contained in the bounding box
    bool contains(const Point3D& p) const {
        return p >= min && p <= max;
    }

    /// checks whether the given bounding box is contained in this bounding box
    bool contains(const AABB& other) const {
        return other.min >= min && other.max <= max;
    }

    /// extend the bounding box to also contain the other bounding box
    AABB operator+(const AABB& other) {
        return {::min(min, other.min), ::max(max, other.max)};
    }

    bool operator==(const AABB& other) const = default;
};

/// "to string"
inline std::ostream& operator<<(std::ostream& os, const AABB& aabb) {
    os << "AABB[" << aabb.min << ", " << aabb.max << ']';
    return os;
}
