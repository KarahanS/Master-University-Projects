#pragma once

#include <cmath> // std::sqrt
#include <algorithm> // std::min, std::max
#include <ostream> // std::ostream

enum class Axis {
    X,
    Y,
    Z
};

struct Point3D {
    float x, y, z;

    /// initialize each member with a different value
    Point3D(float x, float y, float z) : x{x}, y{y}, z{z} {}
    /// initialize all members with the same value
    Point3D(float xyz=0.0f) : Point3D{xyz, xyz, xyz} {}

    bool operator==(const Point3D& other) const = default;

    void operator+=(const Point3D& other) {
        x += other.x;
        y += other.y;
        z += other.z;
    }
    Point3D operator+(const Point3D& other) const {
        Point3D copy{*this};
        copy += other;
        return copy;
    }

    void operator-=(const Point3D& other) {
        x -= other.x;
        y -= other.y;
        z -= other.z;
    }
    Point3D operator-(const Point3D& other) const {
        Point3D copy{*this};
        copy -= other;
        return copy;
    }

    Point3D operator-() const {
        Point3D copy{*this};
        copy.x = -copy.x;
        copy.y = -copy.y;
        copy.z = -copy.z;
        return copy;
    }

    /// compute the component-wise multiplication
    void operator*=(const Point3D& other) {
        x *= other.x;
        y *= other.y;
        z *= other.z;
    }
    /// compute the component-wise multiplication
    Point3D operator*(const Point3D& other) const {
        Point3D copy{*this};
        copy *= other;
        return copy;
    }

    /// compute the component-wise division
    void operator/=(const Point3D& other) {
        x /= other.x;
        y /= other.y;
        z /= other.z;
    }
    /// compute the component-wise division
    Point3D operator/(const Point3D& other) const {
        Point3D copy{*this};
        copy /= other;
        return copy;
    }

    /// compute the L2 norm (the distance to [0.0, 0.0, 0.0])
    float norm() const {
        return std::sqrt(x*x + y*y + z*z);
    }
};

// interoperability with floats

/// addition onto scalar
inline Point3D operator+(float a, const Point3D& b)
{
    return b + a;
}
/// subtraction from scalar
inline Point3D operator-(float a, const Point3D& b)
{
    return Point3D{a} - b;
}
/// multiplication onto scalar
inline Point3D operator*(float a, const Point3D& b)
{
    return b * a;
}
/// fraction of scalar
inline Point3D operator/(float a, const Point3D& b)
{
    return Point3D{a} / b;
}

/// compute the component-wise minimum
inline Point3D min(const Point3D& a, const Point3D& b) {
    return {std::min(a.x, b.x), std::min(a.y, b.y), std::min(a.z, b.z)};
}

/// compute the component-wise maximum
inline Point3D max(const Point3D& a, const Point3D& b) {
    return {std::max(a.x, b.x), std::max(a.y, b.y), std::max(a.z, b.z)};
}

/// compute the dot product of a and b
inline float dot(const Point3D& a, const Point3D& b) {
    return a.x*b.x + a.y*b.y + a.z*b.z;
}

/// compute the distance between a and b
inline float distance(const Point3D& a, const Point3D& b) {
    return (a-b).norm();
}

/// check if all components are less than the other point's components
inline bool operator<(const Point3D& a, const Point3D& b) {
    return a.x < b.x && a.y < b.y && a.z < b.z;
}

/// check if all components are greater than the other point's components
inline bool operator>(const Point3D& a, const Point3D& b) {
    return a.x > b.x && a.y > b.y && a.z > b.z;
}

/// check if all components are less than the other point's components or equal
inline bool operator<=(const Point3D& a, const Point3D& b) {
    return a.x <= b.x && a.y <= b.y && a.z <= b.z;
}

/// check if all components are greater than the other point's components or equal
inline bool operator>=(const Point3D& a, const Point3D& b) {
    return a.x >= b.x && a.y >= b.y && a.z >= b.z;
}

/// "to string"
inline std::ostream& operator<<(std::ostream& os, const Point3D& p) {
    os << '[' << p.x << ", " << p.y << ", " << p.z << ']';
    return os;
}
