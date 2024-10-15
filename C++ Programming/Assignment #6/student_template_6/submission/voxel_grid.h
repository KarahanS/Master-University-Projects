#pragma once

#include "point3d.h"
#include "aabb.h"
#include "shapes.h"

#include <cstdint>
#include <memory>
#include <ostream>
#include <vector>

struct VoxelSlice {
    std::vector<std::vector<bool>> data;

    VoxelSlice(uint32_t res_x, uint32_t res_y)
        : data(res_y, std::vector<bool>(res_x)) {}
};

class VoxelGrid final : public Shape {
public:
    /// determines how many voxels to allocate per unit of space
    const static uint32_t level_of_detail = 8;

private:
    /// space represented by the voxel grid
    AABB bounds {0.0f, 0.0f};

    /// resolution of the voxel grid in x, y, z
    uint32_t res_x {0}, res_y {0}, res_z {0};
    /// flat voxel storage
    std::vector<bool> voxels;

public:
    /// implicit conversion from shape to voxel grid - this declaration disables the default constructor
    VoxelGrid(const Shape& shape);

    /// returns the actual resolution of the voxel grid
    std::tuple<uint32_t, uint32_t, uint32_t> getResolution() const;

    /// returns the yz/zx/xy plane at the specified x/y/z index
    VoxelSlice extractSlice(Axis axis, uint32_t slice) const;

private:
    Shape clone_impl() const override;
    AABB getBounds_impl() const override;
    bool isInside_impl(const Point3D &p) const override;

    bool isSet(uint32_t x, uint32_t y, uint32_t z) const;
    Point3D voxelCenter(uint32_t x, uint32_t y, uint32_t z) const;
};


/// append a string representation of the voxel slice to the stream (e.g. VoxelSlice vs; std::cout << vs;)
std::ostream& operator<<(std::ostream& ostream, const VoxelSlice& slice);

/// append a string representation of the voxel grid to the stream (e.g. VoxelGrid vg; std::cout << vg;)
std::ostream& operator<<(std::ostream& ostream, const VoxelGrid& vg);
