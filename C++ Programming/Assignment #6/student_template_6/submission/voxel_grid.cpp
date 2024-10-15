#include "voxel_grid.h"

#include "transformations.h"
#include "operations.h"

#include <cassert>
#include <iostream>
#include <sstream>


std::tuple<uint32_t, uint32_t, uint32_t> VoxelGrid::getResolution() const
{
    return {res_x, res_y, res_z};
}


Shape VoxelGrid::clone_impl() const
{
    return {std::make_shared<VoxelGrid>(*this)};
}

AABB VoxelGrid::getBounds_impl() const
{
    return bounds;
}


Point3D VoxelGrid::voxelCenter(uint32_t x, uint32_t y, uint32_t z) const {
    float stepsize_per_subdivision_x = (bounds.max.x - bounds.min.x) / res_x;
    float stepsize_per_subdivision_y = (bounds.max.y - bounds.min.y) / res_y;
    float stepsize_per_subdivision_z = (bounds.max.z - bounds.min.z) / res_z;

    return Point3D{bounds.min.x + (x + 0.5f) * stepsize_per_subdivision_x,
                   bounds.min.y + (y + 0.5f) * stepsize_per_subdivision_y,
                   bounds.min.z + (z + 0.5f) * stepsize_per_subdivision_z};
}

VoxelGrid::VoxelGrid(const Shape& shape) : bounds{shape.getBounds()} {
    float float_res_x = float(level_of_detail) * (bounds.max.x - bounds.min.x); 
    float float_res_y = float(level_of_detail) * (bounds.max.y - bounds.min.y);
    float float_res_z = float(level_of_detail) * (bounds.max.z - bounds.min.z);

    // Make sure not to cast ±∞ to int, since that is undefined behavior.
    res_x = std::isfinite(float_res_x) ? uint32_t(float_res_x) : 1u;
    res_y = std::isfinite(float_res_y) ? uint32_t(float_res_y) : 1u;
    res_z = std::isfinite(float_res_z) ? uint32_t(float_res_z) : 1u;

    res_x = std::max(res_x, 1u);  // at least 1
    res_y = std::max(res_y, 1u);
    res_z = std::max(res_z, 1u);

    voxels.resize(res_x * res_y * res_z, false);
    for (uint32_t x = 0; x < res_x; x++) {
        for (uint32_t y = 0; y < res_y; y++) {
            for (uint32_t z = 0; z < res_z; z++) {
                uint32_t idx = x + res_x * y + res_x * res_y * z;
                Point3D center = voxelCenter(x, y, z);
                if (shape.isInside(center)) {
                    voxels[idx] = true;
                }
            }
        }
    }    
}

bool VoxelGrid::isSet(uint32_t x, uint32_t y, uint32_t z) const {
    assert(x < res_x);
    assert(y < res_y);
    assert(z < res_z);

    return voxels[x + res_x * y + res_x * res_y * z];
}

VoxelSlice VoxelGrid::extractSlice(Axis axis, uint32_t slice) const {
    uint32_t res_i, res_j;
    if (axis == Axis::X)  { res_i = res_y; res_j = res_z; }
    else if (axis == Axis::Y) {  res_i = res_x; res_j = res_z; }
    else {  res_i = res_x; res_j = res_y; }

    VoxelSlice voxelSlice(res_i, res_j);
    voxelSlice.data.resize(res_j, std::vector<bool>(res_i));

    for (uint32_t i = 0; i < res_i; ++i) {
        for (uint32_t j = 0; j < res_j; ++j) {
            if (axis == Axis::X) voxelSlice.data[j][i] = isSet(slice, i, j);
            else if (axis == Axis::Y) voxelSlice.data[j][i] = isSet(i, slice, j);
            else voxelSlice.data[j][i] = isSet(i, j, slice);
        }
    }

    return voxelSlice;
}

std::ostream& operator<<(std::ostream& ostream, const VoxelSlice& slice) {
    for (const std::vector<bool>& row: slice.data) {
        for (bool b: row) ostream << (b ? 'X' : '.') << " ";
        ostream << '\n';
    }
    return ostream;
}

std::ostream& operator<<(std::ostream& ostream, const VoxelGrid& vg)  {
    uint32_t res_z;
    std::tie(std::ignore, std::ignore, res_z) = vg.getResolution();

    for (uint32_t z = 0; z < res_z; z++) {
        VoxelSlice slice = vg.extractSlice(Axis::Z, z);
        ostream << slice << std::endl;
    }
    return ostream;
}

bool VoxelGrid::isInside_impl(const Point3D &p) const {
    if (!bounds.contains(p)) return false;    // outside of bounds

    float stepsize_per_subdivision_x = (bounds.max.x - bounds.min.x) / res_x;
    float stepsize_per_subdivision_y = (bounds.max.y - bounds.min.y) / res_y;
    float stepsize_per_subdivision_z = (bounds.max.z - bounds.min.z) / res_z;

    uint32_t x = uint32_t((p.x - bounds.min.x) / stepsize_per_subdivision_x);
    uint32_t y = uint32_t((p.y - bounds.min.y) / stepsize_per_subdivision_y);
    uint32_t z = uint32_t((p.z - bounds.min.z) / stepsize_per_subdivision_z);

    if (x >= res_x || y >= res_y || z >= res_z) return false;  // invalid
    return isSet(x, y, z);
}
