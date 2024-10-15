#include "submission/shapes.h"
#include "submission/transformations.h"
#include "submission/operations.h"
#include "submission/voxel_grid.h"

#include <cstdlib>
#include <iostream>
#include <stdexcept>

Shape your_shape(); // implemented in your_shape.cpp

int main() {
    // until implemented in task 6.4, this will throw an exception
    try {
        VoxelGrid vg = your_shape();
        std::cout << vg;
    }
    catch (std::logic_error& e) {
        std::cout << "you need to implement " << e.what() << " to get some output here!" << std::endl;
    }

    return EXIT_SUCCESS;
}
