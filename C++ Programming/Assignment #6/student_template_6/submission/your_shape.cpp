#include "shapes.h"
#include "transformations.h"
#include "operations.h"
#include "voxel_grid.h"

#include "shapes_test.h"

// example shapes
// TODO: enable this in the CMakelists.txt when you are finished with your implementation to see some example shapes
#ifdef ENABLE_DEMO_CODE
Shape outline(const VoxelGrid& vg) {
    constexpr float voxel_size = 1.0f/VoxelGrid::level_of_detail;
    Shape s {vg.clone()}; // create a clone - this creates a reference that can be used multiple times without copying
    
    return (s.translated({voxel_size, voxel_size, 0.0f})|s.translated({voxel_size, -voxel_size, 0.0f})|s.translated({-voxel_size, voxel_size, 0.0f})|s.translated({-voxel_size, -voxel_size, 0.0f}))^s;
}

Shape cpp2023() {
    const Shape o    = Cylinder{} - Cylinder{}.scaled({0.5f, 0.5f, 1.0f});
    const Shape c    = o - Cube{}.scaled({0.5f, 1.0f, 1.0f}).translated({0.5f, 0.0f, 0.0f});
    const Shape plus = Cube{}.scaled({0.2f, 0.6f, 1.0f}) + Cube{}.scaled({0.6f, 0.2f, 1.0f});
    const Shape cpp  = c + plus.translated({0.8f, 0.0f, 0.0f}) + plus.translated({2.3f, 0.0f, 0.0f});
    const Shape zero = o.scaled({0.6f, 1.0f, 1.0f});
    const Shape two  = zero - Cube{}.scaled({0.6f, 0.5f, 1.0f}).translated({0.0f, 0.5f, 0.0f}) + Cube{}.scaled({0.6f, 0.15f, 1.0f}).translated({0.0f, 0.85f, 0.0f})
                     + Cube{}.scaled({0.6f, 0.15f, 1.0f}).rotated(Axis::Z, -40.0f*static_cast<float>(M_PI)/180.0f).translated({0.0f, 0.35f, 0.0f})
                     + Cylinder{}.scaled({0.15f, 0.15f, 1.0f}).translated({0.45f, 0.0f, 0.0f}) + Cylinder{}.scaled({0.15f, 0.15f, 1.0f}).translated({-0.45f, 0.75f, 0.0f});
    const Shape three = (o & Cube{}.scaled({0.5f, 0.8f, 1.0f}).translated({0.5f,  0.2f, 0.0f})).scaled({1.0f, 0.6f, 1.0f}).translated({-0.6f,  0.4f, 0.0f})
                      + (o & Cube{}.scaled({0.5f, 0.8f, 1.0f}).translated({0.5f, -0.2f, 0.0f})).scaled({1.0f, 0.6f, 1.0f}).translated({-0.6f, -0.4f, 0.0f});
    const Shape one  = Cube{}.scaled({0.15f, 1.0f, 1.0f}).translated({0.15f, 0.0f, 0.0f}) + Cube{}.scaled({0.3f, 0.15f, 1.0f}).rotated(Axis::Z, -27.0f*static_cast<float>(M_PI)/180.0f).translated({-0.1f, -0.75f, 0.0f});

    const Shape twenty_twenty_three = two+zero.translated({1.4f, 0.0f, 0.0f})+two.translated({2.8f, 0.0f, 0.0f})+three.translated({4.2f, 0.0f, 0.0f});

    return (cpp+twenty_twenty_three.translated({3.8f, 0.0f, 0.0f})).scaled({1.0f, 1.0f, 0.1f});
}

Shape NeverGonnaGiveYouUp() {
    const Shape N = Cube{}.scaled({0.15f, 0.9f, 1.0f}).translated({0.0f, 0.0f, 0.0f})
              + Cube{}.scaled({0.15f, 0.9f, 1.0f}).translated({1.0f, 0.0f, 0.0f})
              + Cube{}.scaled({0.15f, 0.9f, 1.0f}).rotated(Axis::Z, 25.0f * M_PI / 180.0f)
                  .translated({0.5f, 0.0f, 0.0f});

    const Shape E    = Cube{}.scaled({0.15f, 0.7f, 1.0f}).translated({0.0f, 0.0f, 0.0f})   
                + Cube{}.scaled({0.6f, 0.15f, 1.0f}).translated({0.45f, 0.8f, 0.0f})    
                + Cube{}.scaled({0.6f, 0.15f, 1.0f}).translated({0.45f, 0.00f, 0.0f})    
                + Cube{}.scaled({0.6f, 0.15f, 1.0f}).translated({0.45f, -0.8f, 0.0f});   


    // Shape for 'V'
    const Shape V = Cube{}.scaled({0.15f, 1.0f, 1.0f}).rotated(Axis::Z, -30.0f * static_cast<float>(M_PI) / 180.0f).translated({0.5f, 0.0f, 0.0f})   // Left diagonal line
              + Cube{}.scaled({0.15f, 1.0f, 1.0f}).rotated(Axis::Z, 30.0f * static_cast<float>(M_PI) / 180.0f).translated({-0.5f, 0.0f, 0.0f});  // Right diagonal line

    // Shape for 'R'
    
    //const Shape R = Cube{}.scaled({0.15f, 0.9f, 1.0f})
    //                + Cylinder{}.scaled({0.6f, 0.4f, 1.0f}).translated({0.45f, -0.5f, 0.0f}) 
    //                + Cube{}.scaled({0.10f, 0.95f, 1.0f}).rotated(Axis::Z, -20.0f * static_cast<float>(M_PI) / 180.0f).translated({0.55f, 0.0f, 0.0f});
                    
    
    // Shape for 'I'
    const Shape I    = Cube{}.scaled({0.15f, 0.9f, 1.0f}).translated({0.0f, 0.0f, 0.0f});
    const Shape o    = (Cylinder{} - Cylinder{}.scaled({0.8f, 0.8f, 1.0f})).scaled({1.0f, 0.9f, 0.1f});
    const Shape c    = o - Cube{}.scaled({0.5f, 1.0f, 1.0f}).translated({0.8f, 0.0f, 0.0f});
    const Shape G = c  + I.scaled({0.6f, 0.5f, 1.0f}).translated({0.25f, 0.45f, 0.0f}) 
    + I.scaled({0.5f, 0.3f, 1.0f}).translated({0.1f, 0.10f, 0.0f}).rotated(Axis::Z, -90.0f * static_cast<float>(M_PI) / 180.0f);


    const Shape O = Cylinder{}.scaled({1.0f, 1.0f, 0.2f})  // Outer ring
               - Cylinder{}.scaled({0.6f, 0.6f, 0.2f});

    const Shape U1 = O - Cube{}.scaled({1.5f, 0.4f, 1.0f}).translated({0.5f, -0.7f, 0.0f})
                    + Cube{}.scaled({0.2f, 0.5f, 1.0f}).translated({-0.75f, -0.45f, 0.0f})
                    + Cube{}.scaled({0.2f, 0.5f, 1.0f}).translated({0.75f, -0.45f, 0.0f}) ;
    const Shape U2 = O - Cube{}.scaled({1.5f, 0.4f, 1.0f}).translated({0.5f, -0.7f, 0.0f})
                    + Cube{}.scaled({0.2f, 0.6f, 1.0f}).translated({-0.75f, -0.45f, 0.0f})
                    + Cube{}.scaled({0.2f, 0.6f, 1.0f}).translated({0.75f, -0.45f, 0.0f})
                    - Cube{}.scaled({0.1f, 0.15f, 1.0f}).translated({1.0f, -0.05f, 0.0f}) ;

    const Shape inner_cylinder = Cylinder{}.scaled({0.4f, 0.3f, 1.0f}).translated({0.3f, 0.0f, 0.0f});  // Inner hollow part (subtracted)
    const Shape outer_cylinder = Cylinder{}.scaled({0.5f, 0.6f, 1.0f}).translated({0.4f, 0.0f, 0.0f});  // Outer round part
    // Create the "P" shape by subtracting the inner cylinder from the outer one
    const Shape rounded_part = outer_cylinder - inner_cylinder;

    // Add a vertical bar to form the spine of the "P"
    const Shape spine = Cube{}.scaled({0.15f, 1.0f, 1.0f}).translated({-0.1f, 0.0f, 0.0f});

    // Combine both parts to form the letter "P"
    const Shape P =(rounded_part.translated({0.0f, -0.45f, 0.0f}) + spine);
    const Shape R = P + Cube{}.scaled({0.1f, 0.6f, 1.0f}).rotated(Axis::Z, +30.0f * static_cast<float>(M_PI) / 180.0f).translated({0.5f, 0.5f, 0.0f});

    // Combine the legs and the crossbar to form the "A"
    const Shape A =  Cube{}.scaled({0.15f, 1.0f, 1.0f}).rotated(Axis::Z, 30.0f * static_cast<float>(M_PI) / 180.0f).translated({0.5f, 0.0f, 0.0f})   // Left diagonal line
            + Cube{}.scaled({0.15f, 1.0f, 1.0f}).rotated(Axis::Z, -30.0f * static_cast<float>(M_PI) / 180.0f).translated({-0.5f, 0.0f, 0.0f}) // Right diagonal line
            + Cube{}.scaled({0.6f, 0.1f, 1.0f}).translated({0.2f, 0.3f, 0.0f});
            

   const Shape Y = Cube{}.scaled({0.1f, 0.8f, 1.0f})
                     .rotated(Axis::Z, -45.0f * static_cast<float>(M_PI) / 180.0f)
                     .translated({0.4f, -0.5f, 0.0f})
              + Cube{}.scaled({0.1f, 0.8f, 1.0f})
                     .rotated(Axis::Z, 45.0f * static_cast<float>(M_PI) / 180.0f)
                     .translated({-0.4f, -0.5f, 0.0f})
              + Cube{}.scaled({0.15f, 0.5f, 1.0f})
                     .translated({0.0f, 0.5f, 0.0f});

    
    const Shape never = N.translated({0.1f, 0.0f, 0.0f}) + E.translated({1.8f, 0.0f, 0.0f}) + V.translated({4.4f, 0, 0}) +  E.translated({6.0f, 0.0f, 0.0f}) + R.translated({7.7f, 0.0f, 0.0f}).scaled({1.0f, 0.9f, 1.0f});
    const Shape gonna = G + O.translated({1.8f, 0.0f, 0.0f}).scaled({1.0f, 0.9f, 1.0f}) + N.translated({3.4f, 0.0f, 0.0f}) + N.translated({5.1f, 0.0f, 0.0f}) + A.translated({7.8f, 0.0f, 0.0f});
    const Shape give =  G + I.translated({0.9f, 0.0f, 0.0f}) + V.translated({2.5f, 0.0f, 0.0f}) + E.translated({4.2f, 0.0f, 0.0f});
    const Shape you =  Y + O.translated({2.3f, 0.0f, 0.0f}) + U1.translated({4.6f, 0.0f, 0.0f});
    const Shape up   = U2 + P.translated({1.8f, 0.0f, 0.0f});
    
    //return up.scaled({1.0f, 1.0f, 0.1f});
    return (never.translated({-0.1f, 0.0f, 0.0f}) + gonna.translated({0.2f, 2.2f, 0.0f}) + give.translated({1.8f, 4.4f, 0.0f}) + you.translated({1.7f, 6.7f, 0.0f}) + up.translated({2.8f, 9.0f, 0.0f})).scaled({1.0f, 1.0f, 0.1f});

}

// base case for variadic template
Shape shape_list() {
    return Empty{}.clone();
}

// variadic template
// you can create arbitrarily long shape lists with this, you don't have to understand how this works (yet)
#if __cpp_lib_concepts >= 202002L
// only available if concepts are supported by your c++ standard library
template<ShapeFullyImplemented FirstShape, ShapeFullyImplemented... RemainingShapes>
#else
template<typename FirstShape, typename... RemainingShapes>
#endif
Shape shape_list(FirstShape first, RemainingShapes... rest) {
    return first + shape_list(rest...).translated({2.0f, 0.0f, 0.0f});
}

Shape composite_shape() {
    return (((!(Sphere{}-Cube{}.scaled(0.5f).translated(0.5f)))&Cylinder{})+Octahedron{}.scaled(0.5f).translated({0.0f, 0.0f, -0.5f})).rotated(Axis::Z, -45.0f*static_cast<float>(M_PI)/180.0f);
}

Shape example_shape() {
    // return shape_list(Cube{}, Sphere{}, Octahedron{}, Cylinder{});
    // return composite_shape();
    return outline(NeverGonnaGiveYouUp());
}
#else
// fallback example shape (returns a simple cube)
Shape example_shape() {
    return Cube{}.clone();
}
#endif // end example shapes


/// implementation of your custom shape (bonus task)
Shape your_shape() {
    // return whatever shape you like in here - this file does not influence the evaluation of your submission (except potential bonus points)

    return example_shape();
}
