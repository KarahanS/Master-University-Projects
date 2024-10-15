/*
Oguz Ata Cal    - 6661014  
Karahan Saritas - 6661689
*/

// estimate_function_root.cpp - student template:
// Estimate the root (or x) of a given linear function, that is, f(x) = 0 using interval bisection.
#include <iostream>
#include <stdio.h>
#include <cmath>

/* Given: increasing linear function to test f(x)=32*x-1 */
float exampleIncreasingLinearFunc(float x)
{
    return 32 * x - 1;
}

/* Given: decreasing linear function to test f(x)=-32*x-1 */
float exampleDecreasingLinearFunc(float x)
{
    return -32 * x - 1;
}
float func(float x)
{
    return 5 * x - 7;
}

/* TODO: 1a)
Rounds a float value to n decimal places. E.g. val=1.555, n=2 returns 1.56*/
float roundValToNDecimals(float val, unsigned int n)
{
    //TODO: 1a)
    float div = float(std::pow(10.0, n));             // 100
    return float(std::round(val * div)) / div;        //  1.555 * 100 = 155.5   -->  round(155.5) = 156  -->  156/100 = 1.56
}

/* TODO: 1b)
 Returns true if the absolute difference of x1 and x2 is smaller or equal than a given epsilon,
otherwise returns false. Default epsilon checks for 5 decimal precision.*/
bool isAlmostEqual(float x1, float x2, float epsilon = 1.0e-5)
{
    //TODO: 1b)
    float diff = float(std::abs(x1 - x2));
    return diff <= epsilon;
}

/* TODO: 1c-f)
Estimates the root, the x, of the linear function that is f(x)=0 within the specified interval [xLower, xUpper].
If the root is equal to the interval bounds or the midpoint it returns the corresponding x-value.
If the root is estimated (the intervall becomes small enough) the resulting x-value is rounded to n decimal places. 
If linear function in the specified interval does not have a root, it returns NAN. */
float estimateFunctionRoot(float (*linearFunc)(float), float xLower, float xUpper, unsigned int nDecimals)
{
    float xMiddle = xLower + (xUpper - xLower) / 2; 
    float yLower = linearFunc(xLower);
    float yUpper = linearFunc(xUpper);
    float yMiddle = linearFunc(xMiddle);

    // are we done?
    if(isAlmostEqual(xLower, xUpper)) {
        if( (yLower > 0 && yUpper > 0) || (yLower < 0 && yUpper < 0) )
            return NAN;  // root is not present
        else
            return roundValToNDecimals(xLower, nDecimals);
    }

    // check the boundaries first
    if(isAlmostEqual(yLower, 0.0)) return xLower;  
    if(isAlmostEqual(yMiddle, 0.0)) return xMiddle;
    if(isAlmostEqual(yUpper, 0.0)) return xUpper;

    if((yLower > 0 && yMiddle < 0) || (yLower < 0 && yMiddle > 0)) return estimateFunctionRoot(linearFunc, xLower, xMiddle, nDecimals);
    else if((yMiddle > 0 && yUpper < 0) || (yMiddle < 0 && yUpper > 0)) return estimateFunctionRoot(linearFunc, xMiddle, xUpper, nDecimals);
    else return NAN;  // root is not present

}

/* Calls estimateFunctionRoot of increasing example function with the specified interval [lowerBound; upperBound] and prints out the result. */
void testAndPrint(float (*exampleFunc)(float), float xLower, float xUpper, unsigned int nDecimals = 5)
{
    float result = estimateFunctionRoot(exampleFunc, xLower, xUpper, nDecimals);
    std::cout << "interval [" << xLower << ", " << xUpper << "], \t result = " << result << std::endl;
}

/* Test your implementation */
void testEstimateFunctionRoot()
{ 
    // Test 1a) Round up with 1 decimal, round down with 1 decimal, round with 4 decimal precision
    std::cout << "Round 1.55 to 1 decimal: " << roundValToNDecimals(0.49, 0) << ", expected: " << 0 << std::endl;
    std::cout << "Round 1.55 to 1 decimal: " << roundValToNDecimals(1.55, 0) << ", expected: " << 2 << std::endl;
    std::cout << "Round 1.55 to 1 decimal: " << roundValToNDecimals(1.55, 1) << ", expected: " << 1.6 << std::endl;
    std::cout << "Round 1.54 to 1 decimal: " << roundValToNDecimals(1.54, 1) << ", expected: " << 1.5 << std::endl << std::endl;

    // Test 1b) abs(x1 - x2) <= epsilon
    // x1 = 0.1, x2 = 0.11, decimal precision (epsilon) varies
    std::cout << "Is x1 = 0.1 almost equal to x2 = 0.11 with 1 decimal precision?: " << isAlmostEqual(0.1f, 0.11f, 0.1f) << ", expected: " << "1" << std::endl;
    std::cout << "Is x1 = 0.1 almost equal to x2 = 0.11 with 2 decimal precision?: " << isAlmostEqual(0.1f, 0.11f, 1e-2f) << ", expected: " << "1" << std::endl;
    std::cout << "Is x1 = 0.1 almost equal to x2 = 0.11 with 3 decimal precision?: " << isAlmostEqual(0.1f, 0.11f, 1e-3f) << ", expected: " << "0" << std::endl << std::endl;

    // Precision with 5 decimals (default of function)
    std::cout << "Is x1 = 0.00001 almost equal to x2 = 0.000019 with 5 decimal precision?: " << isAlmostEqual(0.00001f, 0.000019f) << ", expected: " << "1" << std::endl;
    std::cout << "Is x1 = 0.00001 almost equal to x2 = 0.00002 with 5 decimal precision?: " << isAlmostEqual(0.00001f, 0.00002f) << ", expected: " << "1" << std::endl;
    std::cout << "Is x1 = 0.00001 almost equal to x2 = 0.000021 with 5 decimal precision?: " << isAlmostEqual(0.00001f, 0.000021f) << ", expected: " << "0" << std::endl << std::endl;

    
    // Test 1c) Estimate rounded
    testAndPrint(&exampleIncreasingLinearFunc, 0.01f, 1.5f);
    // Test 1d) Special cases: xUpper, xLower, midPoint (no rounding necessary)
    // f(x) = 0 for x = xUpper
    testAndPrint(&exampleIncreasingLinearFunc, 0, 0.03125);


    // f(x) = 0 for x = xLower
    testAndPrint(&exampleIncreasingLinearFunc, 0.03125, 1);

    // f(x) = 0 for x = midpoint = (xUpper - xLower)/2
    testAndPrint(&exampleIncreasingLinearFunc, 0, 0.03125 * 2);

    // Test 1e) Special case: no root found, return NAN
    testAndPrint(&exampleIncreasingLinearFunc, 0.0f, 0.02f);

    // Test 1f) Test decreasing function as well
    testAndPrint(&exampleDecreasingLinearFunc, -1.0f, 1.0f);

    // 
    testAndPrint(&func, -5.0f, 6.0f);
    testAndPrint(&func, 1.4f, 1.4f);
    testAndPrint(&func, 0.0f, 0.0f);
    testAndPrint(&func, 1.39f, 1.41f);

}