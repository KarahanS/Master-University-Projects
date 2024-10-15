#pragma once
bool isAlmostEqual(float x1, float x2, float epsilon = 1.0e-5f);
float roundValToNDecimals(float val, unsigned int n);
float estimateFunctionRoot(float (*linearFunc)(float), float xLower, float xUpper, unsigned int nDecimals);
void testEstimateFunctionRoot();
