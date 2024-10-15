#pragma once
#include <vector>
#include <functional>

// TODO 7.1.a: Implement the square function
template <typename T> T square(T a) {
    return a*a;
}

// TODO 7.1.b: Implement the halve function
template <typename T> double halve(T a) {
    double casted = static_cast<double>(a);
    return casted/2;
}
// TODO 7.1.c: Implement the add function
template <typename T> T add(T a, T b) {
  return a+b;
}
// TODO 7.1.d: Implement the multiply function
template <typename T> T multiply(T a, T b) {
  return a*b;
}

// TODO 7.1.d: Implement the reduce function
template <typename T> T reduce(std::function<T(T, T)> op, std::vector<T> vals, T neutral) {
    T result = neutral;
    for (size_t i = 0; i < vals.size(); i++) {
        result = op(result, vals[i]);
    }
    return result;
}
// TODO 7.1.f: Implement the map function
template <typename T> std::vector<T> map(std::function<T(T)> op, std::vector<T> vals) {
    std::vector<T> result;
    for (size_t i = 0; i < vals.size(); i++) {
        result.push_back(op(vals[i]));
    }
    return result;
}