#pragma once

#include <iostream>

// TODO: task 7.2 a)
template <class T> class ComplexNumber
{
public:
    ComplexNumber() noexcept = default;
    ComplexNumber(T real, T imaginary = 0.0) : real_number(real), imaginary_number(imaginary) {};

    ComplexNumber& operator+=(const ComplexNumber& rhs);

    template <typename U> friend std::ostream& operator<<(std::ostream& out, const ComplexNumber& complex); // dummy typename to fix compiler issue

    T getRe() const { return real_number; }
    T getIm() const { return imaginary_number; };
private:
    T real_number{};
    T imaginary_number{};
};

// TODO: task 7.2 b) - unary operator+= (class member, modifying)
template <class T> ComplexNumber<T>& ComplexNumber<T>::operator+=(const ComplexNumber<T>& other)
{
    T real = this->real_number + other.getRe();
    T imaginary = this->imaginary_number + other.getIm();

    this->real_number = real;
    this->imaginary_number = imaginary;

    return *this;
}

// TODO: task 7.2 b) - binary operator+ (free function, non-modifying: returns new complex number)
template <class T> ComplexNumber<T> operator +(const ComplexNumber<T>& lhs, const ComplexNumber<T>& rhs)
{
    T real = lhs.getRe() + rhs.getRe();
    T imaginary = lhs.getIm() + rhs.getIm();

    return ComplexNumber<T>(real, imaginary);
}

template <class T> std::ostream& operator<<(std::ostream& out, const ComplexNumber<T>& complex)
{
    out << complex.getRe() << "+" << complex.getIm() << "i";
    return out;
}
