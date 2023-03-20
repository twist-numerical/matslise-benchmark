#include <iostream>
#include <boost/format.hpp>
#include <boost/math/constants/constants.hpp>
#include <boost/multiprecision/float128.hpp>
#include <matslise/matslise.h>

using boost::math::constants::pi;
using boost::multiprecision::float128;

float128 mathieuPotential(float128 x) {
    return 2 * cos(2 * x);
}

int main() {
    matslise::Matslise<float128> problem(
            &mathieuPotential, 0, pi<float128>(), 1e-25q);

    auto boundary = matslise::Y<float128>::Dirichlet();
    auto eigs = problem.eigenvaluesByIndex(0, 10, boundary);
    for (auto [i, E]: eigs) {
        float128 error = problem.eigenvalueError(E, boundary, i);
        std::cout << boost::format(
                "Eigenvalue %1$d:%2$30.25f  (error: %3$.1e)")
                     % i % E % error << std::endl;
    }

    return 0;
}
