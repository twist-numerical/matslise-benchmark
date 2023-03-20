#include <iostream>
#include <boost/format.hpp>
#include <boost/math/constants/constants.hpp>
#include <boost/multiprecision/float128.hpp>
#include <matslise/matslise.h>

using boost::math::constants::pi;
using boost::multiprecision::float128;


int main() {
    matslise::Matslise<float128> problem(
            [](float128 x) { return exp(x); }, 0, pi<float128>(), 1e-32q);

    auto boundary = matslise::Y<float128>::Dirichlet();
    auto eigs = problem.eigenvaluesByIndex(0, 12, boundary);
    for(auto[i, E] : eigs){
        auto error = problem.eigenvalueError(E, boundary, boundary, i);
        std::cout << boost::format(
                "Eigenvalue %1$d:%2$30.24f  (error: %3$.1e)")
                     % i % E % error << std::endl;
    }

    for (auto [i, E0]: std::vector<std::pair<int, float128>>{{0, 3},
                                                             {2, 18},
                                                             {9, 102}}) {
        float128 E = E0;
        std::vector<float128> results;
        results.push_back(E);
        float128 err, derr, theta;
        do {
            std::tie(err, derr, theta) = problem.matchingError(E, boundary, boundary);
            E -= err / derr;
            results.push_back(E);
        } while (abs(err) > 1e-32);

        while (results.back() == 0)
            results.pop_back();

        float128 exact = eigs[i].second;
        if (abs(exact - E) > 1e-10)
            throw std::runtime_error("Converged to the wrong value.");

        std::cout << i << " " << exact;
        for (auto v: results) {
            std::cout << " " << abs(exact - v)/exact;
        }
        std::cout << std::endl;
    }

    return 0;
}
