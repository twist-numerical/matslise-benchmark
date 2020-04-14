#include "catch.hpp"

#include <iostream>
#include <matslise/matslise.h>

using namespace matslise;
using namespace std;
using namespace Eigen;

Y<> dirichlet_y0 = Y<>::Dirichlet();

Matslise<> mathieu_init(double q) {
    return Matslise<>([=](double x) -> double {
        return 2 * q * cos(2 * x);
    }, 0, M_PI, 1e-8);
}

vector<Array<Y<>, Dynamic, 1>> eigenfunctions(
        const Matslise<> &problem, const vector<pair<int, double>> eigenvalues, const ArrayXd &x) {
    vector<Array<Y<>, Dynamic, 1>> r;
    for (auto &iE : eigenvalues) {
        r.push_back(problem.eigenfunction(
                get<1>(iE), dirichlet_y0, x, get<0>(iE)
        ));
    }
    return r;
}


TEST_CASE("Mathieu", "[matslise][mathieu]") {
    BENCHMARK("Mathieu: 11 eigenvalues (q=1)"
    ) {
          return mathieu_init(1).eigenvaluesByIndex(0, 11, dirichlet_y0);
      };

    BENCHMARK("Mathieu: 11 eigenvalues (q=10)"
    ) {
          return mathieu_init(10).eigenvaluesByIndex(0, 11, dirichlet_y0);
      };

    Matslise<> mathieu1 = mathieu_init(1);
    vector<pair<int, double>> eigenvalues1 = mathieu1.eigenvaluesByIndex(0, 11, dirichlet_y0);
    ArrayXd x100 = ArrayXd::LinSpaced(100, 0, M_PI);
    BENCHMARK("Mathieu: 11 eigenfunctions (q=1, n=100)"
    ) {
          eigenfunctions(mathieu1, eigenvalues1, x100);
      };

    ArrayXd x1000 = ArrayXd::LinSpaced(1000, 0, M_PI);
    BENCHMARK("Mathieu: 11 eigenfunctions (q=1, n=1000)"
    ) {
          eigenfunctions(mathieu1, eigenvalues1, x1000);
      };

    ArrayXd x10000 = ArrayXd::LinSpaced(10000, 0, M_PI);
    BENCHMARK("Mathieu: 11 eigenfunctions (q=1, n=10000)"
    ) {
          eigenfunctions(mathieu1, eigenvalues1, x10000);
      };
}