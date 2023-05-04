#include <matslise/matslise.h>
#include <matslise/util/calculateEta.h>
#include <boost/format.hpp>

using boost::multiprecision::float128;

template<typename Scalar>
Eigen::Array<Scalar, Eigen::Dynamic, 1> getZ() {
    Eigen::Array<Scalar, Eigen::Dynamic, 1> Z{58};
    for (int i = 0; i <=     57; ++i) {
        Scalar sz = Scalar(4) - Scalar(.5) * i;
        Scalar z = sz < 0 ? -sz * sz : sz * sz;
        Z(i) = z;
    }
    return Z;
}

template<typename Scalar, int n>
Eigen::Array<Scalar, n, Eigen::Dynamic> getData() {
    return calculateEta<Scalar, n>(getZ<Scalar>());
}

int main() {
    constexpr int n = 10;

    auto data_d = getData<double, n>();
    auto data_ld = getData<long double, n>();
    auto data_q = getData<float128, n>();

    auto Z = getZ<float128>();

    for (int i = 0; i < Z.rows(); ++i) {
        float128 z = Z[i];
        float128 sz = sqrt(abs(z));
        if (z < 0) sz = -sz;
        for (int j = 0; j < n; ++j) {
            std::cout << boost::format(
                    "%1.20f    %2.20f    %3d    %4.16e    %5.20e    %6.32e")
                         % sz % z % (j-1) % (data_d(j, i)) % (data_ld(j, i)) % (data_q(j, i)) << std::endl;
        }
        std::cout << std::endl;
    }


    return 0;
}
