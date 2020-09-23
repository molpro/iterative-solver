#include <boost/multiprecision/float128.hpp>
#include <molpro/Profiler.h>
using boost::multiprecision::float128;

int main(int argc, char* argv[]) {
  auto profiler = molpro::Profiler("Quadruple precision benchmark");
  const size_t n = 1.0e6;
  const size_t repeats = 10;
  const double alpha = 11. / 3. / std::sqrt(n * repeats);
  const double beta = 7. / 6. / std::sqrt(n * repeats);
  auto x = std::vector<double>(n, alpha);
  auto y = std::vector<double>(n, beta);
  auto resd = double{0.};
  auto resq = float128{0.};
  {
    auto p = profiler.push("Double precision");
    for (size_t i = 0; i < repeats; ++i) {
      for (size_t j = 0; j < x.size(); ++j) {
        resd += x[j] * y[j];
      }
      p += n;
    }
  }
  {
    auto p = profiler.push("Quadruple precision");
    for (size_t i = 0; i < repeats; ++i) {
      for (size_t j = 0; j < x.size(); ++j) {
        resq += x[j] * y[j];
      }
      p += n;
    }
  }
  auto reference = float128{alpha} * float128{beta} * n * repeats;
  std::cout.setf(std::ios_base::showpoint);
  std::cout << std::setprecision(std::numeric_limits<double>::max_digits10);
  std::cout << "double    : " << resd << ",  error =" << (resd - reference) << std::endl;
  std::cout << "quadruple : " << resq << ",  error =" << (resq - reference) << std::endl;
  std::cout << profiler.str();
  return 0;
}
