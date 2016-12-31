#ifndef RANDOM_NUM_GENERATOR_H
#define RANDOM_NUM_GENERATOR_H
#include <boost/random/variate_generator.hpp>
#include <boost/random/normal_distribution.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_real.hpp>
#include <boost/random/uniform_int.hpp>
#include <boost/random/discrete_distribution.hpp>
typedef boost::mt19937 base_generator_type;
typedef boost::normal_distribution<float> normal_distribution_type;
typedef boost::uniform_real<float> uniform_distribution_type;
typedef boost::uniform_int<int>  uniform_int_distribution_type;
typedef boost::random::discrete_distribution<> discrete_distribution_type;
typedef boost::variate_generator<base_generator_type&, normal_distribution_type > normal_distribution;
typedef boost::variate_generator<base_generator_type&, uniform_distribution_type > uniform_distribution;
typedef boost::variate_generator<base_generator_type&, discrete_distribution_type > discrete_distribution;
typedef boost::variate_generator<base_generator_type&, uniform_int_distribution_type >   uniform_int_distribution;
#endif // RANDOM_NUM_GENERATOR_H
