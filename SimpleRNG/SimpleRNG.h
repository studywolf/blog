#ifndef SIMPLERNG_H
#define SIMPLERNG_H

// A simple random number generator based on George Marsaglia's MWC (Multiply With Carry) generator.
// This is not intended to take the place of the library's primary generator, Mersenne Twister.
// Its primary benefit is that it is simple to extract its state.

class SimpleRNG
{
public:
    
    SimpleRNG();

    // Seed the random number generator 
    void SetState(unsigned int u, unsigned int v);

    // Extract the internal state of the generator
    void GetState(unsigned int& u, unsigned int& v);

    // A uniform random sample from the open interval (0, 1) 
    double GetUniform();

    // A uniform random sample from the set of unsigned integers 
    unsigned int GetUint();

    // This stateless version makes it more convenient to get a uniform 
    // random value and transfer the state in and out in one operation.
    double GetUniform(unsigned int& u, unsigned int& v);

    // This stateless version makes it more convenient to get a random unsigned integer 
    // and transfer the state in and out in one operation.
    unsigned int GetUint(unsigned int& u, unsigned int& v);
        
    // Normal (Gaussian) random sample 
    double GetNormal(double mean, double standardDeviation);

    // Exponential random sample 
    double GetExponential(double mean);

	// Gamma random sample
    double GetGamma(double shape, double scale);

	// Chi-square sample
    double GetChiSquare(double degreesOfFreedom);

	// Inverse-gamma sample
    double GetInverseGamma(double shape, double scale);

	// Weibull sample
    double GetWeibull(double shape, double scale);

	// Cauchy sample
    double GetCauchy(double median, double scale);

	// Student-t sample
    double GetStudentT(double degreesOfFreedom);

    // The Laplace distribution is also known as the double exponential distribution.
    double GetLaplace(double mean, double scale);

	// Log-normal sample
    double GetLogNormal(double mu, double sigma);

	// Beta sample
    double GetBeta(double a, double b);

	// Poisson sample
	int GetPoisson(double lambda);

private:
    unsigned int m_u, m_v;
	int PoissonLarge(double lambda);
	int PoissonSmall(double lambda);
	double LogFactorial(int n);
};


#endif