package uk.ac.shef.wit.nhmm;

import static java.lang.Math.*;
import static uk.ac.shef.wit.nhmm.Constants.COMP_THRESHOLD;

public class Simulation {
    static boolean is_first_normal_value = true;

   static int generateBernoulli(double[] dist, int dim) {
  /* Generates a state (1 through dim) accrding to a Bernoulli distribution with dim states and distribution
     dist for the states.  Assumes that the total probability mass in dist adds up to 1. */

        int state = 0;     // Generated state
        double sum = 0.0;   // Probability mass counter
        double dice_roll; // Random number

        dice_roll = Constants.drand48();
        while (sum < dice_roll) {
            sum += dist[state];
            state++;
        }

        if (state == 0)
            return (0);
        else
            return (state - 1);
    }

    static double second_value;      // Second value (to be carried over for the next run)

    static double generateUnitNormal() {
        /* Generating a univariate normally distributed value with zero mean and unit variance */
        /* Box-Muller */

        /* Following "Numerical Recipes in C", 2nd edition */

        /* Two generated random variables */
        double first_value;              // First

        double v1, v2;                   // Uniformly distributed in (-1,1)
        double d;                        // Length of (v1,v2)
        double fac;                      // Multiplication factor

        if (is_first_normal_value) { /* Simulating two normal values */
            do {
                v1 = 2.0 * Constants.drand48() - 1.0;
                v2 = 2.0 * Constants.drand48() - 1.0;
                d = v1 * v1 + v2 * v2;
            }
            while (d >= 1.0 || abs(d) <= COMP_THRESHOLD);

            fac = sqrt(-2.0 * log(d) / d);
            first_value = v1 * fac;
            second_value = v2 * fac;
            is_first_normal_value = false;

            return (first_value);
        } /* Simulating two normal values */ else { /* Using left-over value */
            is_first_normal_value = true;
            return (second_value);
        } /* Using left-over value */
    }

    static double generateExponential(double lambda) {
        /* Generating a sample from a univariate exponential distribution f(x)=l*exp(-l*x) */
        return (-log(1.0 - Constants.drand48()) / lambda);
    }

    static double generateGamma(double a, double b) {
        /* Generating a sample from a Gamma distribution f(x)=x^(a-1)exp(-b*x)*b^a/G(a) */

        double x = 0.0;
        double u, y, v, c, p;

        boolean done = false;

        /* First sampling from f(x)=x^(a-1)exp(-x)/G(a) */

  /* The method implmented below is based on rejection method as sugested by
     Ahrens and summarized in "The Art of Computer Programming" by D. Knuth */

        /* Cases of a>1 and 0<a<=1 are treated separately */
        if (a > 1.0)
            while (!done) {
                u = Constants.drand48();
                y = tan(PI * u);
                x = sqrt(2.0 * a - 1.0) * y + a - 1.0;
                if (x > 0) {
                    /* Rejection in action */
                    v = Constants.drand48();
                    if (v <= (1.0 + y * y) * exp((a - 1.0) * log(x / (a - 1.0)) - sqrt(2.0 * a - 1.0) * y))
                        /* Value under the curve -- accepting */
                        done = true;
                }
            }
        else if (a > 0.0) {
            c = (exp(1.0) + a) / exp(1.0);
            while (!done) {
                u = Constants.drand48();
                p = c * u;
                if (p > 1.0) {
                    x = -log((c - p) / a);
                    v = Constants.drand48();
                    if (v <= pow(x, a - 1.0))
                        done = true;
                } else {
                    x = pow(p, 1.0 / a);
                    u = Constants.drand48();
                    if (u <= exp(-x))
                        done = true;
                }
            }
        }

        return (x / b);
    }

    static double generateLognormal(double M, double S2) {
        double x;

        x = exp(sqrt(S2) * generateUnitNormal() + M);

        return (x);
    }

    static void SimulateNormal(double[] mu, double[][] Sigma, double[] x, int dim) {
  /* Simulating x according to the multivariate normal distribution with mean mu
     and upper-triangle Cholesky decomposition Sigma of the covariance matrix */

        double[] unitnormal;

        if (dim != 1) { /* Multivariate normal */
            unitnormal = new double[dim];

            for (int i = 0; i < dim; i++) {
                unitnormal[i] = generateUnitNormal();
                x[i] = mu[i];
                for (int j = 0; j <= i; j++)
                    x[i] += unitnormal[j] * Sigma[j][i];
            }

            /* Applying the covariance structure */

            unitnormal = null;
        } /* Multivariate normal */ else
            x[0] = mu[0] + Sigma[0][0] * generateUnitNormal();
    }
}