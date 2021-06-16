package uk.ac.shef.wit.nhmm;

import java.util.Random;

import static java.lang.Double.isNaN;
import static java.lang.Math.log;

public class Constants {

    /* COMPILE CONSTANTS */

    public static final boolean FLAG_KL = false;
    public static final boolean BLAH = false;


    /* Constants */

    public static final boolean INPUT_VERBOSE = true;
    public static final boolean SIM_VERBOSE = true;
    public static final boolean HOLE_VERBOSE = true;
    public static final boolean VITERBI_VERBOSE = true;
    public static final boolean VERBOSE_BRIEF = true;
    public static final boolean XVAL_VERBOSE = true;
    public static final boolean NEWTONRAPHSON_VERBOSE = true;

    public static final boolean MEASURE_TIME = true;

    /* Maximum number of hidden states */
    //int MAXNUMSTATES = 20;
    public static final int MAXNUMVARIABLES = 20;      // Ceiling on the number of latent variables (in the Markov Chain)

    /* Data types */
    public static final int DATA_TYPE_FINITE = 0;
    public static final int DATA_TYPE_REAL = 1;

    /* Model types */
    public enum ModelType {
        hmm, nhmm, mix, nmix("non-homogeneous mixture");

        private final String description;

        ModelType() {
            description = name();
        }

        ModelType(String description) {
            this.description = description;
        }

        public String getDescription() {
            return description;
        }
    }

//    public static final int MODEL_TYPE_HMM = 0;      // Hidden Markov model
//    public static final int MODEL_TYPE_NHMM = 1;      // Non-homogeneous hidden Markov model
//    public static final int MODEL_TYPE_MIX = 2;      // Mixture model
//    public static final int MODEL_TYPE_NMIX = 3;      // Non-homogeneous mixture model

    /* Distribution type codes */

    public enum DistributionType {
        /* Complex */
        DIST_FACTOR("Product of (conditionally) independent distributions"),
        DIST_MIXTURE("Mixture of distributions"),
        DIST_CLMIXTURE("Mixture with tree dependence"),

        /* Univariate finite-valued */
        bernoulli("Bernoulli distribution (univariate multinomial)"),
        DIST_CONDBERNOULLI("Conditional Bernoulli distribution (conditional univariate multinomial)"),
        DIST_CONDBERNOULLIG("Same as before but initial sequence value is computed by"),
        // averaging all entries
        DIST_UNICONDMVME("Univariate conditional multi-values MaxEnt model"),

        /* Multivariate finite-valued */
        DIST_CHOWLIU("Distribution with a Chow-Liu tree structure"),
        DIST_CONDCHOWLIU("Distribution with a conditional Chow-Liu tree structure"),
        DIST_ME_BIVAR("Full bivariate MaxEnt"),
        DIST_BN_ME_BIVAR("Bayesian network decomposition using binary uni- and bi-variate MaxEnt"),
        DIST_BN_CME_BIVAR("Same as previous but for conditional distributions"),

        /* Univariate real-valued */
        DIST_DELTAEXP("Delta-exponential mixture"),
        DIST_DELTAGAMMA("Delta-gamma mixture"),
        DIST_DIRACDELTA("Dirac delta function"),
        DIST_EXP("Exponential (geometric) distribution"),
        DIST_GAMMA("Gamma distribution"),
        DIST_LOGNORMAL("Log-normal distribution"),

        /* Multivariate real-valued */
        DIST_NORMAL("Gaussian distribution"),
        DIST_NORMALCHAIN("Gaussian dependent on the previous value"),
        DIST_NORMALCL("Tree-structured Gaussian"),

        /* Logistic */
        DIST_LOGISTIC("Logistic distribution"),
        DIST_TRANSLOGISTIC("Logistic distribution with transition terms"),
        DIST_TRANSLOGISTICG("Same as TRANSLOGISTIC but initial sequence value is computed by"),
        // averaging over all entries

        /* Aliases */
        independent("Conditionally independent distributions"),
        DIST_ALIAS_MIXTURE("Mixture distribution"),
        DIST_ALIAS_CLMIXTURE("Tree-dependent mixture distribution"),
        DIST_MIXTURE_PRIOR("Mixture distribution with Dirichlet prior on mixing probabilities"),
        DIST_ALIAS_MIXTURE_PRIOR("Mixture distribution with Dirichlet prior on the mixing probabilities"),
        DIST_ALIAS_CLMIXTURE_PRIOR("Tree-dependent mixture distribution with a prior on edges and parameters"),
        DIST_CHOWLIU_MDL("Distribution with a Chow-Liu forest with MDL prior"),
        DIST_CHOWLIU_DIR_MDL("Chow-Liu forests with MDL prior on structures and Dirichlet prior on parameters"),
        DIST_CONDCHOWLIU_MDL("Distribution with a conditional Chow-Liu forests with MDL prior"),
        DIST_BERNOULLI_PRIOR("Multinomial distribution with Dirichlet prior"),
        DIST_CONDBERNOULLI_PRIOR("Conditional univariate multinomial with Dirichlet prior"),
        DIST_CONDBERNOULLIG_PRIOR("Same");

        private final String description;

        DistributionType() {
            description = name();
        }

        DistributionType(String description) {
            this.description = description;
        }

        public String getDescription() {
            return description;
        }
    }


//    /* Complex */
//    public static final int DIST_FACTOR = 101; // Product of (conditionally) independent distributions
//    public static final int DIST_MIXTURE = 102; // Mixture of distributions
//    public static final int DIST_CLMIXTURE = 103; // Mixture with tree dependence
//
//    /* Univariate finite-valued */
//    public static final int DIST_BERNOULLI = 1; // Bernoulli distribution (univariate multinomial)
//    public static final int DIST_CONDBERNOULLI = 2; // Conditional Bernoulli distribution (conditional univariate multinomial)
//    public static final int DIST_CONDBERNOULLIG = 3; // Same as before but initial sequence value is computed by
//    // averaging all entries
//    public static final int DIST_UNICONDMVME = 4; // Univariate conditional multi-values MaxEnt model
//
//    /* Multivariate finite-valued */
//    public static final int DIST_CHOWLIU = 11; // Distribution with a Chow-Liu tree structure
//    public static final int DIST_CONDCHOWLIU = 12; // Distribution with a conditional Chow-Liu tree structure
//    public static final int DIST_ME_BIVAR = 13; // Full bivariate MaxEnt
//    public static final int DIST_BN_ME_BIVAR = 14; // Bayesian network decomposition using binary uni- and bi-variate MaxEnt
//    public static final int DIST_BN_CME_BIVAR = 15; // Same as previous but for conditional distributions
//
//    /* Univariate real-valued */
//    public static final int DIST_DELTAEXP = 21; // Delta-exponential mixture
//    public static final int DIST_DELTAGAMMA = 22; // Delta-gamma mixture
//    public static final int DIST_DIRACDELTA = 23; // Dirac delta function
//    public static final int DIST_EXP = 24; // Exponential (geometric) distribution
//    public static final int DIST_GAMMA = 25; // Gamma distribution
//    public static final int DIST_LOGNORMAL = 26; // Log-normal distribution
//
//    /* Multivariate real-valued */
//    public static final int DIST_NORMAL = 31; // Gaussian distribution
//    public static final int DIST_NORMALCHAIN = 32; // Gaussian dependent on the previous value
//    public static final int DIST_NORMALCL = 33; // Tree-structured Gaussian
//
//    /* Logistic */
//    public static final int DIST_LOGISTIC = 41; // Logistic distribution
//    public static final int DIST_TRANSLOGISTIC = 42; // Logistic distribution with transition terms
//    public static final int DIST_TRANSLOGISTICG = 43; // Same as TRANSLOGISTIC but initial sequence value is computed by
//    // averaging over all entries
//
//    /* Aliases */
//    public static final int DIST_ALIAS_CI = 51; // Conditionally independent distributions
//    public static final int DIST_ALIAS_MIXTURE = 52; // Mixture distribution
//    public static final int DIST_ALIAS_CLMIXTURE = 53; // Tree-dependent mixture distribution
//    public static final int DIST_MIXTURE_PRIOR = 54; // Mixture distribution with Dirichlet prior on mixing probabilities
//    public static final int DIST_ALIAS_MIXTURE_PRIOR = 55; // Mixture distribution with Dirichlet prior on the mixing probabilities
//    public static final int DIST_ALIAS_CLMIXTURE_PRIOR = 56; // Tree-dependent mixture distribution with a prior on edges and parameters
//    public static final int DIST_CHOWLIU_MDL = 61; // Distribution with a Chow-Liu forest with MDL prior
//    public static final int DIST_CHOWLIU_DIR_MDL = 62; // Chow-Liu forests with MDL prior on structures and Dirichlet prior on parameters
//    public static final int DIST_CONDCHOWLIU_MDL = 63; // Distribution with a conditional Chow-Liu forests with MDL prior
//    public static final int DIST_BERNOULLI_PRIOR = 71; // Multinomial distribution with Dirichlet prior
//    public static final int DIST_CONDBERNOULLI_PRIOR = 72; // Conditional univariate multinomial with Dirichlet prior
//    public static final int DIST_CONDBERNOULLIG_PRIOR = 73; // Same

    /* Cross-validation types */
    public static final int XVAL_NONE = 0;
    public static final int XVAL_LEAVENOUT = 1;

    /* Analysis types */
    public static final int ANALYSIS_MEAN = 0;
    public static final int ANALYSIS_CORR = 1;
    public static final int ANALYSIS_PERS = 2;
    public static final int ANALYSIS_DRY = 3;
    public static final int ANALYSIS_WET = 4;
    public static final int ANALYSIS_MI = 5;
    public static final int ANALYSIS_LO = 6;
    public static final int ANALYSIS_COMP = 100;

    /* Negative infinity for all practical purposes */
    public static final int NEG_INF = Integer.MIN_VALUE;
    public static final int POS_INF = Integer.MAX_VALUE;

    /* Sensitivity constant for EM */
    public static final double EM_EPSILON = 5E-05;
    public static final double EM_MIX_EPSILON = 5E-05;

    /* Newton-Raphson constants */
    public static final double NR_EPSILON = 1E-04;
    public static final double MIN_NR_EPSILON = 3E-03;
    public static final int MAX_NR_ITERATIONS = 50;

    public static final double INIT_EPSILON = 1E-08;

    /* Comparison epsilon */
    public static final double COMP_EPSILON = 1E-12;

    /* Minimal accepted MaxEnt distance */
    public static final double MAXENT_EPSILON = 1E-04;

    /* Minimal change in log-likelihood needed to add an edge */
    public static final double MIN_ME_LL_CHANGE = 1E-02;
    public static final double CONJ_GRAD_EPSILON = 1E-05;

    /* Making sure no probability falls out of (1e-12, 1-1e-12) range */
//#define MAX_ABS_PARAM_VALUE log(1E+12-1)

    /* Number of burn-off iterations */
    public static final int NUM_BURNOFF1_ITER = 5;
    public static final int NUM_BURNOFF2_ITER = 25;
    public static final int NUM_BURNOFF3_ITER = 60;

    /* Maximum number of best sequences for Viterbi */
    public static final int MAXBESTSEQ = 1000;

    /* Maximum number of allowed discrete values per variable (not enforced) */
    public static final int MAXNUMVALUES = 100;

    /* Comment symbol for the parameter (and data) files */
    public static final char COMMENT_SYMBOL = '#';

    /* Missing value constant for categorical data */
    public static final int CONST_MISSING_VALUE = -999;

    /* Maximum number of features for MaxEnt distributions */
    public static final int MAXNUMMAXENTFEATURES = 100;

    /* Action codes */
    public enum ActionCode {
        unknown("unknown"),
        learn("learning: parameter estimation"),
        viterbi("viterbi: best sequence calculation"),
        ll("log-likelihood calculation"),
        sim("simulation"),
        analyze("data analysis"),
        filling("hole filling evaluation"),
        sim_fill("simulation of data for missing data"),
        predict("prediction"),
        init("parameter initialization"),
        kl("Kullback-Leibler divergence"),
        lltrain("log-likelihood calculation of the training set"),
        debug("debugging");

        private final String description;

        ActionCode(String description) {
            this.description = description;
        }

        public String getDescription() {
            return description;
        }
    }

//    public static final int ACTION_CODE_UNKNOWN = 0;
//    public static final int ACTION_CODE_LEARN = 1;
//    public static final int ACTION_CODE_VITERBI = 2;
//    public static final int ACTION_CODE_LL = 3;
//    public static final int ACTION_CODE_SIM = 4;
//    public static final int ACTION_CODE_ANALYZE = 5;
//    public static final int ACTION_CODE_FILLING = 6;
//    public static final int ACTION_CODE_SIM_FILL = 7;
//    public static final int ACTION_CODE_PREDICT = 8;
//    public static final int ACTION_CODE_INIT = 9;
//    public static final int ACTION_CODE_KL = 10;
//    public static final int ACTION_CODE_LLTRAIN = 11;
//    public static final int ACTION_CODE_DEBUG = 1000;

    /* Types for the coefficients of logistic distribution */
    public static final int PARAM_TYPE_TRANS = 0;
    public static final int PARAM_TYPE_LINEAR = 1;

    /* Hole-poking evaluation type */
    public static final int POKING_TYPE_LOGP = 0;
    public static final int POKING_TYPE_PRED = 1;
    public static final int MISSING_IND_PROB = 2;
    public static final int MISSING_IND_PRED = 3;
    public static final int HIDDEN_STATE_PROB = 4;
    public static final int VITERBI_FILL_IN = 5;

    /* log(2) */
    public static final double CONST_LN2 = Math.log(2.0);

    /* Initialization type */
    public static final int INIT_RANDOM = 0;
    public static final int INIT_EM = 1;

    /* Minimum and maximum values for the constraints */
    public static final double CONSTRAINT_MIN_VALUE = 1E-012;
    public static final double CONSTRAINT_MAX_VALUE = 1 - 1E-012;

    /* Gibbs sampler parameters */
    public static final int GIBBS_BURNIN = 500;
    public static final int GIBBS_LAG = 30;
    public static final int GIBBS_SAMPLES_PER_STATE = 1000;

    /* Parameter for the exponent of the MDL prior */
    public static final double DEFAULT_MDL_BETA = 2E-00;

    /* Threshold for comparison of double-precision floats */
    public static final double COMP_THRESHOLD = 1E-012;

    private static long seed = System.nanoTime();

    public static void srand48(long s) {
        seed = s & 0xFFFFFFFFL;
        seed = (seed << 16) | 0x330E;
    }

    public static double drand48() {
        seed = (0x5DEECE66DL * seed + 0xBL) & ((1L << 48) - 1);
        return (double) seed / (1L << 48);
    }

    public static double xlogx(double x) {
        /* Function calculating x*ln(x) */

        /* x cannot be negative or NaN, and these values should trigger an error. */
        if (x < 0.0 || isNaN(x)) {
            //      System.err.format( "Warning: Unexpected value passed to x*log(x): x=%.12f\n", x );
        }

        if (x >= 0.0) {
            if (x < 1e-012)
                /* !!! lim x.0 x*log(x)=0 !!! */
                return (0.0);
            else
                return (x * log(x));
        }

        return (0.0);
    }
}




