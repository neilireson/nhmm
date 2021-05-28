package uk.ac.shef.wit.nhmm;

import java.io.File;
import java.io.IOException;
import java.io.PrintStream;

import static java.lang.Double.isNaN;
import static java.lang.Math.*;
import static uk.ac.shef.wit.nhmm.Constants.*;
import static uk.ac.shef.wit.nhmm.Data.is_missing;
import static uk.ac.shef.wit.nhmm.Data.missing_value;
import static uk.ac.shef.wit.nhmm.Matrix.find_det;
import static uk.ac.shef.wit.nhmm.Matrix.find_inv;
import static uk.ac.shef.wit.nhmm.Simulation.generateBernoulli;

public class Distribution {

    int type;                // Type of distribution
    int dim;                 // Dimension of input or the number of subcomponents
    int num_states;          // Number of states for discrete distributions
    int[] dim_index;          // Indices of components covered by the distribution
    int num_dists;           // For nested-divide distribution
    double[] state_prob;       // Array of probabilities for Bernoulli
    double[] log_state_prob;   // Array of log-probabilities for Bernoulli
    double[] mu;               // Means of the gaussian
    double[][] sigma;           // Covariance matrix or logistic transition matrix
    double[][] inv_sigma;       // Inverse of the covariance matrix
    double[][] chol_sigma;      // Upper-triangular Cholesky decomposition
    double det;               // Determinant of the covariance matrix
    Distribution[][] subdist;  // Array of sub-components
    double[] lambda;           // Probability of first state in logistic
    // double[][] sigma
    double[][] rho;             // Coefficients for linear terms in logistic
    // int is_singular;       // Flag whether the covariance matrix is singular
    int num_edges;           // The number of edges in the Chow-Liu forest
    double[][] ind_prob;        // Individual probabilities for each dimension
    int[][] edge;              // The list of edges (if any)
    double[][][] edge_prob;      // Corresponding edge probabilities
    double[] edge_MI;          // Corresponding mututal information (strength)
    double mdl_beta;          // Exponent modifier for the MDL prior
    double[][] W;               // Weight matrix for the decomposable tree prior
    Envelope[][] envelope;// Clusters of edges in the dependence tree
    int num_envelopes;       // Number of clusters of edges
    int[] node_used;          // The array indicating whether a node has been already used
    // double[][] sigma
    // double[][] inv_sigma
    double[] first_mu;         // Mean for the first observation in the sequence
    double[][] first_sigma;     // Covariance for the first observation in the sequence
    double[][] inv_first_sigma; // The inverse of the covariance of first observations
    double[][] chol_first_sigma;// Upper-triangular Cholesky decomposition
    double first_det;         // Determinant of the covariance matrix for the first state
    // double[][] W;               // Transformation matrix
    // double[] mu
    // double[][] sigma
    // double[][] inv_sigma
    // double det
    double[][] cond_state_prob; // Array of conditional probabilities for conditional (chain) Bernoulli
    double[][] log_cond_state_prob; // Array of conditional log-probabilities for conditional Bernoulli
    double[] mix_prob;         // Mixing probabilities for delta- distributions
    double[] exp_param;        // Parameter used with delta-exponential distribution
    double[] gamma_param1;     // Parameters used for delta-gamma distribution
    double[] gamma_param2;
    /* Bivariate MaxEnt */
    // double[][] sigma -- correlation parameters
    //   with the diagonal for the univariate constraints
    // double det -- normalization constant
    int is_sim_initialized;  // Indicator whether the burn-off is already performed
    int[] last_sim;           // Last simulated entry
    /* Tree structured Gaussian */
    // double[][] mu
    // double[][] sigma
    // double[][] inv_sigma
    // double det
    // double[][] chol_sigma
    // int num_edges
    // int[][] edge
    /* Dirac delta */
    double delta_value;
    /* Exponential (geometric) */
    double exp_param1;
    /* Gamma */
    double gamma1;
    double gamma2;
    /* Log-normal */
    double log_normal1;
    double log_normal2;

    // Univariate conditional exponential
    // double[] lambda -- scaling parameters for the regression in the exponential
    // double[] state_prob -- first sequence entry probabilities

    // Bayesian network decomposition bivariate MaxEnt
    // double[][] sigma -- parameters for the exponential model
    int[][] feature_index;     // Index for features for Bayesian networks MaxEnt
    int[][] feature_value;     // Values for feature functions for multinomial MaxEnt
    int[] num_features;       // Number of features for each of the factors
    int[] sim_order;          // Order in which the nodes need to be simulated

    /* Prior parameters */
    double[] pcount_single;    // Fictitious counts for a single variable (Dirichlet MAP)
    double[][] pcount_uni;      // Fictitious counts for univariate marginals (Dirichlet MAP)
    double[][][][] pcount_bi;     // Fictitious counts for bivariate marginals (Dirichlet MAP)
    double pcount;            // Total fictitious count

    /* Update variables */
    /* Conditional multinomial */
    double[][][][] joint_prob;    // Joint probability of hidden states given evidence (for univariate)
    double[][][][] uni_prob;      // Posterior univariate probabilities for all instances of hidden variables
    double[][][] log_fwd_update; // Log of scaled results of the forward pass
    double[][][] log_bkd_update; // Log of scaled results of the backward pass
    double[][] log_upd_scale;   // Log of inverse scaling factor
    double[][][][] log_p_tr;      // log P(S_nt=i|S_n,t-1=j,X_nt) -- log transition probability (for univariate)
    double[][][][] log_un_prob;   // log of unnormalized P(S_nt=i|S_n,t-1=j,X_nt) (for univariate)

    /* Powers of 2 from 0 to 30 */
    static int power2[]={
            1,2,4,8,
            16,32,64,128,
            256,512,1024,2048,
            4096,8192,16384,32768,
            65536,131072,262144,524288,
            1048576,2097152,4194304,8388608,
            16777216,33554432,67108864,134217728,
            268435456,536870912,1073741824
    };
    public int SINGULAR_FAIL;
    public int is_done;
    public int num_failed;
    public double maxent_epsilon;
    public double cg_epsilon;

    Distribution(int d_type, int d_num_states, int d_dim) {

        type = d_type;
        if (!(d_dim > 0)) {
            System.err.print("Dimensionality of any distribution must be at least 1.  Aborting.\n");
            System.exit(-1);
        }
        dim = d_dim;
        num_states = d_num_states;

        /* Initializing parameter structures */
        switch (type) {
            case DIST_FACTOR:
                subdist = new Distribution[1][][];
                subdist[0] = new Distribution[dim][];
                for (int i = 0; i < dim; i++)
                    subdist[0][i] = null;
                break;
            case DIST_BERNOULLI:
                dim_index = new int[dim];
                for (int i = 0; i < dim; i++)
                    dim_index[i] = i;

                subdist = null;

                state_prob = new double[num_states];
                log_state_prob = new double[num_states];

                /* Initializing */
                for (int i = 0; i < num_states; i++)
                    state_prob[i] = 0.0;
                for (int i = 0; i < num_states; i++)
                    log_state_prob[i] = NEG_INF;

                /* Allocating and initializing prior */
                pcount_single = new double[num_states];
                for (int i = 0; i < num_states; i++)
                    pcount_single[i] = 0.0;
                pcount = 0.0;

                /* Posterior univariate probabilities (for updates) */
                uni_prob = new double[1][][][];
                uni_prob[0] = new double[num_states][][];

                break;
            case DIST_CONDBERNOULLI:
            case DIST_CONDBERNOULLIG:
                /* First statesequence entry probabilities */
                dim_index = new int[dim];
                for (int i = 0; i < dim; i++)
                    dim_index[i] = i;

                subdist = null;

                state_prob = new double[num_states];
                log_state_prob = new double[num_states];

                /* Conditional probabilities */
                cond_state_prob = new double[num_states][];
                for (int i = 0; i < num_states; i++)
                    cond_state_prob[i] = new double[num_states];
                log_cond_state_prob = new double[num_states][];
                for (int i = 0; i < num_states; i++)
                    log_cond_state_prob[i] = new double[num_states];

                /* Initializing */
                for (int i = 0; i < num_states; i++)
                    state_prob[i] = 0.0;
                for (int i = 0; i < num_states; i++)
                    log_state_prob[i] = NEG_INF;

                for (int i = 0; i < num_states; i++)
                    for (int i1 = 0; i1 < num_states; i1++)
                        cond_state_prob[i][i1] = 0.0;
                for (int i = 0; i < num_states; i++)
                    for (int i1 = 0; i1 < num_states; i1++)
                        log_cond_state_prob[i][i1] = NEG_INF;

                /* Posterior univariate probabilities (for updates) */
                uni_prob = new double[1][][][];
                uni_prob[0] = new double[num_states][][];

                /* Joint probabilities (for updates) */
                joint_prob = new double[num_states][][][];
                for (int i = 0; i < num_states; i++)
                    joint_prob[i] = new double[num_states][][];

                /* Scaled forward pass updates */
                log_fwd_update = new double[num_states][][];

                /* Scaled backward pass updates */
                log_bkd_update = new double[num_states][][];

                /* Allocating and initializing the pseudo-counts for the parameter prior */
                pcount_uni = new double[num_states][];
                for (int i = 0; i < num_states; i++) {
                    pcount_uni[i] = new double[num_states];
                    for (int j = 0; j < num_states; j++)
                        pcount_uni[i][j] = 0.0;
                }

                pcount_single = new double[num_states];
                for (int i = 0; i < num_states; i++)
                    pcount_single[i] = 0.0;

                pcount = 0.0;

                break;
            case DIST_UNICONDMVME:
                /* Multinomial (polytomous) MaxEnt (logistic) distribution */

                /* num_states indicates the number of possible values */
                /* dim indicates the number of parameters */

                /* No need for dim_index -- the value in question is always indexed by 0 */

                subdist = null;

                /* Allocating the array of parameters */
                lambda = new double[MAXNUMMAXENTFEATURES];

                /* Initializing the number of features for each function */
                num_features = new int[MAXNUMMAXENTFEATURES];

                /* Initializing the indices for the functions */
                feature_index = new int[MAXNUMMAXENTFEATURES][];

                /* Initializing the values for the functions */
                feature_value = new int[MAXNUMMAXENTFEATURES][];

                break;
            case DIST_CHOWLIU:
                /* Allocating structures for tree-structured distribution */
                dim_index = new int[dim];
                for (int i = 0; i < dim; i++)
                    dim_index[i] = i;
                num_edges = 0;

                subdist = null;

                /* Allocating the array of marginal probabilities for each dimension */
                ind_prob = new double[dim][];
                for (int i = 0; i < dim; i++)
                    ind_prob[i] = new double[num_states];

                for (int i = 0; i < dim; i++)
                    for (int i1 = 0; i1 < num_states; i1++)
                        ind_prob[i][i1] = 0.0;

                /* Allocating the arrays of edges */
                /* There can be no more edges than the number of nodes-1 */
                edge = new int *[dim - 1];
                for (int i = 0; i < dim - 1; i++)
                    /* Each edge connects of a pair of nodes */
                    edge[i] = new int[2];

                edge_prob = new double[dim - 1][][];
                for (int i = 0; i < dim - 1; i++) {
	/* Probability table for each edge consists of a
	   num_states x num_states matrix */
                    edge_prob[i] = new double[num_states][];
                    for (int i1 = 0; i1 < num_states; i1++)
                        edge_prob[i][i1] = new double[num_states];
                }

                /* Allocating array of flags for the nodes */
                node_used = new int[dim];

                /* Allocating the arrays of edge strengths*/
                edge_MI = new double[dim - 1];

                for (int i = 0; i < dim - 1; i++)
                    edge_MI[i] = 0.0;

                /* Allocating and initializing weight matrix for the tree prior */
                W = new double[dim][];
                for (int i = 0; i < dim; i++)
                    W[i] = new double[dim];
                for (int i = 0; i < dim; i++)
                    for (int j = 0; j < dim; j++)
                        if (i == j)
                            W[i][j] = 0.0;
                        else
                            W[i][j] = 1.0;

                /* Allocating and initializing the fictitious counts for the parameter prior */
                pcount_uni = new double[dim][];
                for (int i = 0; i < dim; i++) {
                    pcount_uni[i] = new double[num_states];
                    for (int j = 0; j < num_states; j++)
                        pcount_uni[i][j] = 0.0;
                }

                pcount_bi = new double[dim][][][];
                for (int i = 0; i < dim; i++) {
                    pcount_bi[i] = new double[i][][];
                    for (int j = 0; j < i; j++) {
                        pcount_bi[i][j] = new double[num_states][];
                        for (int i1 = 0; i1 < num_states; i1++) {
                            pcount_bi[i][j][i1] = new double[num_states];
                            for (int i2 = 0; i2 < num_states; i2++)
                                pcount_bi[i][j][i1][i2] = 0.0;
                        }
                    }
                }

                pcount = 0.0;

                /* Initializing the envelopes */
                num_envelopes = 0;
                envelope = new Envelope[dim][];

                break;
            case DIST_CONDCHOWLIU:
                dim_index = new int[dim];
                for (int i = 0; i < dim; i++)
                    dim_index[i] = i;
                num_edges = 0;

                subdist = null;

                /* Allocating the array of marginal probabilities for each dimension (previous + current) */
                ind_prob = new double *[2 * dim];
                for (int i = 0; i < 2 * dim; i++)
                    ind_prob[i] = new double[num_states];

                /* Initializing the probabilities */
                for (int i = 0; i < 2 * dim; i++)
                    for (int i1 = 0; i1 < num_states; i1++)
                        ind_prob[i][i1] = 0.0;

                /* Allocating the arrays of edges */
                /* There can be no more edges than the number of nodes in the predictive space */
                edge = new int[dim][];
                for (int i = 0; i < dim; i++)
                    /* Each edge connects of a pair of nodes */
                    edge[i] = new int[2];

                edge_prob = new double[dim][][];
                for (int i = 0; i < dim; i++) {
	/* Probability table for each edge consists of a
	   num_states x num_states matrix */
                    edge_prob[i] = new double[num_states][];
                    for (int i1 = 0; i1 < num_states; i1++)
                        edge_prob[i][i1] = new double[num_states];
                }

                /* Allocating array of flags for the nodes */
                node_used = new int[2 * dim];

                /* Allocating the arrays of edge strengths */
                edge_MI = new double[dim];

                for (int i = 0; i < dim; i++)
                    edge_MI[i] = 0.0;

                if (BLAH) {
                    /* Allocating and initializing weight matrix for the tree prior */
                    W = new double[2 * dim][];
                    for (int i = 0; i < 2 * dim; i++)
                        W[i] = new double[2 * dim];
                    for (int i = 0; i < 2 * dim; i++)
                        for (int j = 0; j < 2 * dim; j++)
                            if (i == j)
                                W[i][j] = 0.0;
                            else
                                W[i][j] = 1.0;

                    /* Allocating and initializing the fictitious counts for the parameter prior */
                    pcount_uni = new double[2 * dim][];
                    for (int i = 0; i < 2 * dim; i++) {
                        pcount_uni[i] = new double[num_states];
                        for (int j = 0; j < num_states; j++)
                            pcount_uni[i][j] = 0.0;
                    }
                }
                break;
            case DIST_ME_BIVAR:
                dim_index = new int[dim];
                for (int i = 0; i < dim; i++)
                    dim_index[i] = i;

                subdist = null;

                sigma = new double[dim][];
                for (int i = 0; i < dim; i++)
                    sigma[i] = new double[i + 1];
                for (int i = 0; i < dim; i++)
                    for (int i1 = 0; i1 < i + 1; i1++)
                        sigma[i][i1] = 0.0;
                det = (double) dim * CONST_LN2;
                is_sim_initialized = 0;
                last_sim = null;
                break;
            case DIST_BN_ME_BIVAR:
                dim_index = new int[dim];
                for (int i = 0; i < dim; i++)
                    dim_index[i] = i;

                subdist = null;

                num_features = new int[dim];
                sigma = new double[dim][];
                for (int i = 0; i < dim; i++)
                    sigma[i] = new double[dim];
                feature_index = new int[dim][];
                for (int i = 0; i < dim; i++)
                    feature_index[i] = new int[dim];
                for (int i = 0; i < dim; i++) {
                    num_features[i] = 1;
                    feature_index[i][0] = i;
                    sigma[i][0] = 0.0;
                    for (int i1 = 1; i1 < dim; i1++) {
                        feature_index[i][i1] = 0;
                        sigma[i][i1] = 0.0;
                    }
                }
                sim_order = new int[dim];
                for (int i = 0; i < dim; i++)
                    sim_order[i] = i;

                break;
            case DIST_BN_CME_BIVAR:
                dim_index = new int[dim];
                for (int i = 0; i < dim; i++)
                    dim_index[i] = i;

                subdist = null;

                num_features = new int[dim];
                sigma = new double[dim][];
                for (int i = 0; i < dim; i++)
                    sigma[i] = new double[2 * dim];
                feature_index = new int[dim][];
                for (int i = 0; i < dim; i++)
                    feature_index[i] = new int[2 * dim];
                for (int i = 0; i < dim; i++) {
                    num_features[i] = 1;
                    feature_index[i][0] = i;
                    sigma[i][0] = 0.0;
                    for (int i1 = 1; i1 < 2 * dim; i1++) {
                        feature_index[i][i1] = 0;
                        sigma[i][i1] = 0.0;
                    }
                }
                sim_order = new int[dim];
                for (int i = 0; i < dim; i++)
                    sim_order[i] = i;

                /* First state probabilities */
                state_prob = new double[dim];
                for (int i = 0; i < dim; i++)
                    state_prob[i] = 0.0;

                break;
            case DIST_DELTAEXP:
                dim_index = new int[dim];
                for (int i = 0; i < dim; i++)
                    dim_index[i] = i;
                mix_prob = new double[num_states];
                for (int i = 0; i < num_states; i++)
                    mix_prob[i] = 0.0;

                exp_param = new double[num_states - 1];
                for (int i = 0; i < num_states - 1; i++)
                    exp_param[i] = 0.0;
                break;
            case DIST_DELTAGAMMA:
                dim_index = new int[dim];
                for (int i = 0; i < dim; i++)
                    dim_index[i] = i;
                mix_prob = new double[num_states];
                for (int i = 0; i < num_states; i++)
                    mix_prob[i] = 0.0;

                gamma_param1 = new double[num_states - 1];
                gamma_param2 = new double[num_states - 1];
                for (int i = 0; i < num_states - 1; i++) {
                    gamma_param1[i] = 0.0;
                    gamma_param2[i] = 0.0;
                }
                break;
            case DIST_DIRACDELTA:
                /* Dirac delta */
                dim_index = new int[1];
                dim_index[0] = 0;

                /* By default, the value is assumed to be 0 */
                delta_value = 0.0;
                break;
            case DIST_EXP:
                /* Geometric */
                dim_index = new int[1];
                dim_index[0] = 0;
                exp_param1 = 0.0;
                break;
            case DIST_GAMMA:
                /* Gamma distribution */
                dim_index = new int[1];
                dim_index[0] = 0;
                gamma1 = 0.0;
                gamma2 = 0.0;
                break;
            case DIST_LOGNORMAL:
                /* Log-normal distribution */
                dim_index = new int[1];
                dim_index[0] = 0;
                log_normal1 = 0.0;
                log_normal2 = 0.0;
                break;
            case DIST_NORMAL:
                dim_index = new int[dim];
                for (int i = 0; i < dim; i++)
                    dim_index[i] = i;
                mu = new double[dim];
                sigma = new double[dim][];
                for (int i = 0; i < dim; i++)
                    sigma[i] = new double[dim];
                inv_sigma = new double[dim][];
                for (int i = 0; i < dim; i++)
                    inv_sigma[i] = new double[dim];
                chol_sigma = null;

                /* Initializing */
                for (int i = 0; i < dim; i++)
                    mu[i] = 0.0;
                for (int i = 0; i < dim; i++)
                    for (int i1 = 0; i1 < dim; i1++)
                        sigma[i][i1] = 0.0;
                for (int i = 0; i < dim; i++)
                    for (int i1 = 0; i1 < dim; i1++)
                        inv_sigma[i][i1] = 0.0;
                det = 0.0;
                break;
            case DIST_NORMALCHAIN:
                dim_index = new int[dim];
                for (int i = 0; i < dim; i++)
                    dim_index[i] = i;
                first_mu = new double[dim];
                first_sigma = new double[dim][];
                for (int i = 0; i < dim; i++)
                    first_sigma[i] = new double[dim];
                inv_first_sigma = new double[dim][];
                for (int i = 0; i < dim; i++)
                    inv_first_sigma[i] = new double[dim];
                chol_first_sigma = null;

                mu = new double[dim];
                sigma = new double[dim][];
                for (int i = 0; i < dim; i++)
                    sigma[i] = new double[dim];
                inv_sigma = new double[dim][];
                for (int i = 0; i < dim; i++)
                    inv_sigma[i] = new double[dim];
                chol_sigma = null;
                W = new double[dim][];
                for (int i = 0; i < dim; i++)
                    W[i] = new double[dim];

                /* Initializing */
                for (int i = 0; i < dim; i++)
                    first_mu[i] = 0.0;
                for (int i = 0; i < dim; i++)
                    for (int i1 = 0; i1 < dim; i1++)
                        first_sigma[i][i1] = 0.0;
                for (int i = 0; i < dim; i++)
                    for (int i1 = 0; i1 < dim; i1++)
                        inv_first_sigma[i][i1] = 0.0;
                first_det = 0.0;

                for (int i = 0; i < dim; i++)
                    mu[i] = 0.0;
                for (int i = 0; i < dim; i++)
                    for (int i1 = 0; i1 < dim; i1++)
                        sigma[i][i1] = 0.0;
                for (int i = 0; i < dim; i++)
                    for (int i1 = 0; i1 < dim; i1++)
                        inv_sigma[i][i1] = 0.0;
                for (int i = 0; i < dim; i++)
                    for (int i1 = 0; i1 < dim; i1++)
                        W[i][i1] = 0.0;
                det = 0.0;
                break;
            case DIST_NORMALCL:
                dim_index = new int[dim];
                for (int i = 0; i < dim; i++)
                    dim_index[i] = i;
                mu = new double[dim];
                sigma = new double[dim][];
                for (int i = 0; i < dim; i++)
                    sigma[i] = new double[dim];
                inv_sigma = new double[dim][];
                for (int i = 0; i < dim; i++)
                    inv_sigma[i] = new double[dim];
                chol_sigma = null;

                /* Initializing */
                for (int i = 0; i < dim; i++)
                    mu[i] = 0.0;
                for (int i = 0; i < dim; i++)
                    for (int i1 = 0; i1 < dim; i1++)
                        sigma[i][i1] = 0.0;
                for (int i = 0; i < dim; i++)
                    for (int i1 = 0; i1 < dim; i1++)
                        inv_sigma[i][i1] = 0.0;
                det = 0.0;

                num_edges = 0;

                /* Allocating the arrays of edges */
                /* There can be no more edges than the number of nodes-1 */
                edge = new int[dim - 1][];
                for (int i = 0; i < dim - 1; i++)
                    /* Each edge connects of a pair of nodes */
                    edge[i] = new int[2];

                /* Allocating array of flags for the nodes */
                node_used = new int[dim];

                /* Allocating the array of edge strengths */
                edge_MI = new double[dim - 1];

                for (int i = 0; i < dim - 1; i++)
                    edge_MI[i] = 0.0;

                break;
            case DIST_LOGISTIC:
                subdist = null;

                lambda = new double[num_states];
                rho = new double[num_states][];
                for (int i = 0; i < num_states; i++)
                    rho[i] = new double[dim];

                /* Initializing */
                for (int i = 0; i < num_states; i++)
                    lambda[i] = 0.0;
                for (int i = 0; i < num_states; i++)
                    for (int i1 = 0; i1 < dim; i1++)
                        rho[i][i1] = 0.0;

                /* Posterior univariate probabilities (for updates) */
                uni_prob = new double ***[1];
                uni_prob[0] = new double[num_states][][];

                joint_prob = null;

                /* Arrays for storing probability values */
                log_p_tr = new double[num_states][][][];
                for (int i = 0; i < num_states; i++)
                    log_p_tr[i] = new double **[1];

                log_un_prob = new double[num_states][][][];
                for (int i = 0; i < num_states; i++)
                    log_un_prob[i] = new double **[1];

                break;
            case DIST_TRANSLOGISTIC:
            case DIST_TRANSLOGISTICG:
                subdist = null;

                lambda = new double[num_states];
                sigma = new double[num_states][];
                for (int i = 0; i < num_states; i++)
                    sigma[i] = new double[num_states];
                rho = new double[num_states][];
                for (int i = 0; i < num_states; i++)
                    rho[i] = new double[dim];

                /* Initializing */
                for (int i = 0; i < num_states; i++)
                    lambda[i] = 0.0;
                for (int i = 0; i < num_states; i++)
                    for (int i1 = 0; i1 < num_states; i1++)
                        sigma[i][i1] = 0.0;
                for (int i = 0; i < num_states; i++)
                    for (int i1 = 0; i1 < dim; i1++)
                        rho[i][i1] = 0.0;

                /* Posterior univariate probabilities (for updates) */
                uni_prob = new double ***[1];
                uni_prob[0] = new double[num_states][][];

                /* Joint probabilities (for updates) */
                joint_prob = new double[num_states][][][];
                for (int i = 0; i < num_states; i++)
                    joint_prob[i] = new double[num_states][][];

                /* Scaled forward pass updates */
                log_fwd_update = new double[num_states][][];

                /* Scaled backward pass updates */
                log_bkd_update = new double[num_states][][];

                /* Arrays for storing probability values */
                log_p_tr = new double[num_states][][][];
                for (int i = 0; i < num_states; i++)
                    log_p_tr[i] = new double[num_states][][];

                log_un_prob = new double[num_states][][][];
                for (int i = 0; i < num_states; i++)
                    log_un_prob[i] = new double[num_states][][];

                break;
            default:
                ;
        }
    }

    void close() {

        /* Deallocating parameter structures */
        switch (type) {
            case DIST_FACTOR:
                /* Removing sub-component distributions and deallocating the structure */
                for (int i = 0; i < dim; i++)
                    subdist[0][i] = null;
                subdist[0] = null;
                subdist = null;
                break;
            case DIST_BERNOULLI:
                dim_index = null;

                /* Removing mixture components */
                if (subdist != null) {
                    for (int i = 0; i < dim; i++) {
                        for (int j = 0; j < num_states; j++)
                            subdist[i][j] = null;
                        subdist[i] = null;
                    }
                    subdist = null;
                }

                state_prob = null;
                log_state_prob = null;

                /* Deallocating Dirichlet prior parameters */
                pcount_single = null;

                /* Deallocating the array of univariate posterior probabilities */
                uni_prob[0] = null;
                uni_prob = null;

                break;
            case DIST_CONDBERNOULLI:
            case DIST_CONDBERNOULLIG:
                /* Deallocating the array of first sequence entry probabilities */
                dim_index = null;

                /* Removing mixture components */
                if (subdist != null) {
                    for (int i = 0; i < dim; i++) {
                        for (int j = 0; j < num_states; j++)
                            subdist[i][j] = null;
                        subdist[i] = null;
                    }
                    subdist = null;
                }

                state_prob = null;
                log_state_prob = null;

                /* Deallocating the array of conditional probabilities */
                for (int i = 0; i < num_states; i++)
                    cond_state_prob[i] = null;
                cond_state_prob = null;
                for (int i = 0; i < num_states; i++)
                    log_cond_state_prob[i] = null;
                log_cond_state_prob = null;

                /* Deallocating the array of univariate posterior probabilities */
                uni_prob[0] = null;
                uni_prob = null;

                /* Deallocating the array of joint probabilities */
                for (int i = 0; i < num_states; i++)
                    joint_prob[i] = null;
                joint_prob = null;

                /* Deallocating scaled forward and backward pass updates */
                log_fwd_update = null;
                log_bkd_update = null;

                /* Deallocating Dirichlet prior parameters */
                for (int i = 0; i < num_states; i++)
                    pcount_uni[i] = null;
                pcount_uni = null;
                pcount_single = null;

                break;
            case DIST_UNICONDMVME:
                /* Removing mixture components */
                if (subdist != null) {
                    for (int i = 0; i < dim; i++) {
                        for (int j = 0; j < num_states; j++)
                            subdist[i][j] = null;
                        subdist[i] = null;
                    }
                    subdist = null;
                }

                /* Deallocating the array of parameters */
                lambda = null;

                /* Deallocating the indices for the functions */
                for (int i = 0; i < dim; i++)
                    feature_index[i] = null;
                feature_index = null;

                /* Deallocating the values for the functions */
                for (int i = 0; i < dim; i++)
                    feature_value[i] = null;
                feature_value = null;

                /* Deallocating the number of features for each function */
                num_features = null;

                break;
            case DIST_CHOWLIU:
                dim_index = null;

                /* Removing mixture components */
                if (subdist != null) {
                    for (int i = 0; i < dim; i++) {
                        for (int j = 0; j < num_states; j++)
                            subdist[i][j] = null;
                        subdist[i] = null;
                    }
                    subdist = null;
                }

                /* Deallocating the array of marginal probabilities */
                for (int i = 0; i < dim; i++)
                    ind_prob[i] = null;
                ind_prob = null;

                /* Deallocating the edges */
                for (int i = 0; i < dim - 1; i++)
                    edge[i] = null;
                edge = null;

                for (int i = 0; i < dim - 1; i++) {
                    for (int i1 = 0; i1 < num_states; i1++)
                        edge_prob[i][i1] = null;
                    edge_prob[i] = null;
                }
                edge_prob = null;

                /* Deallocating the array of flags */
                node_used = null;

                /* Deallocating the array of edge strengths */
                edge_MI = null;

                /* Deallocating weights for the tree prior distribution */
                for (int i = 0; i < dim; i++)
                    W[i] = null;
                W = null;

                /* Deallocating parameter prior pseudocounts */
                for (int i = 0; i < dim; i++)
                    pcount_uni[i] = null;
                pcount_uni = null;

                for (int i = 0; i < dim; i++) {
                    for (int j = 0; j < i; j++) {
                        for (int i1 = 0; i1 < num_states; i1++)
                            pcount_bi[i][j] = null[i1];
                        pcount_bi[i][j] = null;
                    }
                    pcount_bi[i] = null;
                }
                pcount_bi = null;

                /* Deallocating the envelopes */
                if (subdist != null)
                    for (int i = 0; i < num_envelopes; i++)
                        envelope[i] = null;
                envelope = null;

                break;
            case DIST_CONDCHOWLIU:
                dim_index = null;

                /* Removing mixture components */
                if (subdist != null) {
                    for (int i = 0; i < dim; i++) {
                        for (int j = 0; j < num_states; j++)
                            subdist[i][j] = null;
                        subdist[i] = null;
                    }
                    subdist = null;
                }

                /* Deallocating the array of marginal probabilities */
                for (int i = 0; i < 2 * dim; i++)
                    ind_prob[i] = null;
                ind_prob = null;

                /* Deallocating the edges */
                for (int i = 0; i < dim; i++)
                    edge[i] = null;
                edge = null;

                for (int i = 0; i < dim; i++) {
                    for (int i1 = 0; i1 < num_states; i1++)
                        edge_prob[i][i1] = null;
                    edge_prob[i] = null;
                }
                edge_prob = null;

                /* Deallocating the array of flags */
                node_used = null;

                /* Deallocating the arrays of edge strgenth */
                edge_MI = null;

                break;
            case DIST_ME_BIVAR:
                dim_index = null;

                /* Removing mixture components */
                if (subdist != null) {
                    for (int i = 0; i < dim; i++) {
                        for (int j = 0; j < num_states; j++)
                            subdist[i][j] = null;
                        subdist[i] = null;
                    }
                    subdist = null;
                }

                for (int i = 0; i < dim; i++)
                    sigma[i] = null;
                sigma = null;
                if (last_sim != null)
                    last_sim = null;
                break;
            case DIST_BN_ME_BIVAR:
                dim_index = null;
                num_features = null;
                for (int i = 0; i < dim; i++)
                    sigma[i] = null;
                sigma = null;
                for (int i = 0; i < dim; i++)
                    feature_index[i] = null;
                feature_index = null;
                sim_order = null;
                break;
            case DIST_BN_CME_BIVAR:
                dim_index = null;

                /* Removing mixture components */
                if (subdist != null) {
                    for (int i = 0; i < dim; i++) {
                        for (int j = 0; j < num_states; j++)
                            subdist[i][j] = null;
                        subdist[i] = null;
                    }
                    subdist = null;
                }

                num_features = null;
                for (int i = 0; i < dim; i++)
                    sigma[i] = null;
                sigma = null;
                for (int i = 0; i < dim; i++)
                    feature_index[i] = null;
                feature_index = null;
                sim_order = null;
                state_prob = null;
                break;
            case DIST_DELTAEXP:
                dim_index = null;
                mix_prob = null;
                exp_param = null;
                break;
            case DIST_DELTAGAMMA:
                dim_index = null;
                mix_prob = null;
                gamma_param1 = null;
                gamma_param2 = null;
                break;
            case DIST_DIRACDELTA:
            case DIST_EXP:
            case DIST_GAMMA:
            case DIST_LOGNORMAL:
                dim_index = null;
                break;
            case DIST_NORMAL:
                dim_index = null;
                mu = null;
                for (int i = 0; i < dim; i++)
                    sigma[i] = null;
                sigma = null;
                for (int i = 0; i < dim; i++)
                    inv_sigma[i] = null;
                inv_sigma = null;
                if (chol_sigma != null) {
                    for (int i = 0; i < dim; i++)
                        chol_sigma[i] = null;
                    chol_sigma = null;
                }
                break;
            case DIST_NORMALCHAIN:
                dim_index = null;
                first_mu = null;
                for (int i = 0; i < dim; i++)
                    first_sigma[i] = null;
                first_sigma = null;
                for (int i = 0; i < dim; i++)
                    inv_first_sigma[i] = null;
                inv_first_sigma = null;
                if (chol_first_sigma != null) {
                    for (int i = 0; i < dim; i++)
                        chol_first_sigma[i] = null;
                    chol_first_sigma = null;
                }

                mu = null;
                for (int i = 0; i < dim; i++)
                    sigma[i] = null;
                sigma = null;
                for (int i = 0; i < dim; i++)
                    inv_sigma[i] = null;
                inv_sigma = null;
                if (chol_sigma != null) {
                    for (int i = 0; i < dim; i++)
                        chol_sigma[i] = null;
                    chol_sigma = null;
                }

                for (int i = 0; i < dim; i++)
                    W[i] = null;
                W = null;
                break;
            case DIST_NORMALCL:
                dim_index = null;
                mu = null;
                for (int i = 0; i < dim; i++)
                    sigma[i] = null;
                sigma = null;
                for (int i = 0; i < dim; i++)
                    inv_sigma[i] = null;
                inv_sigma = null;
                if (chol_sigma != null) {
                    for (int i = 0; i < dim; i++)
                        chol_sigma[i] = null;
                    chol_sigma = null;
                }

                for (int i = 0; i < dim - 1; i++)
                    edge[i] = null;
                edge = null;

                node_used = null;

                edge_MI = null;

                break;
            case DIST_LOGISTIC:
                /* Removing mixture components */
                if (subdist != null) {
                    for (int j = 0; j < num_states; j++)
                        subdist[0][j] = null;
                    subdist[0] = null;
                    subdist = null;
                }

                lambda = null;
                for (int i = 0; i < num_states; i++)
                    rho[i] = null;
                rho = null;

                /* Deallocating the array of univariate posterior probabilities */
                uni_prob[0] = null;
                uni_prob = null;

                /* Deallocating arrays of probability values */
                for (int i = 0; i < num_states; i++)
                    log_p_tr[i] = null;
                log_p_tr = null;

                for (int i = 0; i < num_states; i++)
                    log_un_prob[i] = null;
                log_un_prob = null;

                break;
            case DIST_TRANSLOGISTIC:
            case DIST_TRANSLOGISTICG:
                /* Removing mixture components */
                if (subdist != null) {
                    for (int j = 0; j < num_states; j++)
                        subdist[0][j] = null;
                    subdist[0] = null;
                    subdist = null;
                }

                lambda = null;
                for (int i = 0; i < num_states; i++)
                    sigma[i] = null;
                sigma = null;
                for (int i = 0; i < num_states; i++)
                    rho[i] = null;
                rho = null;

                /* Deallocating the array of joint probabilities */
                for (int i = 0; i < num_states; i++)
                    joint_prob[i] = null;
                joint_prob = null;

                /* Deallocating scaled forward and backward pass updates */
                log_fwd_update = null;
                log_bkd_update = null;

                /* Deallocating the array of univariate posterior probabilities */
                uni_prob[0] = null;
                uni_prob = null;

                /* Deallocating arrays of probability values */
                for (int i = 0; i < num_states; i++)
                    log_p_tr[i] = null;
                log_p_tr = null;

                for (int i = 0; i < num_states; i++)
                    log_un_prob[i] = null;
                log_un_prob = null;

                break;
            default:
                ;
        }
    }

    Distribution copy() {
        /* Copy constructor */

        Distribution dist;

        dist = new Distribution(type, num_states, dim);

        switch (type) {
            case DIST_FACTOR:
                for (int i = 0; i < dim; i++)
                    dist.subdist[0][i] = subdist[0][i].copy();
                break;
            case DIST_BERNOULLI:
                for (int i = 0; i < dim; i++)
                    dist.dim_index[i] = dim_index[i];
                if (subdist != null) {
                    dist.subdist = new Distribution * [dim][];
                    for (int i = 0; i < dim; i++) {
                        dist.subdist[i] = new Distribution[num_states][];
                        for (int j = 0; j < num_states; j++)
                            dist.subdist[i][j] = subdist[i][j].copy();
                    }
                }
                for (int i = 0; i < num_states; i++)
                    dist.state_prob[i] = state_prob[i];
                for (int i = 0; i < num_states; i++)
                    dist.log_state_prob[i] = log_state_prob[i];
                for (int i = 0; i < num_states; i++)
                    dist.pcount_single[i] = pcount_single[i];
                dist.pcount = pcount;
                break;
            case DIST_CONDBERNOULLI:
            case DIST_CONDBERNOULLIG:
                for (int i = 0; i < dim; i++)
                    dist.dim_index[i] = dim_index[i];

                /* First sequence entry probabilities */
                for (int i = 0; i < num_states; i++)
                    dist.state_prob[i] = state_prob[i];
                for (int i = 0; i < num_states; i++)
                    dist.log_state_prob[i] = log_state_prob[i];

                /* Conditional probabilities */
                for (int i = 0; i < num_states; i++)
                    for (int j = 0; j < num_states; j++)
                        dist.cond_state_prob[i][j] = cond_state_prob[i][j];
                for (int i = 0; i < num_states; i++)
                    for (int j = 0; j < num_states; j++)
                        dist.log_cond_state_prob[i][j] = log(cond_state_prob[i][j]);

                /* Parameters of the prior */
                for (int i = 0; i < num_states; i++)
                    for (int j = 0; j < num_states; j++)
                        dist.pcount_uni[i][j] = pcount_uni[i][j];
                for (int i = 0; i < num_states; i++)
                    dist.pcount_single[i] = pcount_single[i];
                dist.pcount = pcount;
                break;
            case DIST_UNICONDMVME:
                if (subdist != null) {
                    dist.subdist = new Distribution * [dim][];
                    for (int i = 0; i < dim; i++) {
                        dist.subdist[i] = new Distribution[num_states][];
                        for (int j = 0; j < num_states; j++)
                            dist.subdist[i][j] = subdist[i][j].copy();
                    }
                }
                for (int i = 0; i < dim; i++)
                    dist.lambda[i] = lambda[i];

                for (int i = 0; i < dim; i++)
                    dist.num_features[i] = num_features[i];

                for (int i = 0; i < dim; i++) {
                    feature_index[i] = new int[num_features[i]];
                    feature_value[i] = new int[num_features[i]];
                    for (int i1 = 0; i1 < num_features[i]; i++) {
                        dist.feature_index[i][i1] = feature_index[i][i1];
                        dist.feature_value[i][i1] = feature_value[i][i1];
                    }
                }
                break;
            case DIST_CHOWLIU:
                for (int i = 0; i < dim; i++)
                    dist.dim_index[i] = dim_index[i];
                if (subdist != null) {
                    dist.subdist = new Distribution[dim][];
                    for (int i = 0; i < dim; i++) {
                        dist.subdist[i] = new Distribution[num_states];
                        for (int j = 0; j < num_states; j++)
                            dist.subdist[i][j] = subdist[i][j].copy();
                    }
                }
                /* Copying marginal probabilities */
                for (int i = 0; i < dim; i++)
                    for (int i1 = 0; i1 < num_states; i1++)
                        dist.ind_prob[i][i1] = ind_prob[i][i1];

                /* Copying edge information */
                dist.num_edges = num_edges;

                for (int i = 0; i < num_edges; i++) {
                    dist.edge[i][0] = edge[i][0];
                    dist.edge[i][1] = edge[i][1];
                    for (int i1 = 0; i1 < num_states; i1++)
                        for (int i2 = 0; i2 < num_states; i2++)
                            dist.edge_prob[i][i1][i2] = edge_prob[i][i1][i2];
                }

                for (int i = 0; i < num_edges; i++)
                    dist.edge_MI[i] = edge_MI[i];

                dist.mdl_beta = mdl_beta;

                /* Copying weight matrices for the tree prior */
                for (int i = 0; i < dim; i++)
                    for (int j = 0; j < dim; j++)
                        dist.W[i][j] = W[i][j];

                /* Copying pseudocounts for the parameter prior */
                for (int i = 0; i < dim; i++)
                    for (int j = 0; j < num_states; j++)
                        dist.pcount_uni[i][j] = pcount_uni[i][j];

                for (int i = 0; i < dim; i++)
                    for (int j = 0; j < i; j++)
                        for (int i1 = 0; i1 < num_states; i1++)
                            for (int i2 = 0; i2 < num_states; i2++)
                                dist.pcount_bi[i][j][i1][i2] = pcount_bi[i][j][i1][i2];

                dist.pcount = pcount;
                break;
            case DIST_CONDCHOWLIU:
                for (int i = 0; i < dim; i++)
                    dist.dim_index[i] = dim_index[i];
                if (subdist != null) {
                    dist.subdist = new Distribution[dim][];
                    for (int i = 0; i < dim; i++) {
                        dist.subdist[i] = new Distribution[num_states];
                        for (int j = 0; j < num_states; j++)
                            dist.subdist[i][j] = subdist[i][j].copy();
                    }
                }
                /* Copying marginal probabilities */
                for (int i = 0; i < dim; i++)
                    for (int i1 = 0; i1 < num_states; i1++)
                        dist.ind_prob[i][i1] = ind_prob[i][i1];

                /* Copying edge information */
                dist.num_edges = num_edges;

                for (int i = 0; i < num_edges; i++) {
                    dist.edge[i][0] = edge[i][0];
                    dist.edge[i][1] = edge[i][1];
                    for (int i1 = 0; i1 < num_states; i1++)
                        for (int i2 = 0; i2 < num_states; i2++)
                            dist.edge_prob[i][i1][i2] = edge_prob[i][i1][i2];
                }

                for (int i = 0; i < num_edges; i++)
                    dist.edge_MI[i] = edge_MI[i];

                dist.mdl_beta = mdl_beta;
                break;
            case DIST_ME_BIVAR:
                for (int i = 0; i < dim; i++)
                    dist.dim_index[i] = dim_index[i];
                if (subdist != null) {
                    dist.subdist = new Distribution[dim][];
                    for (int i = 0; i < dim; i++) {
                        dist.subdist[i] = new Distribution[num_states];
                        for (int j = 0; j < num_states; j++)
                            dist.subdist[i][j] = subdist[i][j].copy();
                    }
                }
                for (int i = 0; i < dim; i++)
                    for (int j = 0; j < i + 1; j++)
                        dist.sigma[i][j] = sigma[i][j];
                dist.det = det;
                break;
            case DIST_BN_ME_BIVAR:
                for (int i = 0; i < dim; i++)
                    dist.dim_index[i] = dim_index[i];
                if (subdist != null) {
                    dist.subdist = new Distribution[dim][];
                    for (int i = 0; i < dim; i++) {
                        dist.subdist[i] = new Distribution[num_states];
                        for (int j = 0; j < num_states; j++)
                            dist.subdist[i][j] = subdist[i][j].copy();
                    }
                }
                for (int i = 0; i < dim; i++)
                    dist.num_features[i] = num_features[i];
                for (int i = 0; i < dim; i++)
                    for (int j = 0; j < num_features[i]; j++) {
                        dist.sigma[i][j] = sigma[i][j];
                        dist.feature_index[i][j] = feature_index[i][j];
                    }
                for (int i = 0; i < dim; i++)
                    dist.sim_order[i] = sim_order[i];

                dist.mdl_beta = mdl_beta;
                break;
            case DIST_BN_CME_BIVAR:
                for (int i = 0; i < dim; i++)
                    dist.dim_index[i] = dim_index[i];
                if (subdist != null) {
                    dist.subdist = new Distribution[dim][];
                    for (int i = 0; i < dim; i++) {
                        dist.subdist[i] = new Distribution[num_states];
                        for (int j = 0; j < num_states; j++)
                            dist.subdist[i][j] = subdist[i][j].copy();
                    }
                }
                for (int i = 0; i < dim; i++)
                    dist.num_features[i] = num_features[i];
                for (int i = 0; i < dim; i++)
                    for (int j = 0; j < num_features[i]; j++) {
                        dist.sigma[i][j] = sigma[i][j];
                        dist.feature_index[i][j] = feature_index[i][j];
                    }
                for (int i = 0; i < dim; i++)
                    dist.sim_order[i] = sim_order[i];
                for (int i = 0; i < dim; i++)
                    dist.state_prob[i] = state_prob[i];

                dist.mdl_beta = mdl_beta;
                break;
            case DIST_DELTAEXP:
                for (int i = 0; i < dim; i++)
                    dist.dim_index[i] = dim_index[i];
                /* Copying mixing probabilities */
                for (int i = 0; i < num_states; i++)
                    dist.mix_prob[i] = mix_prob[i];
                for (int i = 0; i < num_states - 1; i++)
                    dist.exp_param[i] = exp_param[i];
                break;
            case DIST_DELTAGAMMA:
                for (int i = 0; i < dim; i++)
                    dist.dim_index[i] = dim_index[i];
                /* Copying mixing probabilities */
                for (int i = 0; i < num_states; i++)
                    dist.mix_prob[i] = mix_prob[i];
                for (int i = 0; i < num_states - 1; i++) {
                    dist.gamma_param1[i] = gamma_param1[i];
                    dist.gamma_param2[i] = gamma_param2[i];
                }
                break;
            case DIST_DIRACDELTA:
                dist.dim_index[0] = dim_index[0];
                break;
            case DIST_EXP:
                dist.dim_index[0] = dim_index[0];
                dist.exp_param1 = exp_param1;
                break;
            case DIST_GAMMA:
                dist.dim_index[0] = dim_index[0];
                dist.gamma1 = gamma1;
                dist.gamma2 = gamma2;
                break;
            case DIST_LOGNORMAL:
                dist.dim_index[0] = dim_index[0];
                dist.log_normal1 = log_normal1;
                dist.log_normal2 = log_normal2;
                break;
            case DIST_NORMAL:
                for (int i = 0; i < dim; i++)
                    dist.dim_index[i] = dim_index[i];
                for (int i = 0; i < dim; i++)
                    dist.mu[i] = mu[i];
                for (int i = 0; i < dim; i++)
                    for (int j = 0; j < dim; j++)
                        dist.sigma[i][j] = sigma[i][j];
                for (int i = 0; i < dim; i++)
                    for (int j = 0; j < dim; j++)
                        dist.inv_sigma[i][j] = inv_sigma[i][j];
                if (chol_sigma != null) {
                    dist.chol_sigma = new double[dim][];
                    for (int i = 0; i < dim; i++) {
                        dist.chol_sigma[i] = new double[dim];
                        for (int j = 0; j < dim; j++)
                            dist.chol_sigma[i][j] = chol_sigma[i][j];
                    }
                }
                dist.det = det;
                break;
            case DIST_NORMALCHAIN:
                for (int i = 0; i < dim; i++)
                    dist.dim_index[i] = dim_index[i];
                for (int i = 0; i < dim; i++)
                    dist.first_mu[i] = first_mu[i];
                for (int i = 0; i < dim; i++)
                    for (int j = 0; j < dim; j++)
                        dist.first_sigma[i][j] = first_sigma[i][j];
                for (int i = 0; i < dim; i++)
                    for (int j = 0; j < dim; j++)
                        dist.inv_first_sigma[i][j] = inv_first_sigma[i][j];
                if (chol_first_sigma != null) {
                    dist.chol_first_sigma = new double[dim][];
                    for (int i = 0; i < dim; i++) {
                        dist.chol_first_sigma[i] = new double[dim];
                        for (int j = 0; j < dim; j++)
                            dist.chol_first_sigma[i][j] = chol_first_sigma[i][j];
                    }
                }

                for (int i = 0; i < dim; i++)
                    dist.mu[i] = mu[i];
                for (int i = 0; i < dim; i++)
                    for (int j = 0; j < dim; j++)
                        dist.sigma[i][j] = sigma[i][j];
                for (int i = 0; i < dim; i++)
                    for (int j = 0; j < dim; j++)
                        dist.inv_sigma[i][j] = inv_sigma[i][j];
                if (chol_sigma != null) {
                    dist.chol_sigma = new double[dim][];
                    for (int i = 0; i < dim; i++) {
                        dist.chol_sigma[i] = new double[dim];
                        for (int j = 0; j < dim; j++)
                            dist.chol_sigma[i][j] = chol_sigma[i][j];
                    }
                }
                for (int i = 0; i < dim; i++)
                    for (int j = 0; j < dim; j++)
                        dist.W[i][j] = W[i][j];
                dist.det = det;
                break;
            case DIST_NORMALCL:
                for (int i = 0; i < dim; i++)
                    dist.dim_index[i] = dim_index[i];
                for (int i = 0; i < dim; i++)
                    dist.mu[i] = mu[i];
                for (int i = 0; i < dim; i++)
                    for (int j = 0; j < dim; j++)
                        dist.sigma[i][j] = sigma[i][j];
                for (int i = 0; i < dim; i++)
                    for (int j = 0; j < dim; j++)
                        dist.inv_sigma[i][j] = inv_sigma[i][j];
                if (chol_sigma != null) {
                    dist.chol_sigma = new double[dim][];
                    for (int i = 0; i < dim; i++) {
                        dist.chol_sigma[i] = new double[dim];
                        for (int j = 0; j < dim; j++)
                            dist.chol_sigma[i][j] = chol_sigma[i][j];
                    }
                }
                dist.det = det;

                /* Copying edge information */
                dist.num_edges = num_edges;

                for (int i = 0; i < num_edges; i++) {
                    dist.edge[i][0] = edge[i][0];
                    dist.edge[i][1] = edge[i][1];
                }

                for (int i = 0; i < num_edges; i++)
                    dist.edge_MI[i] = edge_MI[i];

                break;
            case DIST_LOGISTIC:
                if (subdist != null) {
                    dist.subdist = new Distribution[1][];
                    dist.subdist[0] = new Distribution[num_states];
                    for (int i = 0; i < num_states; i++)
                        dist.subdist[0][i] = subdist[0][i].copy();
                }
                for (int i = 0; i < num_states; i++)
                    dist.lambda[i] = lambda[i];
                for (int i = 0; i < num_states; i++)
                    for (int i1 = 0; i1 < dim; i1++)
                        dist.rho[i][i1] = rho[i][i1];
                break;
            case DIST_TRANSLOGISTIC:
            case DIST_TRANSLOGISTICG:
                if (subdist != null) {
                    dist.subdist = new Distribution[1][];
                    dist.subdist[0] = new Distribution[num_states];
                    for (int i = 0; i < num_states; i++)
                        dist.subdist[0][i] = subdist[0][i].copy();
                }
                for (int i = 0; i < num_states; i++)
                    dist.lambda[i] = lambda[i];
                for (int i = 0; i < num_states; i++)
                    for (int j = 0; j < num_states; j++)
                        dist.sigma[i][j] = sigma[i][j];
                for (int i = 0; i < num_states; i++)
                    for (int i1 = 0; i1 < dim; i1++)
                        dist.rho[i][i1] = rho[i][i1];
                break;
            default:
        }

        return (dist);
    }

    void Initialize(Data data) {
        int num_entries;
        double m, sd;         // Mean and standard deviation
        double lower_bound;   // Lower and upper bounds for the logistic parameters

        double[] node_weight;  // Sum of the incident edge weights (CL-trees)
        boolean[] node_visited;   // Indicator of whether the node has already been visited (CL-trees)
        double[][] W_transition;// Transition matrix (CL-trees)
        int root;            // First node to be selected (CL-trees)

        /* Temporary variable */
        double sum;

        switch (type) {
            case DIST_FACTOR:
                /* Initializing all sub-components */
                for (int i = 0; i < dim; i++)
                    subdist[0][i].Initialize(data);
                break;
            case DIST_BERNOULLI:
                /* Initializing Bernoulli parameters */
                if (BLAH) {
                    sum = 0.0;
                    for (int i = 0; i < num_states; i++) {
                        state_prob[i] = Constants.drand48();
                        sum += state_prob[i];
                    }

                    /* Renormalizing */
                    for (int i = 0; i < num_states; i++)
                        state_prob[i] /= sum;
                    for (int i = 0; i < num_states; i++)
                        log_state_prob[i] = log(state_prob[i]);
                }
                /* Initializing from the prior */
                sum = 0.0;
                /* First, simulating num_states gamma distributed variables */
                for (int i = 0; i < num_states; i++) {
                    state_prob[i] = generateGamma(pcount_single[i] + 1.0, 1.0);
                    sum += state_prob[i];
                }
                /* Normalizing */
                for (int i = 0; i < num_states; i++)
                    state_prob[i] /= sum;
                for (int i = 0; i < num_states; i++)
                    log_state_prob[i] = log(state_prob[i]);

                if (subdist != null)
                    /* Initializing mixture components */
                    for (int i = 0; i < dim; i++)
                        for (int j = 0; j < num_states; j++)
                            subdist[i][j].Initialize(data);
                break;
            case DIST_CONDBERNOULLI:
            case DIST_CONDBERNOULLIG:
                /* Initializing conditional probabilities */
                if (BLAH) {
                    for (int i = 0; i < num_states; i++) {
                        sum = 0.0;
                        for (int j = 0; j < num_states; j++) {
                            cond_state_prob[i][j] = Constants.drand48();
                            sum += cond_state_prob[i][j];
                        }
                        /* Renormalizing */
                        for (int j = 0; j < num_states; j++)
                            cond_state_prob[i][j] /= sum;
                        for (int j = 0; j < num_states; j++)
                            log_cond_state_prob[i][j] = log(cond_state_prob[i][j]);
                    }
                    /* Initializing first sequence entry probabilities */
                    sum = 0.0;
                    for (int i = 0; i < num_states; i++) {
                        state_prob[i] = Constants.drand48();
                        sum += state_prob[i];
                    }
                    /* Renormalizing */
                    for (int i = 0; i < num_states; i++)
                        state_prob[i] /= sum;
                    for (int i = 0; i < num_states; i++)
                        log_state_prob[i] = log(state_prob[i]);
                }

                /* Initializing from the prior */

                /* First sequence entry */
                sum = 0.0;
                /* Simulating gamma distributed variables first */
                for (int i = 0; i < num_states; i++) {
                    state_prob[i] = generateGamma(pcount_single[i] + num_states, 1.0);
                    sum += state_prob[i];
                }
                /* Normalizing */
                for (int i = 0; i < num_states; i++)
                    state_prob[i] /= sum;
                for (int i = 0; i < num_states; i++)
                    log_state_prob[i] = log(state_prob[i]);

                /* Conditional probabilities */
                for (int i = 0; i < num_states; i++) {
                    sum = 0.0;
                    for (int j = 0; j < num_states; j++) {
                        cond_state_prob[i][j] = generateGamma(pcount_uni[i][j] + 1.0, 1.0);
                        sum += cond_state_prob[i][j];
                    }
                    for (int j = 0; j < num_states; j++)
                        cond_state_prob[i][j] /= sum;
                    for (int j = 0; j < num_states; j++)
                        log_cond_state_prob[i][j] = log(cond_state_prob[i][j]);
                }

                if (subdist != null)
                    /* Initializing mixture components */
                    for (int i = 0; i < dim; i++)
                        for (int j = 0; j < num_states; j++)
                            subdist[i][j].Initialize(data);
                break;
            case DIST_UNICONDMVME:
                /* !!! Weights are initialized arbitrarily !!! */
                for (int i = 0; i < dim; i++)
                    lambda[i] = Constants.drand48();
                break;
            case DIST_CHOWLIU:
                /* Initializing with conditional independence assumption */

                /* Generating a random tree */
                /* Allocating necessary structures */
                node_weight = new double[dim];
                node_visited = new boolean[dim];
                W_transition = new double[dim][];
                for (int i = 0; i < dim; i++)
                    W_transition[i] = new double[dim];

                /* Initializing */
                for (int i = 0; i < dim; i++)
                    node_weight[i] = 0.0;
                for (int i = 0; i < dim; i++)
                    for (int j = 0; j < dim; j++)
                        node_weight[i] += W[i][j];

                /* W_transition[j][i] = P(S_t=i|S_{t-1}=j) */
                for (int j = 0; j < dim; j++)
                    for (int i = 0; i < dim; i++)
                        W_transition[j][i] = W[j][i] / node_weight[j];

                sum = 0.0;
                for (int i = 0; i < dim; i++)
                    sum += node_weight[i];
                for (int i = 0; i < dim; i++)
                    node_weight[i] /= sum;

                for (int i = 0; i < dim; i++)
                    node_visited[i] = false;

                num_edges = 0;

                /* Tree generation via a random walk on the weight graph (Andrei Broder, 1989) */
                /* Selecting a random starting node according to the stationary distribution (node_weight) */
                root = generateBernoulli(node_weight, dim);
                int index = root;
                node_visited[index] = true;
                while (num_edges < dim - 1) { /* Some nodes have not been visited yet */
                    int j = index;
                    index = generateBernoulli(W_transition[j], dim);

                    /* Checking whether the new node has been visited already */
                    if (!node_visited[index]) { /* Node has not been visited yet -- adding an edge */
                        node_visited[index] = true;
                        if (index > j) {
                            edge[num_edges][0] = index;
                            edge[num_edges][1] = j;
                        } else {
                            edge[num_edges][0] = j;
                            edge[num_edges][1] = index;
                        }
                        num_edges++;
                    } /* Node has not been visited yet -- adding an edge */
                } /* Some nodes have not been visited yet */

                /* Generating probability values */
                /* Initializing indicator for nodes first */
                for (int i = 0; i < dim; i++)
                    node_visited[i] = false;

                node_visited[root] = true;
                /* Sampling marginal for the root */
                sum = 0.0;
                for (int i = 0; i < num_states; i++) {
                    ind_prob[root][i] = generateGamma(pcount_uni[root][i] + num_states, 1.0);
                    sum += ind_prob[root][i];
                }
                for (int i = 0; i < num_states; i++)
                    ind_prob[root][i] /= sum;

                for (int e = 0; e < num_edges; e++) { /* Edge e */
                    if (node_visited[edge[e][0]]) { /* Second node not initialized */
                        for (int i = 0; i < num_states; i++) {
                            sum = 0.0;
                            for (int j = 0; j < num_states; j++) {
                                edge_prob[e][i][j] = generateGamma(pcount_bi[edge[e][0]][edge[e][1]][i][j] + 1.0, 1.0);
                                sum += edge_prob[e][i][j];
                            }
                            for (int j = 0; j < num_states; j++)
                                edge_prob[e][i][j] = edge_prob[e][i][j] * ind_prob[edge[e][0]][i] / sum;
                        }
                        /* Computing marginal for the second node */
                        for (int j = 0; j < num_states; j++) {
                            ind_prob[edge[e][1]][j] = 0.0;
                            for (int i = 0; i < num_states; i++)
                                ind_prob[edge[e][1]][j] += edge_prob[e][i][j];
                        }
                        /* Marking second node as visited */
                        node_visited[edge[e][1]] = true;
                    } /* Second node not initialized */ else { /* First node not initialized */
                        for (int j = 0; j < num_states; j++) {
                            sum = 0.0;
                            for (int i = 0; i < num_states; i++) {
                                edge_prob[e][i][j] = generateGamma(pcount_bi[edge[e][0]][edge[e][1]][i][j] + 1.0, 1.0);
                                sum += edge_prob[e][i][j];
                            }
                            for (int i = 0; i < num_states; i++)
                                edge_prob[e][i][j] = edge_prob[e][i][j] * ind_prob[edge[e][1]][j] / sum;
                        }
                        /* Computing marginals for the first node */
                        for (int i = 0; i < num_states; i++) {
                            ind_prob[edge[e][0]][i] = 0.0;
                            for (int j = 0; j < num_states; j++)
                                ind_prob[edge[e][0]][i] += edge_prob[e][i][j];
                        }
                        /* Marking first node as visited */
                        node_visited[edge[e][0]] = true;
                    } /* First node not initialized */
                } /* Edge e */

                /* Deallocating tree-generation structures */
                for (int i = 0; i < dim; i++)
                    W_transition[i] = null;
                W_transition = null;
                node_visited = null;
                node_weight = null;
                if (BLAH) {
    /* Assuming there are no edges -- need to initialize only
       marginal probabilities */
                    num_edges = 0;

                    for (int i = 0; i < dim; i++) {
                        sum = 0.0;
                        for (int i1 = 0; i1 < num_states; i1++) {
                            ind_prob[i][i1] = Constants.drand48();
                            sum += ind_prob[i][i1];
                        }

                        for (int i1 = 0; i1 < num_states; i1++)
                            ind_prob[i][i1] /= sum;
                    }
                }

                if (subdist != null)
                    /* Initializing mixture components */
                    for (int i = 0; i < dim; i++)
                        for (int j = 0; j < num_states; j++)
                            subdist[i][j].Initialize(data);

                if (subdist != null) { /* Mixture */
                    /* Computing the envelopes */
                    for (int i = 0; i < num_envelopes; i++)
                        envelope[i] = null;

                    num_envelopes = compute_envelopes_full();
                } /* Mixture */

                break;
            case DIST_CONDCHOWLIU:
                /* !!! Initializing with conditional independence assumption !!! */

    /* Assuming there are no edges -- need to initialize only
       marginal probabilities of the unobserved variables */
                num_edges = 0;

                for (int i = 0; i < dim; i++) {
                    sum = 0.0;
                    for (int i1 = 0; i1 < num_states; i1++) {
                        ind_prob[i][i1] = Constants.drand48();
                        sum += ind_prob[i][i1];
                    }

                    for (int i1 = 0; i1 < num_states; i1++)
                        ind_prob[i][i1] /= sum;
                }

                if (subdist != null)
                    /* Initializing mixture components */
                    for (int i = 0; i < dim; i++)
                        for (int j = 0; j < num_states; j++)
                            subdist[i][j].Initialize(data);
                break;
            case DIST_ME_BIVAR:
                /* Exponent of the normalization constant */
                det = 0.0;

                /* Initializing only the diagonal */
                for (int i = 0; i < dim; i++) {
                    sigma[i][i] = generateUnitNormal();
                    det += log(exp(sigma[i][i]) + 1.0);
                }

                for (int i = 0; i < dim; i++)
                    for (int j = 0; j < i; j++)
                        sigma[i][j] = 0.0;

                if (subdist != null)
                    /* Initializing mixture components */
                    for (int i = 0; i < dim; i++)
                        for (int j = 0; j < num_states; j++)
                            subdist[i][j].Initialize(data);
                break;
            case DIST_BN_ME_BIVAR:
                /* Assuming conditional independence */
                /* Initializing the coefficients as normally distributed around zero */
                for (int i = 0; i < dim; i++) {
                    num_features[i] = 1;
                    feature_index[i][0] = i;
                }

                for (int i = 0; i < dim; i++)
                    sigma[i][0] = generateUnitNormal();

                if (subdist != null)
                    /* Initializing mixture components */
                    for (int i = 0; i < dim; i++)
                        for (int j = 0; j < num_states; j++)
                            subdist[i][j].Initialize(data);
                break;
            case DIST_BN_CME_BIVAR:
                /* Assuming conditional independence */
                /* Initializing the coefficients as normally distributed around zero */
                for (int i = 0; i < dim; i++) {
                    num_features[i] = 1;
                    feature_index[i][0] = i;
                }

                for (int i = 0; i < dim; i++)
                    sigma[i][0] = generateUnitNormal();

                /* First entry probabilities */
                for (int i = 0; i < dim; i++)
                    state_prob[i] = Constants.drand48();

                if (subdist != null)
                    /* Initializing mixture components */
                    for (int i = 0; i < dim; i++)
                        for (int j = 0; j < num_states; j++)
                            subdist[i][j].Initialize(data);
                break;
            case DIST_DELTAEXP:
                /* f(x)=l*exp(-l*x) */
                /* Initializing mixing probabilities */
                if (BLAH) {
                    /* Old initialization */
                    sum = 0.0;
                    for (int i = 0; i < num_states; i++) {
                        mix_prob[i] = Constants.drand48();
                        sum += mix_prob[i];
                    }
                    for (int i = 0; i < num_states; i++)
                        mix_prob[i] /= sum;
                }
                /* Assuming uniform prior on the mixing probabilities */
                sum = 0.0;
                /* Simulating mixing probabilities from Dirichlet prior */
                for (int i = 0; i < num_states; i++) {
                    mix_prob[i] = generateGamma(1.0, 1.0);
                    sum += mix_prob[i];
                }
                /* Normalizing */
                for (int i = 0; i < num_states; i++)
                    mix_prob[i] /= sum;

                /* !!! Exponential distribution parameters -- HACK !!! */
                /* !!! Choosing the parameter between 0.01 and 1 !!! */
                for (int i = 0; i < num_states - 1; i++)
                    exp_param[i] = 1.0 / (1.0 + 99.0 * Constants.drand48());

                break;
            case DIST_DELTAGAMMA:
                /* f(x)=x^(param1-1)*exp(-x*param2)*param2^param1/Gamma(param1) */
                if (BLAH) {
                    /* Old initialization of the mixing probabilities */
                    /* Initializing mixing probabilities */
                    sum = 0.0;
                    for (int i = 0; i < num_states; i++) {
                        mix_prob[i] = Constants.drand48();
                        sum += mix_prob[i];
                    }
                    for (int i = 0; i < num_states; i++)
                        mix_prob[i] /= sum;
                }
                /* Assuming uniform prior on the mixing probabilities */
                sum = 0.0;
                /* Simulating mixing probabilities from Dirichlet prior */
                for (int i = 0; i < num_states; i++) {
                    mix_prob[i] = generateGamma(1.0, 1.0);
                    sum += mix_prob[i];
                }
                /* Normalizing */
                for (int i = 0; i < num_states; i++)
                    mix_prob[i] /= sum;

                /* !!! HACK !!! */
                /* Choosing first parameter between 0.1 and 1 and second between 0.01 and 1 */
                for (int i = 0; i < num_states - 1; i++) {
                    gamma_param1[i] = 1.0 / (1.0 + 9.0 * Constants.drand48());
                    gamma_param2[i] = 1.0 / (1.0 + 99.0 * Constants.drand48());
                }
                break;
            case DIST_DIRACDELTA:
                break;
            case DIST_EXP:
                /* f(x)=l*exp(-l*x) */
                /* !!! Choosing the parameter between 0.01 and 1 !!! */
                exp_param1 = 1.0 / (1.0 + 99.0 * Constants.drand48());
                break;
            case DIST_GAMMA:
                /* !!! Choosing first parameter between 0.1 and 1 and second between 0.01 and 1 !!! */
                gamma1 = 1.0 / (1.0 + 9.0 * Constants.drand48());
                gamma2 = 1.0 / (1.0 + 99.0 * Constants.drand48());
                break;
            case DIST_LOGNORMAL:
                /* !!! Choosing parameters between 0.1 and 1 !!! */
                log_normal1 = 1.0 / (1.0 + 9.0 * Constants.drand48());
                log_normal2 = 1.0 / (1.0 + 9.0 * Constants.drand48());
                break;
            case DIST_NORMAL:
                /* Tricky */

                /* Taking the covariance of the data as the covariance matrix */
                if (data != null) {
                    /* !!! Better data retrieval needed !!! */

                    num_entries = 0;

                    /* Displaying data */

                    /* Zeroing out the mean */
                    for (int i = 0; i < dim; i++)
                        mu[i] = 0.0;

                    /* Zeroing out the covariance matrix */
                    for (int i = 0; i < dim; i++)
                        for (int j = 0; j < dim; j++)
                            sigma[i][j] = 0.0;

                    /* Looping over all input entries while calculating the mean */
                    for (int n = 0; n < data.num_seqs; n++) {
                        for (int t = 0; t < data.sequence[n].seq_length;
                             t++)
                            for (int i = 0; i < dim; i++)
                                mu[i] += data.sequence[n].entry[t].rdata[dim_index[i]];

                        num_entries += data.sequence[n].seq_length;
                    }

                    for (int i = 0; i < dim; i++)
                        mu[i] /= (double) num_entries;

                    /* Calculating covariance */
                    for (int i = 0; i < dim; i++)
                        for (int j = 0; j <= i; j++)
                            for (int n = 0; n < data.num_seqs; n++)
                                for (int t = 0; t < data.sequence[n].seq_length;
                                     t++) {
                                    sigma[i][j] += ((data.sequence[n].entry[t].rdata[dim_index[i]] - mu[i]) *
                                            (data.sequence[n].entry[t].rdata[dim_index[j]] - mu[j]));
                                    sigma[j][i] = sigma[i][j];
                                }

                    /* Unbiased estimate of the covariance */
                    for (int i = 0; i < dim; i++)
                        for (int j = 0; j < dim; j++)
                            sigma[i][j] /= (double) (num_entries - 1);
                    if (BLAH) {
                        /* !!! Using unit matrix instead !!! */
                        for (int i = 0; i < dim; i++) {
                            for (int j = 0; j < i; j++) {
                                sigma[i][j] = 0.0;
                                sigma[j][i] = 0.0;
                            }
                            sigma[i][i] = 1.0;
                        }
                    }
                } else { /* !!! Bad initialization !!! */
                    for (int i = 0; i < dim; i++)
                        for (int j = 0; j <= i; j++) {
                            sigma[i][j] = Constants.drand48() - 0.5;
                            sigma[j][i] = sigma[i][j];
                        }
                } /* !!! Bad initialization !!! */

                /* !!! Has to be changed !!! */
                for (int i = 0; i < dim; i++)
                    mu[i] = Constants.drand48() - 0.5;

                /* Finding inverse and determinant */
                det = find_det(sigma, dim);
                if (SINGULAR_FAIL != null)
                    /* Data is highly correlated */ {
                    System.err.print("Data is highly correlated -- initialization failed.\n");
                    System.exit(-3);
                }


                find_inv(sigma, dim, inv_sigma);
                break;
            case DIST_NORMALCHAIN:
    /* !!! For right now, covariance matrices are initialized to unit matrices, and means
       are initialized to zero vectors. !!! */

                /* First state */
                for (int i = 0; i < dim; i++)
                    /* !!! Uniformly distributed between -0.2 and 0.2 !!! */
                    first_mu[i] = 0.4 * Constants.drand48() - 0.2;
                for (int i = 0; i < dim; i++)
                    for (int j = 0; j < dim; j++)
                        if (i == j)
                            first_sigma[i][j] = 1E-00;
                        else
                            first_sigma[i][j] = 0.0;
                first_det = find_det(first_sigma, dim);
                find_inv(first_sigma, dim, inv_first_sigma);


                /* Transition */
                for (int i = 0; i < dim; i++)
                    /* !!! Uniformly distributed between -0.2 and 0.2 !!! */
                    if (BLAH) {
                        mu[i] = 0.4 * Constants.drand48() - 0.2;
                    }
                /* Not going to use it */
                mu[i] = 0.0;

                for (int i = 0; i < dim; i++)
                    for (int j = 0; j < dim; j++)
                        if (i == j)
                            sigma[i][j] = 1E-00;
                        else
                            sigma[i][j] = 0.0;
                det = find_det(sigma, dim);
                find_inv(sigma, dim, inv_sigma);

                for (int i = 0; i < dim; i++)
                    for (int j = 0; j < dim; j++)
                        if (i == j)
                            W[i][j] = 1.0;
                        else
                            W[i][j] = 0.0;

                break;
            case DIST_NORMALCL:
                /* Tricky */

                /* Taking the covariance of the data as the covariance matrix */
                if (data != null) {
                    /* !!! Better data retrieval needed !!! */
                    num_entries = 0;

                    /* Zeroing out the mean */
                    for (int i = 0; i < dim; i++)
                        mu[i] = 0.0;

                    /* Zeroing out the covariance matrix */
                    for (int i = 0; i < dim; i++)
                        for (int j = 0; j < dim; j++)
                            sigma[i][j] = 0.0;

                    /* Looping over all input entries while calculating the mean */
                    for (int n = 0; n < data.num_seqs; n++) {
                        for (int t = 0; t < data.sequence[n].seq_length;
                             t++)
                            for (int i = 0; i < dim; i++)
                                mu[i] += data.sequence[n].entry[t].rdata[dim_index[i]];

                        num_entries += data.sequence[n].seq_length;
                    }

                    for (int i = 0; i < dim; i++)
                        mu[i] /= (double) num_entries;

                    /* Calculating covariance */
                    for (int i = 0; i < dim; i++)
                        for (int j = 0; j <= i; j++)
                            for (int n = 0; n < data.num_seqs; n++)
                                for (int t = 0; t < data.sequence[n].seq_length;
                                     t++) {
                                    sigma[i][j] += ((data.sequence[n].entry[t].rdata[dim_index[i]] - mu[i]) *
                                            (data.sequence[n].entry[t].rdata[dim_index[j]] - mu[j]));
                                    sigma[j][i] = sigma[i][j];
                                }

                    /* Unbiased estimate of the covariance */
                    for (int i = 0; i < dim; i++)
                        for (int j = 0; j < dim; j++)
                            sigma[i][j] /= (double) (num_entries - 1);
                } else { /* !!! Bad initialization !!! */
                    for (int i = 0; i < dim; i++)
                        for (int j = 0; j <= i; j++) {
                            sigma[i][j] = Constants.drand48() - 0.5;
                            sigma[j][i] = sigma[i][j];
                        }
                } /* !!! Bad initialization !!! */

                /* !!! Has to be changed !!! */
                for (int i = 0; i < dim; i++)
                    mu[i] = Constants.drand48() - 0.5;

                /* Finding inverse and determinant */
                det = find_det(sigma, dim);
                if (SINGULAR_FAIL != null)
                    /* Data is highly correlated */ {
                    System.err.print("Data is highly correlated -- initialization failed.\n");
                    System.exit(-3);
                }

                find_inv(sigma, dim, inv_sigma);
                break;
            case DIST_LOGISTIC:
                /* Logistic distribution */

                /* Initializing */

    /* Since the parameters for each state are rescaled in the end,
       can fix the set of parameters for one of the state (state 0) */

                for (int i = 1; i < num_states; i++)
                    lambda[i] = 0.1 + 0.8 * Constants.drand48();
                for (int i1 = 0; i1 < dim; i1++) { /* Initializing parameters for the linear terms */

                    /* Calculating mean and standard deviation for the i1-th component */
                    m = 0.0;
                    num_entries = 0;
                    for (int n = 0; n < data.num_seqs; n++) {
                        num_entries += data.num_seqs;
                        for (int t = 0; t < data.sequence[n].seq_length;
                             t++)
                            m += data.sequence[n].entry[t].rdata[i1];
                    }

                    m /= (double) num_entries;

                    sd = 0.0;
                    for (int n = 0; n < data.num_seqs; n++)
                        for (int t = 0; t < data.sequence[n].seq_length;
                             t++)
                            sd += (m - data.sequence[n].entry[t].rdata[i1]) * (m - data.sequence[n].entry[t].rdata[i1]);
                    sd = sqrt(sd / (double) num_entries);

                    /* Calculating lower and upper bound for the parameter (mean +- standard deviation) */
                    lower_bound = m - sd;

                    for (int i = 1; i < num_states; i++) {
                        rho[i][i1] = lower_bound + 2 * sd * Constants.drand48();
                        if (abs(rho[i][i1]) < INIT_EPSILON)
                            i--;
                        else
                            rho[i][i1] = 1.0 / rho[i][i1];
                    }

                } /* Initializing parameters for the linear terms */

                if (subdist != null)
                    /* Initializing mixture components */
                    for (int i = 0; i < num_states; i++)
                        subdist[0][i].Initialize(data);
                break;
            case DIST_TRANSLOGISTIC:
            case DIST_TRANSLOGISTICG:
                /* Logistic distribution with transition terms */

                /* Initializing */

    /* Since the parameters for each state are rescaled in the end,
       can fix the set of parameters for one of the state (state 0) */

                for (int i = 1; i < num_states; i++)
                    lambda[i] = 0.1 + 0.8 * Constants.drand48();
                for (int i = 1; i < num_states; i++)
                    for (int i1 = 0; i1 < num_states; i1++)
                        sigma[i1][i] = 0.1 + 0.8 * Constants.drand48();
                for (int i1 = 0; i1 < dim; i1++) { /* Initializing parameters for the linear terms */

                    if (data != null) {
                        /* Calculating mean and standard deviation for the i1-th component */
                        m = 0.0;
                        num_entries = 0;
                        for (int n = 0; n < data.num_seqs; n++) {
                            num_entries += data.num_seqs;
                            for (int t = 0; t < data.sequence[n].seq_length;
                                 t++)
                                m += data.sequence[n].entry[t].rdata[i1];
                        }

                        m /= (double) num_entries;

                        sd = 0.0;
                        for (int n = 0; n < data.num_seqs; n++)
                            for (int t = 0; t < data.sequence[n].seq_length;
                                 t++)
                                sd += (m - data.sequence[n].entry[t].rdata[i1]) * (m - data.sequence[n].entry[t]->
                        rdata[i1]);

                        sd = sqrt(sd / (double) num_entries);

                        /* Calculating lower and upper bound for the parameter (mean +- standard deviation) */
                        lower_bound = m - sd;

                        for (int i = 1; i < num_states; i++) {
                            rho[i][i1] = lower_bound + 2 * sd * Constants.drand48();
                            if (abs(rho[i][i1]) < INIT_EPSILON)
                                i--;
                            else
                                rho[i][i1] = 1.0 / rho[i][i1];
                        }
                    } else
                        rho[i][i1] = generateUnitNormal();

                } /* Initializing parameters for the linear terms */

                if (subdist != null)
                    /* Initializing mixture components */
                    for (int i = 0; i < num_states; i++)
                        subdist[0][i].Initialize(data);
                break;
            default:
                ;
        }

        return;
    }

    double log_prob(DataPoint datum, DataPoint prev_datum) {
        /* Finding the probability of the datum according to the distribution */
        double log_p;
        double n_exp = 0.0;

        double[] xminusmu;
        /* Chow-Liu trees */
        double[][] mult;

        /* Temporary variable(s) */
        double sum;
        int i1, i2, ei, m1;
        double[] exp_sum;
        double temp;
        double[] value_contrib;
        double max_value;

        switch (type) {
            case DIST_FACTOR:
                log_p = 0.0;
                for (int i = 0; i < dim; i++)
                    log_p += subdist[0][i].log_prob(datum, prev_datum);
                break;
            case DIST_BERNOULLI:
                if (subdist != null) { /* Mixture */
                    /* !!! Assuming univariate Bernoulli !!! */
                    value_contrib = new double[num_states];
                    max_value = NEG_INF;
                    for (int i = 0; i < num_states; i++) {
                        value_contrib[i] = log_state_prob[i] + subdist[0][i].log_prob(datum, prev_datum);
                        if (value_contrib[i] > max_value)
                            max_value = value_contrib[i];
                    }

                    if (max_value == Double.NEGATIVE_INFINITY)
                        log_p = NEG_INF;
                    else {
                        sum = 0.0;
                        for (int i = 0; i < num_states; i++)
                            sum += exp(value_contrib[i] - max_value);
                        log_p = max_value + log(sum);
                    }

                    value_contrib = null;
                } /* Mixture */ else {
                    if (dim == 1)
                        if (is_missing(datum.ddata[dim_index[0]]))
                            log_p = 0.0;
                        else
                            log_p = log_state_prob[datum.ddata[dim_index[0]]];
                }
                break;
            case DIST_CONDBERNOULLI:
            case DIST_CONDBERNOULLIG:
                /* !!! Do not know how to handle missing data !!! */
                if (prev_datum != null) {
                    /* Not the first entry in the sequence */
                    if (!is_missing(prev_datum.ddata[dim_index[0]]) &&
                            !is_missing(datum.ddata[dim_index[0]]))
                        log_p = log_cond_state_prob[prev_datum.ddata[dim_index[0]]][datum.ddata[dim_index[0]]];
                } else
                    /* First entry in the sequence */
                    if (!is_missing(datum.ddata[dim_index[0]]))
                        log_p = log_state_prob[datum.ddata[dim_index[0]]];
                    else
                        log_p = 0.0;
                break;
            case DIST_UNICONDMVME:
                /* !!! Do not know how to handle missing data !!! */

    /* datum.ddata contains the value for which the probability is to be computed (index 0),
       and the covariates are in the other components */
                /* prev_datum is irrelevant */

                /* First feature index is always missing as it corresponds to the variable in question */
                /* Feature values for real-valued variables are also not specified */

                /* Initializing the exponent sums for state values */
                exp_sum = new double[num_states];
                for (int b = 0; b < num_states; b++)
                    exp_sum[b] = 0.0;

                /* Computing the normalization constant and the exponent for the state value in question */

                /* Also computing the exponential sum for all values of the variable */
                for (int i = 0; i < dim; i++) { /* Considering function i */
                    temp = lambda[i];
                    for (int j = 1; j < num_features[i]; j++)
                        if (feature_index[i][j] < datum.ddim) { /* Categorical feature */
                            if (feature_value[i][j] != datum.ddata[feature_index[i][j]]) {
                                temp = 0.0;
                                /* No need to look at other features for this function */
                                j = num_features[i];
                            }
                        } /* Categorical feature */ else
                            /* Real-valued feature */
                            temp *= datum.rdata[feature_index[i][j] - datum.ddim];

                    /* Updating the exp-sum for the appropriate value */
                    exp_sum[feature_value[i][0]] += temp;
                } /* Considering function i */

                /* Computing the normalization constant */
                /* !!! Needs to be optimized !!! */
                sum = 0.0;
                for (int b = 0; b < num_states; b++)
                    sum += exp(exp_sum[b]);

                log_p = exp_sum[datum.ddata[0]] - log(sum);

                exp_sum = null;

                break;
            case DIST_CHOWLIU:
                /* Allocating additional variables */
                mult = new double[dim][];
                for (int i = 0; i < dim; i++)
                    mult[i] = new double[num_states];

                log_p = 0.0;

                if (subdist == null) { /* No mixture */
                    /* Computing univariate contributions for instantiated variables */
                    for (int i = 0; i < dim; i++)
                        if (!is_missing(datum.ddata[dim_index[i]]))
                            log_p += log(ind_prob[i][datum.ddata[dim_index[i]]]);

                    /* Computing the envelopes */
                    num_envelopes = compute_envelopes(datum, mult);
                } /* No mixture */ else
                    /* Mixture */
                    /* Precomputing probability values for components */
                    for (int i = 0; i < dim; i++)
                        for (int j = 0; j < num_states; j++)
                            mult[i][j] = exp(subdist[i][j].log_prob(datum, prev_datum));


                for (int env = 0; env < num_envelopes; env++)
                    /* For each envelope */
                    if (!envelope[env].is_missing)
                        /* Envelope contains no missing values */
                        for (int e = 0; e < envelope[env].num_edges;
                             e++) {
                            ei = envelope[env].edge[e];
                            i1 = edge[ei][0];
                            i2 = edge[ei][1];
                            log_p += log(edge_prob[ei][datum.ddata[dim_index[i1]]][datum.ddata[dim_index[i2]]])
                                    - log(ind_prob[i1][datum.ddata[dim_index[i1]]]) - log(ind_prob[i2][datum.ddata[dim_index[i2]]]);
                        }
                    else if (envelope[env].num_nodes == 1) { /* Envelope contains exactly one missing variable. */
                        m1 = envelope[env].node[0];
                        sum = 0.0;
                        for (int b = 0; b < num_states; b++)
                            /* Summing over the missing variable */
                            sum += ind_prob[m1][b] * mult[m1][b];

                        /* Updating the probability */
                        log_p += log(sum);
                    } /* Envelope contains exactly one missing variable. */ else { /* Envelope contains more than one missing variable. */
                        /* Propagating probabilities in a backward pass */
                        for (int e = envelope[env].num_edges - 1;
                             e >= 0;
                             e--) {
                            ei = envelope[env].edge[e];
                            i1 = edge[ei][0];
                            i2 = edge[ei][1];

                            if (envelope[env].node[e + 1] == i2) { /* i1 is the parent of i2 */
                                /* Scaling P(i2->)/P(i2) */
                                sum = 0.0;
                                for (value2 = 0; value2 < num_states; value2++)
                                    sum += mult[i2][value2];
                                for (value2 = 0; value2 < num_states; value2++)
                                    mult[i2][value2] /= sum;
                                log_p += log(sum);

                                for (value1 = 0; value1 < num_states; value1++) { /* Computing the contribution to i1 with value value1 from child i2 */
                                    sum = 0.0;
                                    for (value2 = 0; value2 < num_states; value2++)
                                        sum += mult[i2][value2] * edge_prob[ei][value1][value2];
                                    mult[i1][value1] *= sum / ind_prob[i1][value1];
                                } /* Computing the contribution to i1 with value value1 from child i2 */
                            } /* i1 is the parent of i2 */ else { /* i2 is the parent of i1 */
                                /* Scaling P(i1->)/P(i1) */
                                sum = 0.0;
                                for (value1 = 0; value1 < num_states; value1++)
                                    sum += mult[i1][value1];
                                for (value1 = 0; value1 < num_states; value1++)
                                    mult[i1][value1] /= sum;
                                log_p += log(sum);

                                for (value2 = 0; value2 < num_states; value2++) { /* Computing the contribution to i2 with value value2 from child i1 */
                                    sum = 0.0;
                                    for (value1 = 0; value1 < num_states; value1++)
                                        sum += mult[i1][value1] * edge_prob[ei][value1][value2];
                                    mult[i2][value2] *= sum / ind_prob[i2][value2];
                                } /* Computing the contribution to i2 with value value2 from child i1 */
                            } /* i2 is the parent of i1 */
                        }

                        /* Summing over the root of the envelope */
                        m1 = envelope[env].node[0];
                        sum = 0.0;
                        for (int b = 0; b < num_states; b++)
                            sum += mult[m1][b] * ind_prob[m1][b];
                        /* Updating the probability */
                        log_p += log(sum);
                    } /* Envelope contains more than one missing variable. */

                /* Deallocating */
                for (int i = 0; i < dim; i++)
                    mult[i] = null;
                mult = null;

                if (subdist == null)
                    /* No mixture */
                    for (int env = 0; env < num_envelopes; env++)
                        envelope[env] = null;

                break;
            case DIST_CONDCHOWLIU:
                /* !!! Does not handle missing data !!! */

                /* Initializing the array of flags */
                for (int i = 0; i < dim; i++) {
                    /* Current observation nodes */
                    node_used[i] = 0;
                    /* Previous observation nodes */
                    node_used[i + dim] = 1;
                }

                log_p = 0.0;

                for (int i = 0; i < num_edges; i++) { /* For each edge */

                    /* value1 is the value for the first node; value2 is the value for the second node */
                    /* Assuming indices for the current vector are 0..dim-1 and indices for the previous vector are dim..2*dim-1 */
                    if (edge[i][0] < dim)
                        /* Current observation vector */
                        value1 = datum.ddata[dim_index[edge[i][0]]];
                    else
                        /* Previous observation vector */
                        if (prev_datum != null)
                            /* Current vector not the first in sequence */
                            value1 = prev_datum.ddata[dim_index[edge[i][0] - dim]];
                        else
                            /* Current vector the first in sequence */
                            value1 = missing_value((int) 0);

                    if (edge[i][1] < dim)
                        /* Current observation vector */
                        value2 = datum.ddata[dim_index[edge[i][1]]];
                    else
                        /* Previous observation vector */
                        if (prev_datum != null)
                            /* Current vector not the first in sequence */
                            value2 = prev_datum.ddata[dim_index[edge[i][1] - dim]];
                        else
                            /* Current vector the first in sequence */
                            value2 = missing_value((int) 0);

                    if (is_missing(value1)) {
                        if (!is_missing(value2) && !node_used[edge[i][1]])
                            log_p += log(ind_prob[edge[i][1]][value2]);
                    } else if (is_missing(value2)) {
                        if (!node_used[edge[i][0]])
                            log_p += log(ind_prob[edge[i][0]][value1]);
                    } else {
                        log_p += log(edge_prob[i][value1][value2]);

                        /* Checking whether need to divide by the probability of the node */
                        if (node_used[edge[i][0]])
                            log_p -= log(ind_prob[edge[i][0]][value1]);

                        if (node_used[edge[i][1]])
                            log_p -= log(ind_prob[edge[i][1]][value2]);
                    }

                    node_used[edge[i][0]] = 1;
                    node_used[edge[i][1]] = 1;
                } /* For each edge */

                /* Making sure that all nodes have been used */
                for (int i = 0; i < dim; i++)
                    if (!node_used[i] && !is_missing(datum.ddata[dim_index[i]]))
                        /* Updating the probability taking the unused node into the account */
                        log_p += log(ind_prob[i][datum.ddata[dim_index[i]]]);

                break;
            case DIST_ME_BIVAR:
                /* !!! Can't handle missing data yet !!! */
                sum = 0.0;
                for (int i = 0; i < dim; i++)
                    if (datum.ddata[dim_index[i]] == 1)
                        for (int j = 0; j < i + 1; j++)
                            if (datum.ddata[dim_index[j]] == 1)
                                sum += sigma[i][j];
                log_p = sum - det;
                break;
            case DIST_BN_ME_BIVAR:
                /* !!! Can't handle missing data !!! */
                log_p = 0.0;  // log-probability
                for (int i = 0; i < dim; i++) {
                    sum = sigma[i][0];
                    for (int j = 1; j < num_features[i]; j++)
                        if (datum.ddata[dim_index[feature_index[i][j]]] == 1)
                            sum += sigma[i][j];
                    log_p -= log(1.0 + exp(sum));  // Subtracting the normalization constant
                    if (datum.ddata[dim_index[i]] == 1)
                        log_p += sum;
                }
                break;
            case DIST_BN_CME_BIVAR:
                /* !!! Can't handle missing data !!! */
                if (prev_datum != null) {
                    log_p = 0.0;  // log-probability
                    for (int i = 0; i < dim; i++) {
                        sum = sigma[i][0];
                        for (int j = 1; j < num_features[i]; j++)
                            if (feature_index[i][j] < dim) {
                                if (datum.ddata[dim_index[feature_index[i][j]]] == 1)
                                    sum += sigma[i][j];
                            } else {
                                if (prev_datum.ddata[dim_index[feature_index[i][j] - dim]] == 1)
                                    sum += sigma[i][j];
                            }
                        log_p -= log(1.0 + exp(sum));  // Subtracting the normalization constant
                        if (datum.ddata[dim_index[i]] == 1)
                            log_p += sum;
                    }
                } else {
                    log_p = 0.0;
                    for (int i = 0; i < dim; i++)
                        if (datum.ddata[dim_index[i]] == 1)
                            log_p += log(state_prob[i]);
                        else
                            log_p += log(1.0 - state_prob[i]);
                }
                break;
            case DIST_DELTAEXP:
                /* f(x)=l*exp(-l*x) */
                if (is_missing(datum.rdata[dim_index[0]]))
                    log_p = 0.0;
                else {
                    if (abs(datum.rdata[dim_index[0]]) <= COMP_THRESHOLD)
                        /* First (delta) component */
                        log_p = log(mix_prob[0]);
                    else { /* Mixture of exponential components */
                        value_contrib = new double[num_states - 1];
                        max_value = NEG_INF;
                        for (int i = 0; i < num_states - 1; i++) {
                            value_contrib[i] = log(mix_prob[i + 1]) + log(exp_param[i]) - exp_param[i] * datum.rdata[dim_index[0]];
                            if (max_value < value_contrib[i])
                                max_value = value_contrib[i];
                        }

                        sum = 0.0;
                        for (int i = 0; i < num_states - 1; i++)
                            sum += exp(value_contrib[i] - max_value);

                        log_p = max_value + log(sum);

                        value_contrib = null;
                    } /* Mixture of exponential components */
                }
                break;
            case DIST_DELTAGAMMA:
                /* f(x)=x^(param1-1)*exp(-x*param2)*param2^param1/Gamma(param1) */
                if (is_missing(datum.rdata[dim_index[0]]))
                    log_p = 0.0;
                else {
                    if (abs(datum.rdata[dim_index[0]]) <= COMP_THRESHOLD)
                        /* First (delta) component */
                        log_p = log(mix_prob[0]);
                    else { /* Mix of gamma components */
                        value_contrib = new double[num_states - 1];
                        max_value = NEG_INF;
                        for (int i = 0; i < num_states - 1; i++) {
                            value_contrib[i] = log(mix_prob[i + 1]) + (gamma_param1[i] - 1.0) * log(datum.rdata[dim_index[0]]) - gamma_param2[i] * datum.rdata[dim_index[0]] + gamma_param1[i] * log(gamma_param2[i]) - gammaln(gamma_param1[i]);
                            if (max_value < value_contrib[i])
                                max_value = value_contrib[i];
                        }

                        sum = 0.0;
                        for (int i = 0; i < num_states - 1; i++)
                            sum += exp(value_contrib[i] - max_value);

                        log_p = max_value + log(sum);

                        value_contrib = null;
                    } /* Mix of gamma components */
                }
                break;
            case DIST_DIRACDELTA:
                if (is_missing(datum.rdata[dim_index[0]]))
                    log_p = 0.0;
                else if (abs(datum.rdata[dim_index[0]]) <= COMP_THRESHOLD)
                    log_p = 0.0;
                else
                    log_p = NEG_INF;
                break;
            case DIST_EXP:
                if (is_missing(datum.rdata[dim_index[0]]))
                    log_p = 0.0;
                else if (abs(datum.rdata[dim_index[0]]) <= COMP_THRESHOLD)
                    log_p = NEG_INF;
                else
                    log_p = log(exp_param1) - exp_param1 * datum.rdata[dim_index[0]];
                break;
            case DIST_GAMMA:
                if (is_missing(datum.rdata[dim_index[0]]))
                    log_p = 0.0;
                else if (abs(datum.rdata[dim_index[0]]) <= COMP_THRESHOLD)
                    log_p = NEG_INF;
                else
                    log_p = (gamma1 - 1.0) * log(datum.rdata[dim_index[0]]) - gamma2 * datum.rdata[dim_index[0]] + gamma1 * log(gamma2) - gammaln(gamma1);
                break;
            case DIST_LOGNORMAL:
                if (is_missing(datum.rdata[dim_index[0]]))
                    log_p = 0.0;
                else if (abs(datum.rdata[dim_index[0]]) <= COMP_THRESHOLD)
                    log_p = NEG_INF;
                else
                    log_p = -(log(datum.rdata[dim_index[0]]) - log_normal1) * (log(datum.rdata[dim_index[0]]) - log_normal1) / (2.0 * log_normal2) - log(datum.rdata[dim_index[0]]) - 0.5 * (CONST_LN2 + log(PI) + log(log_normal2));
                break;
            case DIST_NORMAL:
                xminusmu = new double[dim];

                /* Calculating x-mu */
                for (int i = 0; i < dim; i++)
                    xminusmu[i] = datum.rdata[dim_index[i]] - mu[i];

                /* Calculating (x-mu)'*sigma^(-1)*(x-mu) */
                for (int i = 0; i < dim; i++)
                    for (int j = 0; j < dim; j++)
                        n_exp += xminusmu[i] * inv_sigma[i][j] * xminusmu[j];

                xminusmu = null;

                log_p = -0.5 * n_exp - 0.5 * log(det) - 0.5 * (double) dim * (CONST_LN2 + log(PI));
                break;
            case DIST_NORMALCHAIN:
                /* !!! Needs to be rechecked !!! */
                xminusmu = new double[dim];

                if (prev_datum != null) {
                    /* Calculating W*x_{t-1}+mu-x_t */
                    for (int i = 0; i < dim; i++) {
                        xminusmu[i] = mu[i] - datum.rdata[dim_index[i]];
                        for (int j = 0; j < dim; j++)
                            xminusmu[i] += W[i][j] * prev_datum.rdata[dim_index[j]];
                    }

                    /* Calculating (W*x_{t-1}+mu-x_t)'*sigma^(-1)*(W*x_{t-1}+mu-x_t) */
                    for (int i = 0; i < dim; i++)
                        for (int j = 0; j < dim; j++)
                            n_exp += xminusmu[i] * inv_sigma[i][j] * xminusmu[j];

                    log_p = -0.5 * n_exp - 0.5 * log(det) - 0.5 * (double) dim * (CONST_LN2 + log(PI));
                } else {
                    /* Calculating x-mu */
                    for (int i = 0; i < dim; i++)
                        xminusmu[i] = datum.rdata[dim_index[i]] - first_mu[i];

                    /* Calculating (x-mu)'*sigma^(-1)*(x-mu) */
                    for (int i = 0; i < dim; i++)
                        for (int j = 0; j < dim; j++)
                            n_exp += xminusmu[i] * inv_first_sigma[i][j] * xminusmu[j];

                    log_p = -0.5 * n_exp - 0.5 * log(first_det) - 0.5 * (double) dim * (CONST_LN2 + log(PI));
                }

                xminusmu = null;

                break;
            case DIST_NORMALCL:
                xminusmu = new double[dim];

                /* Calculating x-mu */
                for (int i = 0; i < dim; i++)
                    xminusmu[i] = datum.rdata[dim_index[i]] - mu[i];

                /* Calculating (x-mu)'*sigma^(-1)*(x-mu) */
                for (int i = 0; i < dim; i++)
                    for (int j = 0; j < dim; j++)
                        n_exp += xminusmu[i] * inv_sigma[i][j] * xminusmu[j];

                xminusmu = null;

                log_p = -0.5 * n_exp - 0.5 * log(det) - 0.5 * (double) dim * (CONST_LN2 + log(PI));
                break;
            case DIST_LOGISTIC:
            case DIST_TRANSLOGISTIC:
            case DIST_TRANSLOGISTICG:
                /* Implemented in probLogistic */
                break;
            default:
                log_p = 0.0;
        }

        return (log_p);
    }

    void probLogistic(DataPoint datum, int prev_state, double[]output) {
        /* Calculates the probabilities of all states given the previous state and X */

        switch (type) {
            case DIST_LOGISTIC:
                for (int i = 0; i < num_states; i++) {
                    output[i] = lambda[i];
                    for (int i1 = 0; i1 < dim; i1++)
                        output[i] += rho[i][i1] * datum.rdata[i1];
                }
                normalize_logistic(output, output, num_states, 0);
                break;
            case DIST_TRANSLOGISTIC:
            case DIST_TRANSLOGISTICG:
                for (int i = 0; i < num_states; i++) {
                    if (prev_state < 0)
                        output[i] = lambda[i];
                    else
                        output[i] = sigma[prev_state][i];

                    for (int i1 = 0; i1 < dim; i1++)
                        output[i] += rho[i][i1] * datum.rdata[i1];
                }
                normalize_logistic(output, output, num_states, 0);
                break;
            default:
                ;
        }

        return;
    }

    double  log_prior() {
        /* Log-probability of the prior distribution */
        double lp = 0.0;

        switch (type) {
            case DIST_FACTOR:
                for (int i = 0; i < dim; i++)
                    lp += subdist[0][i].log_prior();
                break;
            case DIST_CHOWLIU:
                /* Unnormalized prior on the edge structures */

                /* MDL penalty on the number of variables */
                lp -= (double) num_edges * ((double) num_states - 1.0) * ((double) num_states - 1.0) * mdl_beta;

                /* Log-weight term */
                for (int e = 0; e < num_edges; e++)
                    lp += log(W[edge[e][0]][edge[e][1]]);

                /* Unnormalized prior on the parameters */
                /* Prior on univariate marginals first */
                for (int i = 0; i < dim; i++)
                    for (int j = 0; j < num_states; j++)
                        if (pcount_uni[i][j] > 0)
                            lp += pcount_uni[i][j] * log(ind_prob[i][j]);

                /* Prior on the edge parameters */
                for (int e = 0; e < num_edges; e++)
                    for (int i = 0; i < num_states; i++)
                        for (int j = 0; j < num_states; j++)
                            if (pcount_bi[edge[e][0]][edge[e][1]][i][j] > 0)
                                lp += pcount_bi[edge[e][0]][edge[e][1]][i][j]
                                        * (log(edge_prob[e][i][j]) - log(ind_prob[edge[e][0]][i]) - log(ind_prob[edge[e][1]][j]));

                break;
            case DIST_CONDCHOWLIU:
                lp -= (double) num_edges * ((double) num_states - 1.0) * ((double) num_states - 1.0) * mdl_beta;
                break;
            case DIST_BN_ME_BIVAR:
            case DIST_BN_CME_BIVAR:
                for (int i = 0; i < dim; i++)
                    lp -= mdl_beta * ((double) num_features[i] - 1);
                break;
            case DIST_BERNOULLI:
                for (int i = 0; i < num_states; i++)
                    if (abs(pcount_single[i]) > COMP_THRESHOLD)
                        lp += pcount_single[i] * log_state_prob[i];
                for (int i = 0; i < num_states; i++)
                    lp -= gammaln(pcount_single[i] + 1.0);
                lp += gammaln(pcount + (double) num_states);
                /* Mixture components */
                if (subdist != null)
                    for (int i = 0; i < dim; i++)
                        for (int j = 0; j < num_states; j++)
                            lp += subdist[i][j].log_prior();
                break;
            case DIST_CONDBERNOULLI:
            case DIST_CONDBERNOULLIG:
                /* Conditional probabilities first */
                for (int i = 0; i < num_states; i++)
                    for (int j = 0; j < num_states; j++)
                        if (abs(pcount_uni[i][j]) > COMP_THRESHOLD)
                            lp += pcount_uni[i][j] * log_cond_state_prob[i][j];
                /* Unnormalized */

                /* Marginal probabilities */
                for (int i = 0; i < num_states; i++)
                    if (abs(pcount_single[i]) > COMP_THRESHOLD)
                        lp += pcount_single[i] * log_state_prob[i];
                /* Unnormalized */
                break;
            case DIST_UNICONDMVME:
            case DIST_ME_BIVAR:
            case DIST_DELTAEXP:
            case DIST_DELTAGAMMA:
            case DIST_DIRACDELTA:
            case DIST_EXP:
            case DIST_GAMMA:
            case DIST_LOGNORMAL:
            case DIST_NORMAL:
            case DIST_NORMALCHAIN:
            case DIST_NORMALCL:
            case DIST_LOGISTIC:
            case DIST_TRANSLOGISTIC:
            case DIST_TRANSLOGISTICG:
                break;
            default:
                ;
        }

        return (lp);
    }

    double  log_un_probLogistic(DataPoint datum, int cur_state, int prev_state) {
        /* Calculates the log of unnormalized probabilities for the current state given the previsous state and X */

        double output = 0;

        switch (type) {
            case DIST_LOGISTIC:
                output = lambda[cur_state];
                for (int i1 = 0; i1 < dim; i1++)
                    output += rho[cur_state][i1] * datum.rdata[i1];
                break;
            case DIST_TRANSLOGISTIC:
            case DIST_TRANSLOGISTICG:
                if (prev_state < 0)
                    output = lambda[cur_state];
                else
                    output = sigma[prev_state][cur_state];

                for (int i1 = 0; i1 < dim; i1++)
                    output += rho[cur_state][i1] * datum.rdata[i1];
                break;
            default:
                ;
        }

        return (output);
    }

    void WriteToFile(PrintStream out) {
        /* Writing distribution to a file */

        /* Types of distribution are not written -- only the parameters */
        switch (type) {
            case DIST_FACTOR:
                for (int i = 0; i < dim; i++)
                    subdist[0][i].WriteToFile(out);
                break;
            case DIST_BERNOULLI:
                // out.format("%c Mapping of vector dimensions:\n", COMMENT_SYMBOL );
                if (subdist != null) {
                    out.format("%c %d-component mixture\n", COMMENT_SYMBOL, num_states);
                    out.format("%c Mixing probabilities:\n", COMMENT_SYMBOL);
                } else {
                    for (int i = 0; i < dim; i++)
                        out.format("%d ", dim_index[i]);
                    out.format("\n");
                }
                for (int i = 0; i < num_states; i++)
                    out.format("\t%.12f", state_prob[i]);
                out.format("\n");
                if (subdist != null)
                    /* Mixture components */
                    /* !!! Assuming a one-dimensional multinomial !!! */
                    for (int i = 0; i < num_states; i++) {
                        out.format("%c Mixture component %d:\n", COMMENT_SYMBOL, i + 1);
                        subdist[0][i].WriteToFile(output);
                    }
                break;
            case DIST_CONDBERNOULLI:
            case DIST_CONDBERNOULLIG:
                // out.format("%c Mapping of vector dimensions:\n", COMMENT_SYMBOL );
                for (int i = 0; i < dim; i++)
                    out.format("%d ", dim_index[i]);
                out.format("\n");
                out.format("%c First entry probabilities:\n", COMMENT_SYMBOL);
                for (int i = 0; i < num_states; i++)
                    out.format("\t%.12f", state_prob[i]);
                out.format("\n");
                out.format("%c Conditional probabilities:\n", COMMENT_SYMBOL);
                for (int i = 0; i < num_states; i++) {
                    for (int j = 0; j < num_states; j++)
                        out.format("\t%.12f", cond_state_prob[i][j]);
                    out.format("\n");
                }
                break;
            case DIST_UNICONDMVME:
                out.format("%c Number of functions:\n", COMMENT_SYMBOL);
                out.format("%d\n", dim);

                out.format("%c Number of features for each function:\n", COMMENT_SYMBOL);
                for (int i = 0; i < dim; i++)
                    out.format("%d ", num_features[i]);
                out.format("\n");

                out.format("%c Function feature indices:\n", COMMENT_SYMBOL);
                for (int i = 0; i < dim; i++) {
                    /* First feature index should be missing */
                    out.format("%f ", missing_value(0.0));
                    for (int j = 1; j < num_features[i]; j++)
                        out.format("%d ", feature_index[i][j]);
                    out.format("\n");
                }

                out.format("%c Corresponding function feature values:\n", COMMENT_SYMBOL);
                for (int i = 0; i < dim; i++) {
                    for (int j = 0; j < num_features[i]; j++)
                        if (is_missing(feature_value[i][j]))
                            /* Real-valued features do not have corresponding values */
                            out.format("%f ", missing_value(0.0));
                        else
                            out.format("%d ", feature_value[i][j]);
                }

                out.format("%c Weights:\n", COMMENT_SYMBOL);
                for (int i = 0; i < dim; i++)
                    out.format("\t%.12f", lambda[i]);
                out.format("\n");
                break;
            case DIST_CHOWLIU:
                // out.format("%c Mapping of vector dimensions:\n", COMMENT_SYMBOL );
                if (subdist != null) { /* Mixture */
                    out.format("%c %d-variable %d-component tree-structured mixture\n",
                            COMMENT_SYMBOL, dim, num_states);
                    out.format("%c Distribution for mixing variables:\n", COMMENT_SYMBOL);
                } /* Mixture */ else { /* Not a mixture */
                    for (int i = 0; i < dim; i++)
                        out.format("%d ", dim_index[i]);
                    out.format("\n");
                } /* Not a mixture */

                out.format("%c Marginal probabilities:\n", COMMENT_SYMBOL);
                for (int i = 0; i < dim; i++) {
                    for (int i1 = 0; i1 < num_states; i1++)
                        out.format("\t%.12f", ind_prob[i][i1]);
                    out.format("\n");
                }

                out.format("%c Number of edges:\n", COMMENT_SYMBOL);
                out.format("%d\n", num_edges);

                out.format("%c Edges:\n", COMMENT_SYMBOL);
                for (int i = 0; i < num_edges; i++) {
                    out.format("%d %d", edge[i][0], edge[i][1]);
                    out.format("\t%c MI: %f\n", COMMENT_SYMBOL, edge_MI[i]);
                }
                out.format("%c Edge probability tables\n", COMMENT_SYMBOL);
                for (int i = 0; i < num_edges; i++) {
                    for (int i1 = 0; i1 < num_states; i1++)
                        for (int i2 = 0; i2 < num_states; i2++)
                            out.format("\t%.12f", edge_prob[i][i1][i2]);
                    out.format("\n");
                }

                if (subdist != null)
                    /* Mixture */
                    for (int i = 0; i < dim; i++)
                        for (int j = 0; j < num_states; j++) {
                            out.format("%c Mixture component for variable %d value %d\n",
                                    COMMENT_SYMBOL, i + 1, j);
                            subdist[i][j].WriteToFile(output);
                        }
                break;
            case DIST_CONDCHOWLIU:
                // out.format("%c Mapping of vector dimensions:\n", COMMENT_SYMBOL );
                for (int i = 0; i < dim; i++)
                    out.format("%d ", dim_index[i]);
                out.format("\n");
                out.format("%c Marginal probabilities:\n", COMMENT_SYMBOL);
                for (int i = 0; i < 2 * dim; i++) {
                    for (int i1 = 0; i1 < num_states; i1++)
                        out.format("\t%.12f", ind_prob[i][i1]);
                    out.format("\n");
                }

                out.format("%c Number of edges:\n", COMMENT_SYMBOL);
                out.format("%d\n", num_edges);

                out.format("%c Edges:\n", COMMENT_SYMBOL);
                for (int i = 0; i < num_edges; i++) {
                    out.format("%d %d", edge[i][0], edge[i][1]);
                    out.format("\t%c MI: %f\n", COMMENT_SYMBOL, edge_MI[i]);
                }
                out.format("%c Edge probability tables\n", COMMENT_SYMBOL);
                for (int i = 0; i < num_edges; i++) {
                    for (int i1 = 0; i1 < num_states; i1++)
                        for (int i2 = 0; i2 < num_states; i2++)
                            out.format("\t%.12f", edge_prob[i][i1][i2]);
                    out.format("\n");
                }
                break;
            case DIST_ME_BIVAR:
                for (int i = 0; i < dim; i++)
                    out.format("%d ", dim_index[i]);
                out.format("\n");
                out.format("%c Coefficients:\n", COMMENT_SYMBOL);
                for (int i = 0; i < dim; i++) {
                    for (int j = 0; j < i + 1; j++)
                        out.format("\t%.12f", sigma[i][j]);
                    out.format("\n");
                }
                out.format("%c Exponent of the normalization constant:\n", COMMENT_SYMBOL);
                out.format("%.12f\n", det);
                break;
            case DIST_BN_ME_BIVAR:
                for (int i = 0; i < dim; i++)
                    out.format("%d ", dim_index[i]);
                out.format("\n");
                out.format("%c Numbers of features:\n", COMMENT_SYMBOL);
                for (int i = 0; i < dim; i++)
                    out.format("%d ", num_features[i]);
                out.format("\n");
                out.format("%c Features:\n", COMMENT_SYMBOL);
                for (int i = 0; i < dim; i++) {
                    for (int j = 0; j < num_features[i]; j++)
                        out.format("%d ", feature_index[i][j]);
                    out.format("\n");
                }
                out.format("%c Parameter values:\n", COMMENT_SYMBOL);
                for (int i = 0; i < dim; i++) {
                    for (int j = 0; j < num_features[i]; j++)
                        out.format("\t%.12f", sigma[i][j]);
                    out.format("\n");
                }
                out.format("%c Simulation order:\n", COMMENT_SYMBOL);
                for (int i = 0; i < dim; i++)
                    out.format("%d ", sim_order[i]);
                out.format("\n");
                break;
            case DIST_BN_CME_BIVAR:
                for (int i = 0; i < dim; i++)
                    out.format("%d ", dim_index[i]);
                out.format("\n");
                out.format("%c First entry probabilities of 1:\n", COMMENT_SYMBOL);
                for (int i = 0; i < dim; i++)
                    out.format("\t%.12f", state_prob[i]);
                out.format("\n");
                out.format("%c Numbers of features:\n", COMMENT_SYMBOL);
                for (int i = 0; i < dim; i++)
                    out.format("%d ", num_features[i]);
                out.format("\n");
                out.format("%c Features:\n", COMMENT_SYMBOL);
                for (int i = 0; i < dim; i++) {
                    for (int j = 0; j < num_features[i]; j++)
                        out.format("%d ", feature_index[i][j]);
                    out.format("\n");
                }
                out.format("%c Parameter values:\n", COMMENT_SYMBOL);
                for (int i = 0; i < dim; i++) {
                    for (int j = 0; j < num_features[i]; j++)
                        out.format("\t%.12f", sigma[i][j]);
                    out.format("\n");
                }
                out.format("%c Simulation order:\n", COMMENT_SYMBOL);
                for (int i = 0; i < dim; i++)
                    out.format("%d ", sim_order[i]);
                out.format("\n");
                break;
            case DIST_DELTAEXP:
                // out.format("%c Mapping of vector dimensions:\n", COMMENT_SYMBOL );
                for (int i = 0; i < dim; i++)
                    out.format("%d ", dim_index[i]);
                out.format("\n");
                out.format("%c Mixture component probabilities:\n", COMMENT_SYMBOL);
                for (int i = 0; i < num_states; i++)
                    out.format("\t%.12f", mix_prob[i]);
                out.format("\n");

                out.format("%c Parameters for exponential components:\n", COMMENT_SYMBOL);
                for (int i = 0; i < num_states - 1; i++)
                    out.format("%.12f\n", exp_param[i]);
                break;
            case DIST_DELTAGAMMA:
                // out.format("%c Mapping of vector dimensions:\n", COMMENT_SYMBOL );
                for (int i = 0; i < dim; i++)
                    out.format("%d ", dim_index[i]);
                out.format("\n");
                out.format("%c Mixture component probabilities:\n", COMMENT_SYMBOL);
                for (int i = 0; i < num_states; i++)
                    out.format("\t%.12f", mix_prob[i]);
                out.format("\n");

                out.format("%c Parameters for gamma components:\n", COMMENT_SYMBOL);
                for (int i = 0; i < num_states - 1; i++)
                    out.format("%.12f\t%.12f\n", gamma_param1[i], gamma_param2[i]);
                break;
            case DIST_DIRACDELTA:
                out.format("%c Dirac delta function at %.12f\n", COMMENT_SYMBOL, delta_value);
                out.format("%d\n", dim_index[0]);
                break;
            case DIST_EXP:
                out.format("%d\n", dim_index[0]);
                out.format("%c Parameter for geometric distribution:\n", COMMENT_SYMBOL);
                out.format("%.12f\n", exp_param1);
                break;
            case DIST_GAMMA:
                out.format("%d\n", dim_index[0]);
                out.format("%c Gamma distribution:\n", COMMENT_SYMBOL);
                out.format("%.12f\t%.12f\n", gamma1, gamma2);
                break;
            case DIST_LOGNORMAL:
                out.format("%d\n", dim_index[0]);
                out.format("%c Log-normal distribution:\n", COMMENT_SYMBOL);
                out.format("%.12f\t%.12f\n", log_normal1, log_normal2);
                break;
            case DIST_NORMAL:
                // out.format("%c Mapping of vector dimensions:\n", COMMENT_SYMBOL );
                for (int i = 0; i < dim; i++)
                    out.format("%d ", dim_index[i]);
                out.format("\n");
                out.format("%c Mean vector\n", COMMENT_SYMBOL);
                for (int i = 0; i < dim; i++)
                    out.format("\t%.12f", mu[i]);
                out.format("\n\n");
                out.format("%c Covariance matrix\n", COMMENT_SYMBOL);
                for (int i = 0; i < dim; i++) {
                    for (int j = 0; j < dim; j++)
                        out.format("\t%.12f", sigma[i][j]);
                    out.format("\n");
                }
                break;
            case DIST_NORMALCHAIN:
                // out.format("%c Mapping of vector dimensions:\n", COMMENT_SYMBOL );
                for (int i = 0; i < dim; i++)
                    out.format("%d ", dim_index[i]);
                out.format("\n");
                out.format("%c First state mean:\n", COMMENT_SYMBOL);
                for (int i = 0; i < dim; i++)
                    out.format("\t%.12f", first_mu[i]);
                out.format("\n\n");
                out.format("%c First state covariance matrix\n", COMMENT_SYMBOL);
                for (int i = 0; i < dim; i++) {
                    for (int j = 0; j < dim; j++)
                        out.format("\t%.12f", first_sigma[i][j]);
                    out.format("\n");
                }
                out.format("\n");

                out.format("%c Linear transformation:\n", COMMENT_SYMBOL);
                for (int i = 0; i < dim; i++) {
                    for (int j = 0; j < dim; j++)
                        out.format("\t%.12f", W[i][j]);
                    out.format("\n");
                }
                out.format("\n");

                out.format("%c Translation vector:\n", COMMENT_SYMBOL);
                for (int i = 0; i < dim; i++)
                    out.format("\t%.12f", mu[i]);
                out.format("\n");

                out.format("%c Noise covariance:\n", COMMENT_SYMBOL);
                for (int i = 0; i < dim; i++) {
                    for (int j = 0; j < dim; j++)
                        out.format("\t%.12f", sigma[i][j]);
                    out.format("\n");
                }

                break;
            case DIST_NORMALCL:
                // out.format("%c Mapping of vector dimensions:\n", COMMENT_SYMBOL );
                for (int i = 0; i < dim; i++)
                    out.format("%d ", dim_index[i]);
                out.format("\n");
                out.format("%c Mean vector\n", COMMENT_SYMBOL);
                for (int i = 0; i < dim; i++)
                    out.format("\t%.12f", mu[i]);
                out.format("\n\n");
                out.format("%c Covariance matrix\n", COMMENT_SYMBOL);
                for (int i = 0; i < dim; i++) {
                    for (int j = 0; j < dim; j++)
                        out.format("\t%.12f", sigma[i][j]);
                    out.format("\n");
                }

                out.format("%c Number of edges:\n", COMMENT_SYMBOL);
                out.format("%d\n", num_edges);

                out.format("%c Edges:\n", COMMENT_SYMBOL);
                for (int i = 0; i < num_edges; i++) {
                    out.format("%d %d", edge[i][0], edge[i][1]);
                    out.format("\t%c MI: %f\n", COMMENT_SYMBOL, edge_MI[i]);
                }

                break;
            case DIST_LOGISTIC:
                for (int s = 0; s < num_states; s++) {
                    out.format("%c State %d\n", COMMENT_SYMBOL, s + 1);

                    out.format("%c Constant term\n", COMMENT_SYMBOL);
                    out.format("%.12f\n\n", lambda[s]);

                    out.format("%c Linear terms\n", COMMENT_SYMBOL);
                    for (int i = 0; i < dim; i++)
                        out.format("\t%.12f", rho[s][i]);
                    out.format("\n\n");
                }
                break;
            case DIST_TRANSLOGISTIC:
            case DIST_TRANSLOGISTICG:
                for (int s = 0; s < num_states; s++) {
                    out.format("%c State %d\n", COMMENT_SYMBOL, s + 1);

                    out.format("%c Constant term\n", COMMENT_SYMBOL);
                    out.format("%.12f\n\n", lambda[s]);

                    out.format("%c Transition terms\n", COMMENT_SYMBOL);
                    for (int i = 0; i < num_states; i++)
                        out.format("\t%.12f", sigma[i][s]);
                    out.format("\n\n");

                    out.format("%c Linear terms\n", COMMENT_SYMBOL);
                    for (int i = 0; i < dim; i++)
                        out.format("\t%.12f", rho[s][i]);
                    out.format("\n\n");
                }
                break;
            default:
                ;
        }

        return;
    }

    void  WriteToFile2(PrintStream out) {
        /* Writing distribution to a file WITHOUT dimension indices */

        /* Types of distribution are not written -- only the parameters */
        switch (type) {
            case DIST_FACTOR:
                for (int i = 0; i < dim; i++)
                    subdist[0][i].WriteToFile2(out);
                break;
            case DIST_BERNOULLI:
                if (subdist != null) {
                    out.format("%c %d-component mixture\n", COMMENT_SYMBOL, num_states);
                    out.format("%c Mixing probabilities:\n", COMMENT_SYMBOL);
                }
                for (int i = 0; i < num_states; i++)
                    out.format("\t%.12f", state_prob[i]);
                out.format("\n");
                if (subdist != null)
                    /* Mixture components */
                    /* !!! Assuming a one-dimensional multinomial !!! */
                    for (int i = 0; i < num_states; i++) {
                        out.format("%c Mixture component %d:\n", COMMENT_SYMBOL, i + 1);
                        subdist[0][i].WriteToFile2(output);
                    }
                break;
            case DIST_CONDBERNOULLI:
            case DIST_CONDBERNOULLIG:
                out.format("%c First entry probabilities:\n", COMMENT_SYMBOL);
                for (int i = 0; i < num_states; i++)
                    out.format("\t%.12f", state_prob[i]);
                out.format("\n");
                out.format("%c Conditional probabilities:\n", COMMENT_SYMBOL);
                for (int i = 0; i < num_states; i++) {
                    for (int j = 0; j < num_states; j++)
                        out.format("\t%.12f", cond_state_prob[i][j]);
                    out.format("\n");
                }
                break;
            case DIST_UNICONDMVME:
                out.format("%c Number of functions:\n", COMMENT_SYMBOL);
                out.format("%d\n", dim);

                out.format("%c Number of features for each function:\n", COMMENT_SYMBOL);
                for (int i = 0; i < dim; i++)
                    out.format("%d ", num_features[i]);
                out.format("\n");

                out.format("%c Function feature indices:\n", COMMENT_SYMBOL);
                for (int i = 0; i < dim; i++) {
                    /* First feature index should be missing */
                    out.format("%f ", missing_value(0.0));
                    for (int j = 1; j < num_features[i]; j++)
                        out.format("%d ", feature_index[i][j]);
                    out.format("\n");
                }

                out.format("%c Corresponding function feature values:\n", COMMENT_SYMBOL);
                for (int i = 0; i < dim; i++) {
                    for (int j = 0; j < num_features[i]; j++)
                        if (is_missing(feature_value[i][j]))
                            /* Real-valued features do not have corresponding values */
                            out.format("%f ", missing_value(0.0));
                        else
                            out.format("%d ", feature_value[i][j]);
                }

                out.format("%c Weights:\n", COMMENT_SYMBOL);
                for (int i = 0; i < dim; i++)
                    out.format("\t%.12f", lambda[i]);
                out.format("\n");
                break;
            case DIST_CHOWLIU:
                if (subdist != null) { /* Mixture */
                    out.format("%c %d-variable %d-component tree-structured mixture\n",
                            COMMENT_SYMBOL, dim, num_states);
                    out.format("%c Distribution for mixing variables:\n", COMMENT_SYMBOL);
                } /* Mixture */

                out.format("%c Marginal probabilities:\n", COMMENT_SYMBOL);
                for (int i = 0; i < dim; i++) {
                    for (int i1 = 0; i1 < num_states; i1++)
                        out.format("\t%.12f", ind_prob[i][i1]);
                    out.format("\n");
                }

                out.format("%c Number of edges:\n", COMMENT_SYMBOL);
                out.format("%d\n", num_edges);

                out.format("%c Edges:\n", COMMENT_SYMBOL);
                for (int i = 0; i < num_edges; i++) {
                    out.format("%d %d", edge[i][0], edge[i][1]);
                    out.format("\t%c MI: %f\n", COMMENT_SYMBOL, edge_MI[i]);
                }
                out.format("%c Edge probability tables\n", COMMENT_SYMBOL);
                for (int i = 0; i < num_edges; i++) {
                    for (int i1 = 0; i1 < num_states; i1++)
                        for (int i2 = 0; i2 < num_states; i2++)
                            out.format("\t%.12f", edge_prob[i][i1][i2]);
                    out.format("\n");
                }
                if (subdist != null)
                    /* Mixture */
                    for (int i = 0; i < dim; i++)
                        for (int j = 0; j < num_states; j++) {
                            out.format("%c Mixture component for variable %d value %d\n",
                                    COMMENT_SYMBOL, i + 1, j);
                            subdist[i][j].WriteToFile2(output);
                        }
                break;
            case DIST_CONDCHOWLIU:
                out.format("%c Marginal probabilities:\n", COMMENT_SYMBOL);
                for (int i = 0; i < 2 * dim; i++) {
                    for (int i1 = 0; i1 < num_states; i1++)
                        out.format("\t%.12f", ind_prob[i][i1]);
                    out.format("\n");
                }

                out.format("%c Number of edges:\n", COMMENT_SYMBOL);
                out.format("%d\n", num_edges);

                out.format("%c Edges:\n", COMMENT_SYMBOL);
                for (int i = 0; i < num_edges; i++) {
                    out.format("%d %d", edge[i][0], edge[i][1]);
                    out.format("\t%c MI: %f\n", COMMENT_SYMBOL, edge_MI[i]);
                }
                out.format("%c Edge probability tables\n", COMMENT_SYMBOL);
                for (int i = 0; i < num_edges; i++) {
                    for (int i1 = 0; i1 < num_states; i1++)
                        for (int i2 = 0; i2 < num_states; i2++)
                            out.format("\t%.12f", edge_prob[i][i1][i2]);
                    out.format("\n");
                }
                break;
            case DIST_ME_BIVAR:
                out.format("%c Coefficients:\n", COMMENT_SYMBOL);
                for (int i = 0; i < dim; i++) {
                    for (int j = 0; j < i + 1; j++)
                        out.format("\t%.12f", sigma[i][j]);
                    out.format("\n");
                }
                out.format("%c Exponent of the normalization constant:\n", COMMENT_SYMBOL);
                out.format("%.12f\n", det);
                break;
            case DIST_BN_ME_BIVAR:
                out.format("%c Numbers of features:\n", COMMENT_SYMBOL);
                for (int i = 0; i < dim; i++)
                    out.format("%d ", num_features[i]);
                out.format("\n");
                out.format("%c Features:\n", COMMENT_SYMBOL);
                for (int i = 0; i < dim; i++) {
                    for (int j = 0; j < num_features[i]; j++)
                        out.format("%d ", feature_index[i][j]);
                    out.format("\n");
                }
                out.format("%c Parameter values:\n", COMMENT_SYMBOL);
                for (int i = 0; i < dim; i++) {
                    for (int j = 0; j < num_features[i]; j++)
                        out.format("\t%.12f", sigma[i][j]);
                    out.format("\n");
                }
                out.format("%c Simulation order:\n", COMMENT_SYMBOL);
                for (int i = 0; i < dim; i++)
                    out.format("%d ", sim_order[i]);
                out.format("\n");
                break;
            case DIST_BN_CME_BIVAR:
                out.format("%c First entry probabilities of 1:\n", COMMENT_SYMBOL);
                for (int i = 0; i < dim; i++)
                    out.format("\t%.12f", state_prob[i]);
                out.format("\n");
                out.format("%c Numbers of features:\n", COMMENT_SYMBOL);
                for (int i = 0; i < dim; i++)
                    out.format("%d ", num_features[i]);
                out.format("\n");
                out.format("%c Features:\n", COMMENT_SYMBOL);
                for (int i = 0; i < dim; i++) {
                    for (int j = 0; j < num_features[i]; j++)
                        out.format("%d ", feature_index[i][j]);
                    out.format("\n");
                }
                out.format("%c Parameter values:\n", COMMENT_SYMBOL);
                for (int i = 0; i < dim; i++) {
                    for (int j = 0; j < num_features[i]; j++)
                        out.format("\t%.12f", sigma[i][j]);
                    out.format("\n");
                }
                out.format("%c Simulation order:\n", COMMENT_SYMBOL);
                for (int i = 0; i < dim; i++)
                    out.format("%d ", sim_order[i]);
                out.format("\n");
                break;
            case DIST_DELTAEXP:
                out.format("%c Mixture component probabilities:\n", COMMENT_SYMBOL);
                for (int i = 0; i < num_states; i++)
                    out.format("\t%.12f", mix_prob[i]);
                out.format("\n");

                out.format("%c Parameters for exponential components:\n", COMMENT_SYMBOL);
                for (int i = 0; i < num_states - 1; i++)
                    out.format("%.12f\n", exp_param[i]);
                break;
            case DIST_DELTAGAMMA:
                out.format("%c Mixture component probabilities:\n", COMMENT_SYMBOL);
                for (int i = 0; i < num_states; i++)
                    out.format("\t%.12f", mix_prob[i]);
                out.format("\n");

                out.format("%c Parameters for gamma components:\n", COMMENT_SYMBOL);
                for (int i = 0; i < num_states - 1; i++)
                    out.format("%.12f\t%.12f\n", gamma_param1[i], gamma_param2[i]);
                break;
            case DIST_DIRACDELTA:
                out.format("%c Dirac delta function at %.12f\n", COMMENT_SYMBOL, delta_value);
                break;
            case DIST_EXP:
                out.format("%c Parameter for geometric distribution:\n", COMMENT_SYMBOL);
                out.format("%.12f\n", exp_param1);
                break;
            case DIST_GAMMA:
                out.format("%c Gamma distribution:\n", COMMENT_SYMBOL);
                out.format("%.12f\t%.12f\n", gamma1, gamma2);
                break;
            case DIST_LOGNORMAL:
                out.format("%c Log-normal distribution:\n", COMMENT_SYMBOL);
                out.format("%.12f\t%.12f\n", log_normal1, log_normal2);
                break;
            case DIST_NORMAL:
                out.format("%c Mean vector\n", COMMENT_SYMBOL);
                for (int i = 0; i < dim; i++)
                    out.format("\t%.12f", mu[i]);
                out.format("\n\n");
                out.format("%c Covariance matrix\n", COMMENT_SYMBOL);
                for (int i = 0; i < dim; i++) {
                    for (int j = 0; j < dim; j++)
                        out.format("\t%.12f", sigma[i][j]);
                    out.format("\n");
                }
                break;
            case DIST_NORMALCHAIN:
                out.format("%c First state mean:\n", COMMENT_SYMBOL);
                for (int i = 0; i < dim; i++)
                    out.format("\t%.12f", first_mu[i]);
                out.format("\n\n");
                out.format("%c First state covariance matrix\n", COMMENT_SYMBOL);
                for (int i = 0; i < dim; i++) {
                    for (int j = 0; j < dim; j++)
                        out.format("\t%.12f", first_sigma[i][j]);
                    out.format("\n");
                }
                out.format("\n");

                out.format("%c Linear transformation:\n", COMMENT_SYMBOL);
                for (int i = 0; i < dim; i++) {
                    for (int j = 0; j < dim; j++)
                        out.format("\t%.12f", W[i][j]);
                    out.format("\n");
                }
                out.format("\n");

                out.format("%c Translation vector:\n", COMMENT_SYMBOL);
                for (int i = 0; i < dim; i++)
                    out.format("\t%.12f", mu[i]);
                out.format("\n");

                out.format("%c Noise covariance:\n", COMMENT_SYMBOL);
                for (int i = 0; i < dim; i++) {
                    for (int j = 0; j < dim; j++)
                        out.format("\t%.12f", sigma[i][j]);
                    out.format("\n");
                }

                break;
            case DIST_NORMALCL:
                out.format("%c Mean vector\n", COMMENT_SYMBOL);
                for (int i = 0; i < dim; i++)
                    out.format("\t%.12f", mu[i]);
                out.format("\n\n");
                out.format("%c Covariance matrix\n", COMMENT_SYMBOL);
                for (int i = 0; i < dim; i++) {
                    for (int j = 0; j < dim; j++)
                        out.format("\t%.12f", sigma[i][j]);
                    out.format("\n");
                }

                out.format("%c Number of edges:\n", COMMENT_SYMBOL);
                out.format("%d\n", num_edges);

                out.format("%c Edges:\n", COMMENT_SYMBOL);
                for (int i = 0; i < num_edges; i++) {
                    out.format("%d %d", edge[i][0], edge[i][1]);
                    out.format("\t%c MI: %f\n", COMMENT_SYMBOL, edge_MI[i]);
                }

                break;
            case DIST_LOGISTIC:
                for (int s = 0; s < num_states; s++) {
                    out.format("%c State %d\n", COMMENT_SYMBOL, s + 1);

                    out.format("%c Constant term\n", COMMENT_SYMBOL);
                    out.format("%.12f\n\n", lambda[s]);

                    out.format("%c Linear terms\n", COMMENT_SYMBOL);
                    for (int i = 0; i < dim; i++)
                        out.format("\t%.12f", rho[s][i]);
                    out.format("\n\n");
                }
                break;
            case DIST_TRANSLOGISTIC:
            case DIST_TRANSLOGISTICG:
                for (int s = 0; s < num_states; s++) {
                    out.format("%c State %d\n", COMMENT_SYMBOL, s + 1);

                    out.format("%c Constant term\n", COMMENT_SYMBOL);
                    out.format("%.12f\n\n", lambda[s]);

                    out.format("%c Transition terms\n", COMMENT_SYMBOL);
                    for (int i = 0; i < num_states; i++)
                        out.format("\t%.12f", sigma[i][s]);
                    out.format("\n\n");

                    out.format("%c Linear terms\n", COMMENT_SYMBOL);
                    for (int i = 0; i < dim; i++)
                        out.format("\t%.12f", rho[s][i]);
                    out.format("\n\n");
                }
                break;
            default:
                ;
        }

        return;
    }

    void  WriteToFileBare(PrintStream out) {
        /* Writing distribution to a file WITHOUT dimension indices */

        /* Types of distribution are not written -- only the parameters */
        switch (type) {
            case DIST_FACTOR:
                for (int i = 0; i < dim; i++)
                    subdist[0][i].WriteToFileBare(out);
                break;
            case DIST_BERNOULLI:
                for (int i = 0; i < num_states; i++)
                    out.format("\t%.12f", state_prob[i]);
                out.format("\n");
                if (subdist != null)
                    /* Mixture components */
                    /* !!! Assuming a one-dimensional multinomial !!! */
                    for (int i = 0; i < num_states; i++)
                        subdist[0][i].WriteToFileBare(output);
                break;
            case DIST_CONDBERNOULLI:
            case DIST_CONDBERNOULLIG:
                for (int i = 0; i < num_states; i++)
                    out.format("\t%.12f", state_prob[i]);
                out.format("\n");
                for (int i = 0; i < num_states; i++) {
                    for (int j = 0; j < num_states; j++)
                        out.format("\t%.12f", cond_state_prob[i][j]);
                    out.format("\n");
                }
                break;
            case DIST_UNICONDMVME:
                out.format("%d\n", dim);
                for (int i = 0; i < dim; i++)
                    out.format("%d ", num_features[i]);
                out.format("\n");
                for (int i = 0; i < dim; i++) {
                    /* First feature index should be missing */
                    out.format("%f ", missing_value(0.0));
                    for (int j = 1; j < num_features[i]; j++)
                        out.format("%d ", feature_index[i][j]);
                    out.format("\n");
                }
                for (int i = 0; i < dim; i++) {
                    for (int j = 0; j < num_features[i]; j++)
                        if (is_missing(feature_value[i][j]))
                            /* Real-valued features do not have corresponding values */
                            out.format("%f ", missing_value(0.0));
                        else
                            out.format("%d ", feature_value[i][j]);
                }
                for (int i = 0; i < dim; i++)
                    out.format("\t%.12f", lambda[i]);
                out.format("\n");
                break;
            case DIST_CHOWLIU:
                for (int i = 0; i < dim; i++) {
                    for (int i1 = 0; i1 < num_states; i1++)
                        out.format("\t%.12f", ind_prob[i][i1]);
                    out.format("\n");
                }
                out.format("%d\n", num_edges);
                for (int i = 0; i < num_edges; i++)
                    out.format("%d %d\n", edge[i][0], edge[i][1]);
                for (int i = 0; i < num_edges; i++) {
                    for (int i1 = 0; i1 < num_states; i1++)
                        for (int i2 = 0; i2 < num_states; i2++)
                            out.format("\t%.12f", edge_prob[i][i1][i2]);
                    out.format("\n");
                }
                if (subdist != null)
                    /* Mixture */
                    for (int i = 0; i < dim; i++)
                        for (int j = 0; j < num_states; j++)
                            subdist[i][j].WriteToFileBare(output);
                break;
            case DIST_CONDCHOWLIU:
                for (int i = 0; i < 2 * dim; i++) {
                    for (int i1 = 0; i1 < num_states; i1++)
                        out.format("\t%.12f", ind_prob[i][i1]);
                    out.format("\n");
                }
                out.format("%d\n", num_edges);
                for (int i = 0; i < num_edges; i++)
                    out.format("%d %d\n", edge[i][0], edge[i][1]);
                for (int i = 0; i < num_edges; i++) {
                    for (int i1 = 0; i1 < num_states; i1++)
                        for (int i2 = 0; i2 < num_states; i2++)
                            out.format("\t%.12f", edge_prob[i][i1][i2]);
                    out.format("\n");
                }
                break;
            case DIST_ME_BIVAR:
                for (int i = 0; i < dim; i++) {
                    for (int j = 0; j < i + 1; j++)
                        out.format("\t%.12f", sigma[i][j]);
                    out.format("\n");
                }
                out.format("%.12f\n", det);
                break;
            case DIST_BN_ME_BIVAR:
                for (int i = 0; i < dim; i++)
                    out.format("%d ", num_features[i]);
                out.format("\n");
                for (int i = 0; i < dim; i++) {
                    for (int j = 0; j < num_features[i]; j++)
                        out.format("%d ", feature_index[i][j]);
                    out.format("\n");
                }
                for (int i = 0; i < dim; i++) {
                    for (int j = 0; j < num_features[i]; j++)
                        out.format("\t%.12f", sigma[i][j]);
                    out.format("\n");
                }
                for (int i = 0; i < dim; i++)
                    out.format("%d ", sim_order[i]);
                out.format("\n");
                break;
            case DIST_BN_CME_BIVAR:
                for (int i = 0; i < dim; i++)
                    out.format("\t%.12f", state_prob[i]);
                out.format("\n");
                for (int i = 0; i < dim; i++)
                    out.format("%d ", num_features[i]);
                out.format("\n");
                for (int i = 0; i < dim; i++) {
                    for (int j = 0; j < num_features[i]; j++)
                        out.format("%d ", feature_index[i][j]);
                    out.format("\n");
                }
                for (int i = 0; i < dim; i++) {
                    for (int j = 0; j < num_features[i]; j++)
                        out.format("\t%.12f", sigma[i][j]);
                    out.format("\n");
                }
                for (int i = 0; i < dim; i++)
                    out.format("%d ", sim_order[i]);
                out.format("\n");
                break;
            case DIST_DELTAEXP:
                for (int i = 0; i < num_states; i++)
                    out.format("\t%.12f", mix_prob[i]);
                out.format("\n");
                for (int i = 0; i < num_states - 1; i++)
                    out.format("%.12f\n", exp_param[i]);
                break;
            case DIST_DELTAGAMMA:
                for (int i = 0; i < num_states; i++)
                    out.format("\t%.12f", mix_prob[i]);
                out.format("\n");
                for (int i = 0; i < num_states - 1; i++)
                    out.format("%.12f\t%.12f\n", gamma_param1[i], gamma_param2[i]);
                break;
            case DIST_DIRACDELTA:
                break;
            case DIST_EXP:
                out.format("%.12f\n", exp_param1);
                break;
            case DIST_GAMMA:
                out.format("%.12f\t%.12f\n", gamma1, gamma2);
                break;
            case DIST_LOGNORMAL:
                out.format("%.12f\t%.12f\n", log_normal1, log_normal2);
                break;
            case DIST_NORMAL:
                for (int i = 0; i < dim; i++)
                    out.format("\t%.12f", mu[i]);
                out.format("\n");
                for (int i = 0; i < dim; i++) {
                    for (int j = 0; j < dim; j++)
                        out.format("\t%.12f", sigma[i][j]);
                    out.format("\n");
                }
                break;
            case DIST_NORMALCHAIN:
                for (int i = 0; i < dim; i++)
                    out.format("\t%.12f", first_mu[i]);
                out.format("\n");
                for (int i = 0; i < dim; i++) {
                    for (int j = 0; j < dim; j++)
                        out.format("\t%.12f", first_sigma[i][j]);
                    out.format("\n");
                }
                for (int i = 0; i < dim; i++) {
                    for (int j = 0; j < dim; j++)
                        out.format("\t%.12f", W[i][j]);
                    out.format("\n");
                }
                out.format("\n");
                for (int i = 0; i < dim; i++)
                    out.format("\t%.12f", mu[i]);
                out.format("\n");
                for (int i = 0; i < dim; i++) {
                    for (int j = 0; j < dim; j++)
                        out.format("\t%.12f", sigma[i][j]);
                    out.format("\n");
                }
                break;
            case DIST_NORMALCL:
                for (int i = 0; i < dim; i++)
                    out.format("\t%.12f", mu[i]);
                out.format("\n");
                for (int i = 0; i < dim; i++) {
                    for (int j = 0; j < dim; j++)
                        out.format("\t%.12f", sigma[i][j]);
                    out.format("\n");
                }
                out.format("%d\n", num_edges);
                for (int i = 0; i < num_edges; i++)
                    out.format("%d %d\n", edge[i][0], edge[i][1]);
                break;
            case DIST_LOGISTIC:
                for (int s = 0; s < num_states; s++) {
                    out.format("%.12f\n", lambda[s]);
                    for (int i = 0; i < dim; i++)
                        out.format("\t%.12f", rho[s][i]);
                    out.format("\n");
                }
                break;
            case DIST_TRANSLOGISTIC:
            case DIST_TRANSLOGISTICG:
                for (int s = 0; s < num_states; s++) {
                    out.format("%.12f\n", lambda[s]);
                    for (int i = 0; i < num_states; i++)
                        out.format("\t%.12f", sigma[i][s]);
                    out.format("\n");
                    for (int i = 0; i < dim; i++)
                        out.format("\t%.12f", rho[s][i]);
                    out.format("\n");
                }
                break;
            default:
                ;
        }

        return;
    }

    int  num_params() {
        /* Number of free parameters in the distribution */
        int num = 0;

        switch (type) {
            case DIST_FACTOR:
                for (int i = 0; i < dim; i++)
                    num += subdist[0][i].num_params();
                break;
            case DIST_BERNOULLI:
                num += num_states - 1;
                if (subdist != null)
                    /* Mixture components */
                    for (int i = 0; i < dim; i++)
                        for (int j = 0; j < num_states; j++)
                            num += subdist[i][j].num_params();
                break;
            case DIST_CONDBERNOULLI:
            case DIST_CONDBERNOULLIG:
                /* First sequence entry probabilities */
                num += num_states - 1;
                /* Conditional probabilities */
                num += num_states * (num_states - 1);
                break;
            case DIST_UNICONDMVME:
                /* !!! Only counting weights !!! */
                num += dim;
                break;
            case DIST_CHOWLIU:
                /* Quite tricky */

    /*
       Let nc = number of outcomes, assuming all nodes have the same number of outcomes.
       Each node contributes nc-1 parameters corresponding to probability values.
       Each edge contributes nc^2-1-2*(nc-1)=(nc-1)^2 parameters corresponding to
       probability values.

       BIC implemented here does not count the parameters indicating what edges are
       selected.  At this point in time, it is not clear how to include structure
       specification parameters into BIC.
    */

                /* !!! Assuming all nodes have the same number of outcomes !!! */
                /* Counting node contributions */
                num += dim * (num_states - 1);

                /* Counting edge contributions */
                num += num_edges * (num_states - 1) * (num_states - 1);

                if (subdist != null)
                    /* Mixture */
                    for (int i = 0; i < dim; i++)
                        for (int j = 0; j < num_states; j++)
                            num += subdist[i][j].num_params();
                break;
            case DIST_CONDCHOWLIU:
                /* Quite tricky */

    /*
       Let nc = number of outcomes, assuming all nodes have the same number of outcomes.
       Each node contributes nc-1 parameters corresponding to probability values.
       Each edge contributes nc^2-1-2*(nc-1)=(nc-1)^2 parameters corresponding to
       probability values.

       We need to include nodes for both time t and t-1.

       BIC implemented here does not count the parameters indicating what edges are
       selected.  At this point in time, it is not clear how to include structure
       specification parameters into BIC.
    */

                /* !!! Assuming all nodes have the same number of outcomes !!! */
                /* Counting node contributions */
                num += 2 * dim * (num_states - 1);

                /* Counting edge contributions */
                num += num_edges * (num_states - 1) * (num_states - 1);

                break;
            case DIST_ME_BIVAR:
                num += dim * (dim + 1) / 2;
                break;
            case DIST_BN_ME_BIVAR:
                num = 0;
                for (int i = 0; i < dim; i++)
                    num += num_features[i];
                /* !!! Not counting indices indicating which features are selected !!! */
                break;
            case DIST_BN_CME_BIVAR:
                num = dim;  // First entry probabilities of 1
                for (int i = 0; i < dim; i++)
                    num += num_features[i];
                /* !!! Not counting indices indicating which features are selected !!! */
                break;
            case DIST_DELTAEXP:
                /* Mixing probabilities */
                num += num_states - 1;
                /* Exponential parameters */
                num += num_states - 1;
                break;
            case DIST_DELTAGAMMA:
                /* Mixing probabilities */
                num += num_states - 1;
                /* Gamma parameters */
                num += 2 * (num_states - 1);
                break;
            case DIST_DIRACDELTA:
                break;
            case DIST_EXP:
                num += 1;
                break;
            case DIST_GAMMA:
                num += 2;
                break;
            case DIST_LOGNORMAL:
                num += 2;
                break;
            case DIST_NORMAL:
                /* Mean vector */
                num += dim;
                /* Covariance matrix */
                num += dim * (dim + 1) / 2;
                break;
            case DIST_NORMALCHAIN:
                /* First state mean vector */
                num += dim;
                /* First state covariance matrix */
                num += dim * (dim + 1) / 2;
                /* Linear transformation matrix */
                num += dim ^ 2;
                /* Translation vector */
                num += dim;
                /* Noise covariance matrix */
                num += dim * (dim + 1) / 2;
                break;
            case DIST_NORMALCL:
                /* Mean vector */
                num += dim;
                /* Tree-structured covariance matrix */
                num += (dim + num_edges);
                break;
            case DIST_LOGISTIC:
                /* First state */
                num += num_states - 1;
                /* Linear terms */
                num += dim * (num_states - 1);
                break;
            case DIST_TRANSLOGISTIC:
            case DIST_TRANSLOGISTICG:
                /* First state */
                num += num_states - 1;
                /* Transition */
                num += num_states * (num_states - 1);
                /* Linear terms */
                num += dim * (num_states - 1);
                break;
            default:
                ;
        }

        return (num);
    }

    void  UpdateEmissionParameters(Data data, double[][]prob_s, double norm_const) {
  /* data -- the output data
     prob_s -- p_old(.|R_nt,X_nt)
     norm_const -- sum_n sum_{t_n} p_old(.|R_nt,X_nt)
  */

        switch (type) {
            case DIST_FACTOR:
                /* Conditionally independent components */

                /* Updating parameters of the sub-components */
                for (int i = 0; i < dim; i++)
                    subdist[0][i].UpdateEmissionParameters(data, prob_s, norm_const);
                break;
            case DIST_BERNOULLI:
                /* Univariate Bernoulli */
                UpdateEmissionBernoulli(data, prob_s);
                break;
            case DIST_CONDBERNOULLI:
                /* Univariate conditional Bernoulli */
                UpdateEmissionConditionalBernoulli(data, prob_s, 0);
                break;
            case DIST_CONDBERNOULLIG:
                /* Univariate conditional Bernoulli with averaged first entry probabilities */
                UpdateEmissionConditionalBernoulli(data, prob_s, 1);
            case DIST_UNICONDMVME:
                /* Polytomous logistic distribution */

                /* !!! Updated in a different manner !!! */

                break;
            case DIST_CHOWLIU:
                /* Chow-Liu tree */
                UpdateEmissionChowLiu(data, prob_s);
                break;
            case DIST_CONDCHOWLIU:
                /* Conditional Chow-Liu tree */
                UpdateEmissionConditionalChowLiu(data, prob_s, norm_const);
                break;
            case DIST_ME_BIVAR:
                /* Full bivariate MaxEnt */
                UpdateEmissionFullBivariateMaxEnt(data, prob_s, norm_const);
                break;
            case DIST_BN_ME_BIVAR:
                /* PUC-MaxEnt */
                UpdateEmissionPUCMaxEnt(data, prob_s, norm_const);
                break;
            case DIST_BN_CME_BIVAR:
                /* Conditional PUC-MaxEnt */
                UpdateEmissionConditionalPUCMaxEnt(data, prob_s, norm_const);
                break;
            case DIST_DELTAEXP:
                /* Mixture of one delta function and multiple exponential functions */
                UpdateEmissionDeltaExponential(data, prob_s, norm_const);
                break;
            case DIST_DELTAGAMMA:
                /* Mixture of one gamma function and multiple gamma functions */
                UpdateEmissionDeltaGamma(data, prob_s, norm_const);
                break;
            case DIST_DIRACDELTA:
                /* Dirac's delta */
                break;
            case DIST_EXP:
                /* Exponential (geometric) distribution */
                UpdateEmissionExponential(data, prob_s);
                break;
            case DIST_GAMMA:
                /* Gamma distribution */
                UpdateEmissionGamma(data, prob_s);
                break;
            case DIST_LOGNORMAL:
                /* Log-normal distribution */
                UpdateEmissionLognormal(data, prob_s);
                break;
            case DIST_NORMAL:
                /* Multivariate Gaussian distribution */
                UpdateEmissionGaussian(data, prob_s, norm_const);
                break;
            case DIST_NORMALCHAIN:
                /* Auto-regressive Gaussian */
                UpdateEmissionARGaussian(data, prob_s, norm_const);
                break;
            case DIST_NORMALCL:
                /* Gaussian with tree structured covariance matrix */
                UpdateEmissionGaussianChowLiu(data, prob_s, norm_const);
                break;
            case DIST_LOGISTIC:
            case DIST_TRANSLOGISTIC:
            case DIST_TRANSLOGISTICG:
                System.err.print("Cannot have logistic distribution as an emission!\n");
                System.exit(-1);
                break;
            default:
                ;
        }

    }

    double  weighted_ll(Data data, double[][]prob_s) {
  /* The total expected log-likelihood of the data under the distribution
     and the probabilities of the data points */

        double ll = 0.0;

        for (int n = 0; n < data.num_seqs; n++)
            for (int t = 0; t < data.sequence[n].seq_length;
                 t++)
                if (t == 0)
                    ll += prob_s[n][t] * log_prob(data.sequence[n].entry[t], null);
                else
                    ll += prob_s[n][t] * log_prob(data.sequence[n].entry[t], data.sequence[n].entry[t - 1]);

        return (ll);
    }

    void  UpdateTransitionParameters(Data data,
                               Data input,
                               int iteration) {

        /* Updating the parameters of the input distribution */

  /* Since the logistic dunction does not have a closed-form update rule,
     need to perform the update iteratively */

        int num_grad_runs;    // Number of need to choose a new direction vector

        Distribution g;       // Current gradient
        Distribution g_prev;  // Previous gradient
        Distribution h;       // Current direction vector

        double a, b;

        /* Temporary variable(s) */
        double sum;

        switch (type) {
            case DIST_LOGISTIC:
            case DIST_TRANSLOGISTIC:
            case DIST_TRANSLOGISTICG:
                if (iteration > NUM_BURNOFF3_ITER)
                    num_grad_runs = 4;
                else if (iteration > NUM_BURNOFF2_ITER)
                    num_grad_runs = 3;
                else if (iteration > NUM_BURNOFF1_ITER)
                    num_grad_runs = 2;
                else
                    num_grad_runs = 1;

                /* Calculating the initial gradient vector */
                g = logistic_gradient(input, uni_prob[0], joint_prob, log_un_prob);

                /* Maximizing the objective function in the direction of the gradient */
                h = g.copy();

                linear_search_logistic_NR(input, uni_prob[0], joint_prob, log_un_prob, h, iteration);

                for (int index = 1; index < num_grad_runs; index++) {
                    /* Performing an iteration of Polak-Ribiere conjugate gradient algorithm */
                    g_prev = g;
                    g = logistic_gradient(input, uni_prob[0], joint_prob, log_un_prob);

                    /* Calculating the adjustment coefficient */

                    /* Denominator */
                    b = 0.0;
                    for (int i = 1; i < num_states; i++) {
                        b += g_prev.lambda[i] * g_prev.lambda[i];
                        if (type == DIST_TRANSLOGISTIC)
                            for (int j = 0; j < num_states; j++)
                                b += g_prev.sigma[j][i] * g_prev.sigma[j][i];
                        for (int j = 0; j < dim; j++)
                            b += g_prev.rho[i][j] * g_prev.rho[i][j];
                    }

                    /* Numerator */
                    a = 0.0;
                    for (int i = 1; i < num_states; i++) {
                        a += g.lambda[i] * (g.lambda[i] - g_prev.lambda[i]);
                        if (type == DIST_TRANSLOGISTIC)
                            for (int j = 0; j < num_states; j++)
                                a += g.sigma[j][i] * (g.sigma[j][i] - g_prev.sigma[j][i]);
                        for (int j = 0; j < dim; j++)
                            a += g.rho[i][j] * (g.rho[i][j] - g_prev.rho[i][j]);
                    }

                    /* Deallocating the previous gradient */
                    g_prev = null;

                    a /= b;

                    /* New vector h=g+a*h */
                    for (int i = 1; i < num_states; i++) {
                        h.lambda[i] = g.lambda[i] + a * h.lambda[i];
                        if (type == DIST_TRANSLOGISTIC)
                            for (int j = 0; j < num_states; j++)
                                h.sigma[j][i] = g.sigma[j][i] + a * h.sigma[j][i];
                        for (int j = 0; j < dim; j++)
                            h.rho[i][j] = g.rho[i][j] + a * h.rho[i][j];
                    }

                    linear_search_logistic_NR(input, uni_prob[0], joint_prob, log_un_prob, h, iteration);
                }

                /* Deallocating the current gradient */
                g = null;

                /* Deallocating the direction vector */
                h = null;

                break;
            case DIST_BERNOULLI:
                /* State probabilities */
                for (int i = 0; i < num_states; i++) {
                    state_prob[i] = pcount_single[i];
                    for (int n = 0; n < data.num_seqs; n++)
                        for (int t = 0; t < data.sequence[n].seq_length;
                             t++)
                            state_prob[i] += uni_prob[0][i][n][t];
                }

                sum = 0.0;
                for (int i = 0; i < num_states; i++)
                    sum += state_prob[i];

                /* Normalizing */
                for (int i = 0; i < num_states; i++)
                    state_prob[i] /= sum;
                for (int i = 0; i < num_states; i++)
                    log_state_prob[i] = log(state_prob[i]);

                break;
            case DIST_CONDBERNOULLI:
                /* First state probabilities */
                sum = 0.0;
                for (int i = 0; i < num_states; i++) {
                    state_prob[i] = pcount_single[i];
                    for (int n = 0; n < data.num_seqs; n++)
                        state_prob[i] += uni_prob[0][i][n][0];
                    sum += state_prob[i];
                }

                /* Normalizing */
                for (int i = 0; i < num_states; i++)
                    state_prob[i] /= sum;
                for (int i = 0; i < num_states; i++)
                    log_state_prob[i] = log(state_prob[i]);

                /* Transition probabilities */
                for (int j = 0; j < num_states; j++) { /* Summing over S_{t-1} */
                    sum = 0.0;
                    for (int i = 0; i < num_states; i++) { /* Summing over S_t */
                        cond_state_prob[j][i] = pcount_uni[j][i];

                        for (int n = 0; n < data.num_seqs; n++)
                            for (int t = 1; t < data.sequence[n].seq_length;
                                 t++)
                                cond_state_prob[j][i] += joint_prob[i][j][n][t];

                        sum += cond_state_prob[j][i];
                    } /* Summing over S_t */

                    /* Normalizing */
                    for (int i = 0; i < num_states; i++)
                        cond_state_prob[j][i] /= sum;
                    for (int i = 0; i < num_states; i++)
                        log_cond_state_prob[j][i] = log(cond_state_prob[j][i]);
                } /* Summing over S_{t-1} */
                break;
            case DIST_CONDBERNOULLIG:
                /* First state probabilities */
                sum = 0.0;
                for (int i = 0; i < num_states; i++) {
                    state_prob[i] = pcount_single[i];
                    for (int n = 0; n < data.num_seqs; n++)
                        for (int t = 0; t < data.sequence[n].seq_length;
                             t++)
                            state_prob[i] += uni_prob[0][i][n][t];
                }

                sum = 0.0;
                for (int i = 0; i < num_states; i++)
                    sum += state_prob[i];

                /* Normalizing */
                for (int i = 0; i < num_states; i++)
                    state_prob[i] /= sum;
                for (int i = 0; i < num_states; i++)
                    log_state_prob[i] = log(state_prob[i]);

                /* Transition probabilities */
                for (int j = 0; j < num_states; j++) { /* Summing over S_{t-1} */
                    sum = 0.0;
                    for (int i = 0; i < num_states; i++) { /* Summing over S_t */
                        cond_state_prob[j][i] = pcount_uni[j][i];

                        for (int n = 0; n < data.num_seqs; n++)
                            for (int t = 1; t < data.sequence[n].seq_length;
                                 t++)
                                //		cond_state_prob[j][i]+=Bntij[i][j][n][t];
                                cond_state_prob[j][i] += joint_prob[i][j][n][t];

                        sum += cond_state_prob[j][i];
                    } /* Summing over S_t */

                    /* Normalizing */
                    for (int i = 0; i < num_states; i++)
                        cond_state_prob[j][i] /= sum;
                    for (int i = 0; i < num_states; i++)
                        log_cond_state_prob[j][i] = log(cond_state_prob[j][i]);
                } /* Summing over S_{t-1} */
                break;
            default:
                ;
        }

        return;
    }

    Distribution logistic_gradient(Data input,
                      double[][][] Anti,
                      double[][][][]Bntij,
                      double[][][][]log_un_prob) {
        /* Calculating the gradient vector for the logistic transition */

  /* Variables passed:
     input -- input data
     Anti -- the array of probabilities P(S_nt=i|R_n,X_n,Theta)
     Bntij -- the array of probabilities P(S_nt=i,S_n,t-1=j|R_n,X_n,Theta)
     log_un_prob -- unnormalized log probabilities of S_nt=i|S_n,t-1=j,X_nt
  */

        Distribution gradient;         // The gradient of the Q function evaluated at the current set of parameters
        double[][][][]prob;                // Current normalized probabilities of S_nt=i|S_n,t-1=j,X_nt

        /* Temporary variable(s) */
        double sum;
        double temp;
        double max;

        switch (type) {
            case DIST_TRANSLOGISTIC:
                /* Allocating the structure for the gradient */
                gradient = new Distribution(type, num_states, dim);

                /* Allocating the arrays */
                prob = new double[num_states][][][];

                for (int i = 1; i < num_states; i++) {
                    prob[i] = new double[num_states][][];
                    for (int j = 0; j < num_states; j++) {
                        prob[i][j] = new double *[input.num_seqs];
                        for (int n = 0; n < input.num_seqs; n++)
                            prob[i][j][n] = new double[input.sequence[n].seq_length];
                    }
                }

                /* Calculating P(S_nt=state|X_nt, R_n,t-1,c) */
                for (int n = 0; n < input.num_seqs; n++) {
                    /* First hidden state in the sequence */
                    max = 0.0;

                    for (int i = 1; i < num_states; i++)
                        if (max < log_un_prob[i][0][n][0])
                            max = log_un_prob[i][0][n][0];

                    sum = exp(-max);
                    for (int i = 1; i < num_states; i++)
                        /* Calculating unnormalized P(S_n1=i|X_n1) */
                        sum += exp(log_un_prob[i][0][n][0] - max);

                    /* Calculating normalized P(S_n1=state|X_n1) */
                    for (int i = 1; i < num_states; i++)
                        prob[i][0][n][0] = exp(log_un_prob[i][0][n][0] - max) / sum;

                    for (int t = 1; t < input.sequence[n].seq_length;
                         t++)
                        /* Not the first state in the sequence */
                        for (int j = 0; j < num_states; j++) {
                            max = 0.0;
                            for (int i = 1; i < num_states; i++)
                                if (max < log_un_prob[i][j][n][t])
                                    max = log_un_prob[i][j][n][t];

                            sum = exp(-max);
                            for (int i = 1; i < num_states; i++)
                                /* Calculating unnormalized P(S_nt=i|S_n,t-1=j,X_nt) */
                                sum += exp(log_un_prob[i][j][n][t] - max);

                            /* Calculating normalized P(S_nt=state|S_n,t-1=j,X_nt) */
                            for (int i = 1; i < num_states; i++)
                                prob[i][j][n][t] = exp(log_un_prob[i][j][n][t] - max) / sum;
                        }
                }

                /* Calculating the values of the gradient */
                for (int i = 1; i < num_states; i++) { /* For each state */
                    /* State 1 is ignored since its parameters are never updated */
                    for (int n = 0; n < input.num_seqs; n++)
                        for (int t = 0; t < input.sequence[n].seq_length;
                             t++) {
                            if (t == 0) {
                                gradient.lambda[i] += (Anti[i][n][t] - prob[i][0][n][t]);
                                for (int d = 0; d < dim; d++)
                                    gradient.rho[i][d] += (Anti[i][n][t] - prob[i][0][n][t]) * input.sequence[n]->
                                entry[t].rdata[d];
                            } else {
                                temp = 0.0;
                                for (int j = 0; j < num_states; j++) {
                                    gradient.sigma[j][i] += (Bntij[i][j][n][t] - Anti[j][n][t - 1] * prob[i][j][n][t]);
                                    temp += Anti[j][n][t - 1] * prob[i][j][n][t];
                                }
                                for (int d = 0; d < dim; d++)
                                    gradient.rho[i][d] += (Anti[i][n][t] - temp) * input.sequence[n].entry[t]->
                                rdata[d];
                            }
                        }
                } /* For each state */

                /* Deallocating the arrays */
                for (int i = 1; i < num_states; i++) {
                    for (int j = 0; j < num_states; j++) {
                        for (int n = 0; n < input.num_seqs; n++)
                            prob[i][j] = null[n];
                        prob[i][j] = null;
                    }
                    prob[i] = null;
                }
                prob = null;

                break;
            case DIST_TRANSLOGISTICG:
                /* Allocating the structure for the gradient */
                gradient = new Distribution(type, num_states, dim);

                /* Allocating the arrays */
                prob = new double[num_states][][][];

                for (int i = 1; i < num_states; i++) {
                    prob[i] = new double[num_states][][];
                    for (int j = 0; j < num_states; j++) {
                        prob[i][j] = new double *[input.num_seqs];
                        for (int n = 0; n < input.num_seqs; n++)
                            prob[i][j][n] = new double[input.sequence[n].seq_length];
                    }
                }

                /* Calculating P(S_nt=state|X_nt, R_n,t-1,c) */
                for (int n = 0; n < input.num_seqs; n++) {
                    /* First hidden state in the sequence */
                    max = 0.0;

                    for (int i = 1; i < num_states; i++)
                        if (max < log_un_prob[i][0][n][0])
                            max = log_un_prob[i][0][n][0];

                    sum = exp(-max);
                    for (int i = 1; i < num_states; i++)
                        /* Calculating unnormalized P(S_n1=i|X_n1) */
                        sum += exp(log_un_prob[i][0][n][0] - max);

                    /* Calculating normalized P(S_n1=state|X_n1) */
                    for (int i = 1; i < num_states; i++)
                        prob[i][0][n][0] = exp(log_un_prob[i][0][n][0] - max) / sum;

                    for (int t = 1; t < input.sequence[n].seq_length;
                         t++)
                        /* Not the first state in the sequence */
                        for (int j = 0; j < num_states; j++) {
                            max = 0.0;
                            for (int i = 1; i < num_states; i++)
                                if (max < log_un_prob[i][j][n][t])
                                    max = log_un_prob[i][j][n][t];

                            sum = exp(-max);
                            for (int i = 1; i < num_states; i++)
                                /* Calculating unnormalized P(S_nt=i|S_n,t-1=j,X_nt) */
                                sum += exp(log_un_prob[i][j][n][t] - max);

                            /* Calculating normalized P(S_nt=state|S_n,t-1=j,X_nt) */
                            for (int i = 1; i < num_states; i++)
                                prob[i][j][n][t] = exp(log_un_prob[i][j][n][t] - max) / sum;
                        }
                }

                /* Calculating the values of the gradient */
                for (int i = 1; i < num_states; i++) { /* For each state */
                    /* State 1 is ignored since its parameters are never updated */
                    for (int n = 0; n < input.num_seqs; n++)
                        for (int t = 0; t < input.sequence[n].seq_length;
                             t++) {
                            if (t == 0) {
                                gradient.lambda[i] += (Anti[i][n][t] - prob[i][0][n][t]);
                                for (int d = 0; d < dim; d++)
                                    gradient.rho[i][d] += (Anti[i][n][t] - prob[i][0][n][t]) * input.sequence[n]->
                                entry[t].rdata[d];
                            } else {
                                temp = 0.0;
                                for (int j = 0; j < num_states; j++) {
                                    gradient.sigma[j][i] += (Bntij[i][j][n][t] - Anti[j][n][t - 1] * prob[i][j][n][t]);
                                    temp += Anti[j][n][t - 1] * prob[i][j][n][t];
                                }
                                for (int d = 0; d < dim; d++)
                                    gradient.rho[i][d] += (Anti[i][n][t] - temp) * input.sequence[n].entry[t]->
                                rdata[d];
                            }
                        }
                } /* For each state */

                /* Deallocating the arrays */
                for (int i = 1; i < num_states; i++) {
                    for (int j = 0; j < num_states; j++) {
                        for (int n = 0; n < input.num_seqs; n++)
                            prob[i][j] = null[n];
                        prob[i][j] = null;
                    }
                    prob[i] = null;
                }
                prob = null;
                break;
            case DIST_LOGISTIC:
                /* Allocating the structure for the gradient */
                gradient = new Distribution(type, num_states, dim);

                /* Allocating the arrays */
                prob = new double[num_states][][][];

                for (int i = 1; i < num_states; i++) {
                    prob[i] = new double[num_states][][];
                    prob[i][0] = new double *[input.num_seqs];
                    for (int n = 0; n < input.num_seqs; n++)
                        prob[i][0][n] = new double[input.sequence[n].seq_length];
                }

                /* Calculating P(S_nt=state|X_nt, R_n,t-1,c) */
                for (int n = 0; n < input.num_seqs; n++)
                    for (int t = 0; t < input.sequence[n].seq_length;
                         t++)
                        /* Not the first state in the sequence */ {
                        max = 0.0;

                        for (int i = 1; i < num_states; i++)
                            if (max < log_un_prob[i][0][n][t])
                                max = log_un_prob[i][0][n][t];

                        sum = exp(-max);
                        for (int i = 1; i < num_states; i++)
                            /* Calculating unnormalized P(S_nt=i|X_nt) */
                            sum += exp(log_un_prob[i][0][n][t] - max);

                        /* Calculating normalized P(S_nt=state|X_nt) */
                        for (int i = 1; i < num_states; i++)
                            prob[i][0][n][t] = exp(log_un_prob[i][0][n][t] - max) / sum;
                    }

                /* Calculating the values of the gradient */
                for (int i = 1; i < num_states; i++) { /* For each state */
                    /* State 1 is ignored since its parameters are never updated */
                    for (int n = 0; n < input.num_seqs; n++)
                        for (int t = 0; t < input.sequence[n].seq_length;
                             t++) {
                            gradient.lambda[i] += (Anti[i][n][t] - prob[i][0][n][t]);
                            for (int d = 0; d < dim; d++)
                                gradient.rho[i][d] += (Anti[i][n][t] - prob[i][0][n][t]) * input.sequence[n].entry[t]->
                            rdata[d];
                        }
                } /* For each state */

                /* Deallocating the arrays */
                for (int i = 1; i < num_states; i++) {
                    for (int n = 0; n < input.num_seqs; n++)
                        prob[i][0][n] = null;
                    prob[i][0] = null;
                    prob[i] = null;
                }
                prob = null;

                break;
            default:
                ;
        }

        return (gradient);
    }

    void  linear_search_logistic_NR(Data input,
                              double[][][]Anti,
                              double[][][][]Bntij,
                              double[][][][]log_un_prob,
                              Distribution delta,
                              int iter) {
        /* Line search method (utilizing Newton Raphson) to find the maximum of the function */

  /* Variables passed:
     input -- input data
     Anti -- the array of probabilities P(S_nt=i|R_n,X_n,Theta)
     Bntij -- the array of probabilities P(S_nt=i,S_n,t-1=j|R_n,X_n,Theta)
     log_un_prob -- unnormalized log probabilities of S_nt=i|S_n,t-1=j,X_nt
     delta -- the direction along which the search is performed
     iter -- current iteration of the EM algorithm (if needed)
  */

        double[][][][]adj_fact;            // The adjustment factor (sigma+rho*x) for the direction

        double[][][][]prob;                // Current normalized probabilities of S_nt=i|S_n,t-1=j,X_nt

        double f;                       // Current value of the function (partial derivative of the log-likelihood)
        double f_prev;                  // Previous value of f

        double fprime = 0.0;              // Derivative (second partial derivative of the log-likelihood)
        double factor;                  // f/fprime

        double gamma = 0.0;               // Value of the scalar to be optimized
        double gamma_prev;              // Previous value of gamma

        boolean found_positive = false;
        double gamma_positive;          // Value of gamma corresponding to the positive f
        double fneg;
        boolean found_negative = false;
        double gamma_negative;          // Value of gamma corresponding to the negative f
        double fpos;

        int iteration;                 // Number of iterations

        double epsilon;                 // Sensitivity threshold

        /* Temporary variable(s) */
        double sum;
        double max;

        if (!(type == DIST_LOGISTIC || type == DIST_TRANSLOGISTIC)) { /* !!! !!! */
            System.exit(-1);
        } /* !!! !!! */

        /* Allocating the arrays */
        adj_fact = new double[num_states][][][];
        for (int i = 1; i < num_states; i++) {
            adj_fact[i] = new double[num_states][][];
            switch (type) {
                case DIST_TRANSLOGISTIC:
                    for (int j = 0; j < num_states; j++) {
                        adj_fact[i][j] = new double *[input.num_seqs];
                        for (int n = 0; n < input.num_seqs; n++)
                            adj_fact[i][j][n] = new double[input.sequence[n].seq_length];
                    }
                    break;
                case DIST_LOGISTIC:
                    adj_fact[i][0] = new double *[input.num_seqs];
                    for (int n = 0; n < input.num_seqs; n++)
                        adj_fact[i][0][n] = new double[input.sequence[n].seq_length];
                    break;
                default:
                    ;
            }
        }

        prob = new double[num_states][][][];
        for (int i = 1; i < num_states; i++) {
            prob[i] = new double[num_states][][];
            switch (type) {
                case DIST_TRANSLOGISTIC:
                    for (int j = 0; j < num_states; j++) {
                        prob[i][j] = new double *[input.num_seqs];
                        for (int n = 0; n < input.num_seqs; n++)
                            prob[i][j][n] = new double[input.sequence[n].seq_length];
                    }
                    break;
                case DIST_LOGISTIC:
                    prob[i][0] = new double *[input.num_seqs];
                    for (int n = 0; n < input.num_seqs; n++)
                        prob[i][0][n] = new double[input.sequence[n].seq_length];
                    break;
                default:
                    ;
            }
        }

        /* Precalculating the adjustment values */
        for (int i = 1; i < num_states; i++)
            for (int n = 0; n < input.num_seqs; n++)
                switch (type) {
                    case DIST_TRANSLOGISTIC:
                        adj_fact[i][0][n][0] = delta.log_un_probLogistic(input.sequence[n].entry[0], i, -1);
                        for (int t = 1; t < input.sequence[n].seq_length;
                             t++)
                            for (int j = 0; j < num_states; j++)
                                adj_fact[i][j][n][t] = delta.log_un_probLogistic(input.sequence[n].entry[t], i, j);
                        break;
                    case DIST_LOGISTIC:
                        for (int t = 0; t < input.sequence[n].seq_length;
                             t++)
                            adj_fact[i][0][n][t] = delta.log_un_probLogistic(input.sequence[n].entry[t], i, -1);
                        break;
                    default:
                        ;
                }

        /* Calculating P(S_nt=state|X_nt, R_n,t-1,c) */
        for (int n = 0; n < input.num_seqs; n++)
            switch (type) {
                case DIST_TRANSLOGISTIC:
                    /* First state in the sequence */
                    max = 0.0;
                    for (int i = 1; i < num_states; i++)
                        if (max < log_un_prob[i][0][n][0])
                            max = log_un_prob[i][0][n][0];

                    sum = exp(-max);
                    for (int i = 1; i < num_states; i++)
                        /* Calculating unnormalized P(S_n1=i|X_n1) */
                        sum += exp(log_un_prob[i][0][n][0] - max);

                    /* Calculating normalized P(S_n1=state|X_n1) */
                    for (int i = 1; i < num_states; i++)
                        prob[i][0][n][0] = exp(log_un_prob[i][0][n][0] - max) / sum;

                    for (int t = 1; t < input.sequence[n].seq_length;
                         t++)
                        /* Not the first state in the sequence */
                        for (int j = 0; j < num_states; j++) {
                            max = 0.0;
                            for (int i = 1; i < num_states; i++)
                                if (max < log_un_prob[i][j][n][t])
                                    max = log_un_prob[i][j][n][t];

                            sum = exp(-max);
                            for (int i = 1; i < num_states; i++)
                                /* Calculating unnormalized P(S_nt=i|S_n,t-1=j,X_nt) */
                                sum += exp(log_un_prob[i][j][n][t] - max);

                            /* Calculating normalized P(S_nt=state|S_n,t-1=j,X_nt) */
                            for (int i = 1; i < num_states; i++)
                                prob[i][j][n][t] = exp(log_un_prob[i][j][n][t] - max) / sum;
                        }
                    break;
                case DIST_LOGISTIC:
                    for (int t = 0; t < input.sequence[n].seq_length;
                         t++) {
                        max = 0.0;
                        for (int i = 1; i < num_states; i++)
                            if (max < log_un_prob[i][0][n][t])
                                max = log_un_prob[i][0][n][t];

                        sum = exp(-max);
                        for (int i = 1; i < num_states; i++)
                            sum += exp(log_un_prob[i][0][n][t] - max);

                        for (int i = 1; i < num_states; i++)
                            prob[i][0][n][t] = exp(log_un_prob[i][0][n][t] - max) / sum;
                    }
                    break;
                default:
                    ;
            }

        /* Calculating initial value of the function (gamma=0) */
        f = 0.0;
        for (int i = 1; i < num_states; i++)
            for (int n = 0; n < input.num_seqs; n++) { /* For each sequence */

                /* First state */
                f += adj_fact[i][0][n][0] * (Anti[i][n][0] - prob[i][0][n][0]);

                for (int t = 1; t < input.sequence[n].seq_length;
                     t++)
                    if (type == DIST_LOGISTIC)
                        f += adj_fact[i][0][n][t] * (Anti[i][n][t] - prob[i][0][n][t]);
                    else
                        for (int j = 0; j < num_states; j++)
                            f += adj_fact[i][j][n][t] * (Bntij[i][j][n][t] - Anti[j][n][t - 1] * prob[i][j][n][t]);
            } /* For each sequence */

        if (!found_positive && f > 0.0) {
            gamma_positive = gamma;
            fpos = f;
            found_positive = 1;
        } else if (!found_negative && f < 0.0) {
            gamma_negative = gamma;
            fneg = f;
            found_negative = 1;
        }

        if (NEWTONRAPHSON_VERBOSE) {
            fprintf(stdout, "Initial gamma=%.12f, f(gamma)=%.12f\n", gamma, f);
        }

        epsilon = MIN_NR_EPSILON / (double) iter;

        iteration = 0;
        while (abs(f) > epsilon && iteration < MAX_NR_ITERATIONS) {
            /* Calculating the second derivative fprime */
            fprime = 0.0;
            for (int i = 1; i < num_states; i++)
                for (int n = 0; n < input.num_seqs; n++) {
                    fprime -= adj_fact[i][0][n][0] * adj_fact[i][0][n][0] * prob[i][0][n][0] * (1.0 - prob[i][0][n][0]);
                    for (int t = 1; t < input.sequence[n].seq_length;
                         t++)
                        if (type == DIST_LOGISTIC)
                            fprime -= adj_fact[i][0][n][t] * adj_fact[i][0][n][t] * prob[i][0][n][t] * (1.0 - prob[i][0][n][t]);
                        else
                            for (int j = 0; j < num_states; j++)
                                fprime -= Anti[j][n][t - 1] * adj_fact[i][j][n][t] * adj_fact[i][j][n][t] * prob[i][j][n][t] * (1.0 - prob[i][j][n][t]);
                }

            if (abs(fprime) < NR_EPSILON) {
                //	  System.err.format( "Newton-Raphson is stuck!  Exiting iterations\n" );
                iteration = MAX_NR_ITERATIONS;
            }

            /* Storing the previous values */
            f_prev = f;
            gamma_prev = gamma;

            /* Updating the value in question */
            factor = f / fprime;
            gamma -= factor;

            /* Updating the unnormalized probabilities */
            for (int i = 1; i < num_states; i++)
                for (int n = 0; n < input.num_seqs; n++) {
                    log_un_prob[i][0][n][0] -= factor * adj_fact[i][0][n][0];
                    for (int t = 1; t < input.sequence[n].seq_length;
                         t++)
                        if (type == DIST_LOGISTIC)
                            log_un_prob[i][0][n][t] -= factor * adj_fact[i][0][n][t];
                        else
                            for (int j = 0; j < num_states; j++)
                                log_un_prob[i][j][n][t] -= factor * adj_fact[i][j][n][t];
                }

            /* Calculating normalized probabilities */
            for (int n = 0; n < input.num_seqs; n++) {
                /* First state in the sequence */
                max = 0.0;
                for (int i = 1; i < num_states; i++)
                    if (max < log_un_prob[i][0][n][0])
                        max = log_un_prob[i][0][n][0];

                sum = exp(-max);
                for (int i = 1; i < num_states; i++)
                    /* Calculating unnormalized P(S_n1=i|X_n1) */
                    sum += exp(log_un_prob[i][0][n][0] - max);

                /* Calculating normalized P(S_n1=state|X_n1) */
                for (int i = 1; i < num_states; i++)
                    prob[i][0][n][0] = exp(log_un_prob[i][0][n][0] - max) / sum;

                for (int t = 1; t < input.sequence[n].seq_length;
                     t++)
                    /* Not the first state in the sequence */
                    if (type == DIST_LOGISTIC) {
                        max = 0.0;
                        for (int i = 1; i < num_states; i++)
                            if (max < log_un_prob[i][0][n][t])
                                max = log_un_prob[i][0][n][t];

                        sum = exp(-max);
                        for (int i = 1; i < num_states; i++)
                            sum += exp(log_un_prob[i][0][n][t] - max);

                        for (int i = 1; i < num_states; i++)
                            prob[i][0][n][t] = exp(log_un_prob[i][0][n][t] - max) / sum;
                    } else
                        for (int j = 0; j < num_states; j++) {
                            max = 0.0;
                            for (int i = 1; i < num_states; i++)
                                if (max < log_un_prob[i][j][n][t])
                                    max = log_un_prob[i][j][n][t];

                            sum = exp(-max);
                            for (int i = 1; i < num_states; i++)
                                /* Calculating unnormalized P(S_nt=i|S_n,t-1=j,X_nt) */
                                sum += exp(log_un_prob[i][j][n][t] - max);

                            /* Calculating normalized P(S_nt=state|S_n,t-1=j,X_nt) */
                            for (int i = 1; i < num_states; i++)
                                prob[i][j][n][t] = exp(log_un_prob[i][j][n][t] - max) / sum;
                        }
            }

            /* Calculating the value of the function */
            f = 0.0;
            for (int i = 1; i < num_states; i++)
                for (int n = 0; n < input.num_seqs; n++) { /* For each sequence */

                    /* First state */
                    f += adj_fact[i][0][n][0] * (Anti[i][n][0] - prob[i][0][n][0]);

                    for (int t = 1; t < input.sequence[n].seq_length;
                         t++)
                        if (type == DIST_LOGISTIC)
                            f += adj_fact[i][0][n][t] * (Anti[i][n][t] - prob[i][0][n][t]);
                        else
                            for (int j = 0; j < num_states; j++)
                                f += adj_fact[i][j][n][t] * (Bntij[i][j][n][t] - Anti[j][n][t - 1] * prob[i][j][n][t]);
                } /* For each sequence */

            if (f > 0.0) {
                if (found_positive == null) {
                    gamma_positive = gamma;
                    fpos = f;
                    found_positive = 1;
                } else if (f < fpos) {
                    gamma_positive = gamma;
                    fpos = f;
                }
            } else
                /* f<0.0 */
                if (found_negative == null) {
                    gamma_negative = gamma;
                    fneg = f;
                    found_negative = 1;
                } else if (f > fneg) {
                    gamma_negative = gamma;
                    fneg = f;
                }

            iteration++;

            if (NEWTONRAPHSON_VERBOSE) {
                fprintf(stdout, "Iteration %d: new value=%.12f, f(gamma)=%.12f; previous f'(gamma)=%.8f; factor=%.8f\n",
                        iteration, gamma, f, fprime, factor);
            }

            if ((!finite(f) || (abs(f) > abs(f_prev))) && iteration < MAX_NR_ITERATIONS) { /* Diverging */
                if (NEWTONRAPHSON_VERBOSE) {
                    fprintf(stdout, "Absolute value of f is increasing -- needs adjustment\n");
                }

                /* Dividing factor until reach a better point */
                do {
                    factor /= 2.0;
                    gamma = gamma_prev - factor;
                    f = logistic_derivative_along_vector(input, Anti, Bntij, delta, gamma);
                    if (NEWTONRAPHSON_VERBOSE) {
                        fprintf(stdout, "Trying factor %.12f; new f=%.12f\n", factor, f);
                    }
                }
                while ((!finite(f) && finite(factor)) || (abs(f) > abs(f_prev) && abs(f - f_prev) > COMP_EPSILON));

                if (!finite(f) || abs(f - f_prev) <= COMP_EPSILON) { /* Resetting to previous value of gamma and exiting */
                    gamma = gamma_prev;
                    iteration = MAX_NR_ITERATIONS;
                    if (NEWTONRAPHSON_VERBOSE) {
                        fprintf(stdout, "Unable to improve on f\n");
                    }
                } /* Resetting to previous value of gamma and exiting */ else {
                    if (NEWTONRAPHSON_VERBOSE) {
                        fprintf(stdout, "Updated value=%.12f, f(gamma)=%.12f, factor=%.8f\n",
                                gamma, f, factor);
                    }
                }

                /* Updating log_un_prob */
                for (int i = 1; i < num_states; i++)
                    for (int n = 0; n < input.num_seqs; n++) {
                        log_un_prob[i][0][n][0] = log_un_probLogistic(input.sequence[n].entry[0], i, -1) +
                                gamma * delta.log_un_probLogistic(input.sequence[n].entry[0], i, -1);
                        for (int t = 1; t < input.sequence[n].seq_length;
                             t++)
                            if (type == DIST_LOGISTIC) {
                                log_un_prob[i][0][n][t] = log_un_probLogistic(input.sequence[n].entry[t], i, -1) +
                                        gamma * delta.log_un_probLogistic(input.sequence[n].entry[t], i, -1);
                            } else
                                for (int j = 0; j < num_states; j++)
                                    log_un_prob[i][j][n][t] = log_un_probLogistic(input.sequence[n].entry[t], i, j) +
                                            gamma * delta.log_un_probLogistic(input.sequence[n].entry[t], i, j);

                    }

            } /* Diverging */
        }

        if (iteration >= MAX_NR_ITERATIONS) { /* Newton-Raphson failed */

            if (found_negative && found_positive) { /* Finding a zero by halving the interval */
                if (NEWTONRAPHSON_VERBOSE) {
                    fprintf(stdout, "Newton-Raphson failed; running halving algorithm\n");
                }

                linear_search_logistic_halving(input, Anti, Bntij, log_un_prob,
                        delta, iter, gamma_negative, gamma_positive);
            } /* Finding a zero by halving the interval */ else
                /* Recalculating the original log-unnormalized probabilities */
                for (int i = 1; i < num_states; i++)
                    for (int n = 0; n < input.num_seqs; n++) {
                        log_un_prob[i][0][n][0] = log_un_probLogistic(input.sequence[n].entry[0], i, -1);
                        for (int t = 1; t < input.sequence[n].seq_length;
                             t++)
                            if (type == DIST_LOGISTIC)
                                log_un_prob[i][0][n][t] = log_un_probLogistic(input.sequence[n].entry[t], i, -1);
                            else
                                for (int j = 0; j < num_states; j++)
                                    log_un_prob[i][j][n][t] = log_un_probLogistic(input.sequence[n].entry[t], i, j);
                    }
        } /* Newton-Raphson failed */ else { /* Run finished fine */
            /* Updating the set of parameters by gamma*delta */
            if (fprime > 0.0) {
                System.err.print("f'=%.12f: Newton-Raphson found minimum instead of a maximum!\n", fprime);
            }

            for (int i = 1; i < num_states; i++) {
                lambda[i] += gamma * delta.lambda[i];
                if (type == DIST_TRANSLOGISTIC)
                    for (int j = 0; j < num_states; j++)
                        sigma[j][i] += gamma * delta.sigma[j][i];
                for (int j = 0; j < dim; j++)
                    rho[i][j] += gamma * delta.rho[i][j];
            }
        } /* Run finished fine */

        /* Deallocating the arrays */
        for (int i = 1; i < num_states; i++) {
            if (type == DIST_LOGISTIC) {
                for (int n = 0; n < input.num_seqs; n++)
                    adj_fact[i][0][n] = null;
                adj_fact[i][0] = null;
            } else
                for (int j = 0; j < num_states; j++) {
                    for (int n = 0; n < input.num_seqs; n++)
                        adj_fact[i][j] = null[n];
                    adj_fact[i][j] = null;
                }
            adj_fact[i] = null;
        }
        adj_fact = null;

        for (int i = 1; i < num_states; i++) {
            if (type == DIST_LOGISTIC) {
                for (int n = 0; n < input.num_seqs; n++)
                    prob[i][0][n] = null;
                prob[i][0] = null;
            } else
                for (int j = 0; j < num_states; j++) {
                    for (int n = 0; n < input.num_seqs; n++)
                        prob[i][j] = null[n];
                    prob[i][j] = null;
                }
            prob[i] = null;
        }
        prob = null;

        return;
    }

    void  linear_search_logistic_halving(Data input,
                                   double[][][]Anti,
                                   double[][][][]Bntij,
                                   double[][][][]log_un_prob,
                                   Distribution delta,
                                   int iter,
                                   double gamma_negative,
                                   double gamma_positive) {
        /* Line search method (utilizing Newton Raphson) to find the maximum of the function */

  /* Variables passed:
     input -- input data
     Anti -- the array of probabilities P(S_nt=i|R_n,X_n,Theta)
     Bntij -- the array of probabilities P(S_nt=i,S_n,t-1=j|R_n,X_n,Theta)
     log_un_prob -- unnormalized log probabilities of S_nt=i|S_n,t-1=j,X_nt
     delta -- the direction along which the search is performed
     iter -- current iteration of the EM algorithm (if needed)
     gamma_negative -- value of gamma corresponding to the negative value of f
     gamma_positive -- value of gamma corresponding to the positive value of f
  */

        double f;                       // Current value of the function (partial derivative of the log-likelihood)
        double gamma;                   // Value of the scalar to be optimized
        double epsilon;                 // Sensitivity threshold

        if (!(type == DIST_TRANSLOGISTIC || type == DIST_LOGISTIC)) { /* !!! !!! */
            System.exit(-1);
        } /* !!! !!! */

        epsilon = MIN_NR_EPSILON / (double) iter;

        if (NEWTONRAPHSON_VERBOSE) {
            f = logistic_derivative_along_vector(input, Anti, Bntij, delta, gamma_negative);
            fprintf(stdout, "Negative gamma=%.12f; f=%.12f\n", gamma_negative, f);
            f = logistic_derivative_along_vector(input, Anti, Bntij, delta, gamma_positive);
            fprintf(stdout, "Positive gamma=%.12f; f=%.12f\n", gamma_positive, f);
        }

        do {
            gamma = 0.5 * (gamma_negative + gamma_positive);
            f = logistic_derivative_along_vector(input, Anti, Bntij, delta, gamma);

            if (NEWTONRAPHSON_VERBOSE) {
                fprintf(stdout, "Current gamma=%.12f; current f=%.12f\n", gamma, f);
            }
            if (f < 0.0)
                gamma_negative = gamma;
            else
                gamma_positive = gamma;
        }
        while (abs(f) > epsilon && abs(gamma_negative - gamma_positive) > COMP_EPSILON);

        /* Updating the set of parameters */
        for (int i = 1; i < num_states; i++) {
            lambda[i] += gamma * delta.lambda[i];
            if (type == DIST_TRANSLOGISTIC)
                for (int j = 0; j < num_states; j++)
                    sigma[j][i] += gamma * delta.sigma[j][i];
            for (int j = 0; j < dim; j++)
                rho[i][j] += gamma * delta.rho[i][j];
        }

        /* Recalculating log-unnormalized probabilities */
        for (int i = 1; i < num_states; i++)
            for (int n = 0; n < input.num_seqs; n++) {
                log_un_prob[i][0][n][0] = log_un_probLogistic(input.sequence[n].entry[0], i, -1);
                for (int t = 1; t < input.sequence[n].seq_length;
                     t++)
                    if (type == DIST_LOGISTIC)
                        log_un_prob[i][0][n][t] = log_un_probLogistic(input.sequence[n].entry[t], i, -1);
                    else
                        for (int j = 0; j < num_states; j++)
                            log_un_prob[i][j][n][t] = log_un_probLogistic(input.sequence[n].entry[t], i, j);
            }

        return;
    }

    double logistic_derivative_along_vector(Data input,
                                     double[][][]Anti,
                                     double[][][][]Bntij,
                                     Distribution delta,
                                     double gamma) {
        double f;

        double[]prob;
        double[]adj_fact;

        prob = new double[num_states];
        adj_fact = new double[num_states];


        f = 0.0;
        for (int n = 0; n < input.num_seqs; n++) {
            /* First entry */
            /* Calculating adjustments */
            for (int i = 1; i < num_states; i++)
                adj_fact[i] = delta.log_un_probLogistic(input.sequence[n].entry[0], i, -1);

            /* Calculating probabilities P(Sn1=i|X_n1,new Theta) */
            for (int i = 1; i < num_states; i++)
                prob[i] = log_un_probLogistic(input.sequence[n].entry[0], i, -1) + gamma * adj_fact[i];

            normalize_logistic(prob, prob, num_states, 1);

            for (int i = 1; i < num_states; i++)
                f += adj_fact[i] * (Anti[i][n][0] - prob[i]);

            /* Rest of the entries */
            for (int t = 1; t < input.sequence[n].seq_length;
                 t++)
                if (type == DIST_LOGISTIC) {
                    for (int i = 1; i < num_states; i++) {
                        adj_fact[i] = delta.log_un_probLogistic(input.sequence[n].entry[t], i, -1);
                        prob[i] = log_un_probLogistic(input.sequence[n].entry[t], i, -1) + gamma * adj_fact[i];
                    }

                    normalize_logistic(prob, prob, num_states, 1);

                    for (int i = 1; i < num_states; i++)
                        f += adj_fact[i] * (Anti[i][n][t] - prob[i]);
                } else
                    for (int j = 0; j < num_states; j++) {
                        /* Calculating adjustments */
                        for (int i = 1; i < num_states; i++)
                            adj_fact[i] = delta.log_un_probLogistic(input.sequence[n].entry[t], i, j);

                        /* Calculating probabilities P(Snt=i|S_n,t-1=j, X_nt,new Theta) */
                        for (int i = 1; i < num_states; i++)
                            prob[i] = log_un_probLogistic(input.sequence[n].entry[t], i, j) + gamma * adj_fact[i];

                        normalize_logistic(prob, prob, num_states, 1);

                        for (int i = 1; i < num_states; i++)
                            f += adj_fact[i] * (Bntij[i][j][n][t] - Anti[j][n][t - 1] * prob[i]);
                    }
        }

        prob = null;
        adj_fact = null;

        return (f);
    }


    int  compute_envelopes(DataPoint v, double[][]mult) {
  /* Given a forest structure and a mask indicating missing entries,
     partitions the nodes corresponding to missing observations into maximal
     trees (envelopes) */

        int num_envelopes = 0;

        int[]node_index;

        /* Initializing */
        node_index = new int[dim];
        for (int i = 0; i < dim; i++)
            node_index[i] = -1;

        for (int i = 0; i < dim; i++)
            for (int i1 = 0; i1 < num_states; i1++)
                mult[i][i1] = 1.0;

        for (int e = 0; e < num_edges; e++) { /* For each of the edges */
            /* Shortcuts */
           int  i1 = edge[e][0];
          int   i2 = edge[e][1];

            if (!is_missing(v.ddata[dim_index[i1]])) { /* First variable is instantiated */
                if (!is_missing(v.ddata[dim_index[i2]])) { /* Both variables are instantiated */
                    if (node_index[i1] != -1) { /* First variable belongs to an envelope -- adding the second variable to the same envelope */
                        envelope[node_index[i1]].edge[envelope[node_index[i1]].num_edges] = e;
                        envelope[node_index[i1]].num_edges++;
                        envelope[node_index[i1]].node[envelope[node_index[i1]].num_nodes] = i2;
                        envelope[node_index[i1]].num_nodes++;
                        node_index[i2] = node_index[i1];
                    } /* First variable belongs to an envelope -- adding the second variable to the same envelope */ else if (node_index[i2] != -1) { /* Second variables belongs to an envelope -- adding the first variables to the same envelope */
                        envelope[node_index[i2]].edge[envelope[node_index[i2]].num_edges] = e;
                        envelope[node_index[i2]].num_edges++;
                        envelope[node_index[i2]].node[envelope[node_index[i2]].num_nodes] = i1;
                        envelope[node_index[i2]].num_nodes++;
                        node_index[i1] = node_index[i2];
                    } /* Second variables belongs to an envelope -- ading the first variables to the same envelope */ else { /* Neither variable is in an envelope -- creating a new envelope */
                        envelope[num_envelopes] = new Envelope(dim);
                        envelope[num_envelopes].is_missing = 0;
                        envelope[num_envelopes].num_nodes = 2;
                        envelope[num_envelopes].num_edges = 1;
                        envelope[num_envelopes].edge[0] = e;
                        envelope[num_envelopes].node[0] = edge[e][0];
                        envelope[num_envelopes].node[1] = edge[e][1];
                        node_index[i1] = num_envelopes;
                        node_index[i2] = num_envelopes;
                        num_envelopes++;
                    } /* Neither variable is in an envelope -- creating a new envelope */
                } /* Both variables are instantiated */ else { /* First variable is instantiated; the second one is not. */
                    if (node_index[i2] != -1) { /* Missing variable already belongs to an envelope. */
                        /* Updating the multiplication factor for node i2 */
                        for (int i = 0; i < num_states; i++)
                            mult[i2][i] *= edge_prob[e][v.ddata[dim_index[i1]]][i] / (ind_prob[i1][v.ddata[dim_index[i1]]] * ind_prob[i2][i]);
                    } /* Missing variable already belongs to an envelope; adding the edge to the same envelope. */ else { /* Creating an envelope for the uninstantiated variable */
                        envelope[num_envelopes] = new Envelope(dim);
                        envelope[num_envelopes].is_missing = 1;
                        envelope[num_envelopes].num_nodes = 1;
                        envelope[num_envelopes].num_edges = 0;
                        envelope[num_envelopes].node[0] = i2;
                        /* Updating the multiplication factor for node i2 */
                        for (int i = 0; i < num_states; i++)
                            mult[i2][i] *= edge_prob[e][v.ddata[dim_index[i1]]][i] / (ind_prob[i1][v.ddata[dim_index[i1]]] * ind_prob[i2][i]);
                        node_index[i2] = num_envelopes;
                        num_envelopes++;
                    } /* Creating an envelope for the uninstantiated variable */
                } /* First variable is instantiated; the second one is not. */
            } /* First variable is instantiated */ else { /* First variable is missing */
                if (!is_missing(v.ddata[dim_index[i2]])) { /* First variable is missing; the second one is present. */
                    if (node_index[i1] != -1) { /* Missing variable already belongs to an envelope. */
                        /* Updating the multiplication factor for node i1 */
                        for (int i = 0; i < num_states; i++)
                            mult[i1][i] *= edge_prob[e][i][v.ddata[dim_index[i2]]] / (ind_prob[i1][i] * ind_prob[i2][v.ddata[dim_index[i2]]]);
                    } /* Missing variable already belongs to an envelope.  Adding the edge to that envelope. */ else { /* Creating an envelope for the missing variable */
                        envelope[num_envelopes] = new Envelope(dim);
                        envelope[num_envelopes].is_missing = 1;
                        envelope[num_envelopes].num_nodes = 1;
                        envelope[num_envelopes].num_edges = 0;
                        envelope[num_envelopes].node[0] = i1;
                        /* Updating the multiplication factor for node i1 */
                        for (int i = 0; i < num_states; i++)
                            mult[i1][i] *= edge_prob[e][i][v.ddata[dim_index[i2]]] / (ind_prob[i1][i] * ind_prob[i2][v.ddata[dim_index[i2]]]);
                        node_index[i1] = num_envelopes;
                        num_envelopes++;
                    } /* Creating an envelope for the missing variable */
                } /* First variable is missing; the second one is present. */ else { /* Both variables are missing */
                    if (node_index[i1] != -1) { /* First variable belongs to an envelope; adding the second variable to the same envelope. */
                        envelope[node_index[i1]].edge[envelope[node_index[i1]].num_edges] = e;
                        envelope[node_index[i1]].num_edges++;
                        envelope[node_index[i1]].node[envelope[node_index[i1]].num_nodes] = i2;
                        envelope[node_index[i1]].num_nodes++;
                        node_index[i2] = node_index[i1];
                    } /* First variable belongs to an envelope; adding the second variable to the same envelope. */ else if (node_index[i2] != -1) { /* Second variable belongs to an envelope; adding the first variable to the same envelope. */
                        envelope[node_index[i2]].edge[envelope[node_index[i2]].num_edges] = e;
                        envelope[node_index[i2]].num_edges++;
                        envelope[node_index[i2]].node[envelope[node_index[i2]].num_nodes] = i1;
                        envelope[node_index[i2]].num_nodes++;
                        node_index[i1] = node_index[i2];
                    } /* Second variable belongs to an envelope; adding the first variable to the same envelope. */ else { /* Creating a new envelope */
                        envelope[num_envelopes] = new Envelope(dim);
                        envelope[num_envelopes].is_missing = 1;
                        envelope[num_envelopes].num_nodes = 2;
                        envelope[num_envelopes].num_edges = 1;
                        envelope[num_envelopes].node[0] = i1;
                        envelope[num_envelopes].node[1] = i2;
                        envelope[num_envelopes].edge[0] = e;
                        node_index[i1] = num_envelopes;
                        node_index[i2] = num_envelopes;
                        num_envelopes++;
                    } /* Creating a new envelope */
                } /* Both variables are missing */
            } /* First variable is missing */
        } /* For each of the edges */

        /* Making sure all uninstantiated nodes are assigned to some envelope */
        for (int i = 0; i < dim; i++)
            if (node_index[i] == -1 && is_missing(v.ddata[dim_index[i]])) { /* Creating a single node envelope */
                envelope[num_envelopes] = new Envelope(1);
                envelope[num_envelopes].is_missing = 1;
                envelope[num_envelopes].num_nodes = 1;
                envelope[num_envelopes].num_edges = 0;
                envelope[num_envelopes].node[0] = i;
                node_index[i] = num_envelopes;
                num_envelopes++;
            } /* Creating a single node envelope */

        node_index = null;

        return (num_envelopes);
    }

    int compute_envelopes_full() {
        /* Given a forest structure, partitions the edges into trees (envelopes)
         */
        int num_envelopes = 0;

        int[]node_index;

        /* Initializing */
        node_index = new int[dim];
        for (int i = 0; i < dim; i++)
            /* Nodes are not yet assigned to envelopes */
            node_index[i] = -1;

        for (int e = 0; e < num_edges; e++) { /* For each of the edges */
            /* Shortcuts */
            int i1 = edge[e][0];
            int i2 = edge[e][1];

            if (node_index[i1] != -1) { /* First variable belongs to an envelope; adding the second variable to the same envelope. */
                envelope[node_index[i1]].edge[envelope[node_index[i1]].num_edges] = e;
                envelope[node_index[i1]].num_edges++;
                envelope[node_index[i1]].node[envelope[node_index[i1]].num_nodes] = i2;
                envelope[node_index[i1]].num_nodes++;
                node_index[i2] = node_index[i1];
            } /* First variable belongs to an envelope; adding the second variable to the same envelope. */ else if (node_index[i2] != -1) { /* Second variable belongs to an envelope; adding the first variable to the same envelope. */
                envelope[node_index[i2]].edge[envelope[node_index[i2]].num_edges] = e;
                envelope[node_index[i2]].num_edges++;
                envelope[node_index[i2]].node[envelope[node_index[i2]].num_nodes] = i1;
                envelope[node_index[i2]].num_nodes++;
                node_index[i1] = node_index[i2];
            } /* Second variable belongs to an envelope; adding the first variable to the same envelope. */ else { /* Creating a new envelope */
                envelope[num_envelopes] = new Envelope(dim);
                envelope[num_envelopes].is_missing = 1;
                envelope[num_envelopes].num_nodes = 2;
                envelope[num_envelopes].num_edges = 1;
                envelope[num_envelopes].node[0] = i1;
                envelope[num_envelopes].node[1] = i2;
                envelope[num_envelopes].edge[0] = e;
                node_index[i1] = num_envelopes;
                node_index[i2] = num_envelopes;
                num_envelopes++;
            } /* Creating a new envelope */
        } /* For each edge */

        /* Making sure all nodes are assigned to some envelope */
        for (int i = 0; i < dim; i++)
            if (node_index[i] == -1) { /* Creating a single node envelope */
                envelope[num_envelopes] = new Envelope(1);
                envelope[num_envelopes].is_missing = 1;
                envelope[num_envelopes].num_nodes = 1;
                envelope[num_envelopes].num_edges = 0;
                envelope[num_envelopes].node[0] = i;
                node_index[i] = num_envelopes;
                num_envelopes++;
            } /* Creating a single node envelope */

        node_index = null;

        return (num_envelopes);
    }

    void  PostProcess() {
        /* Rearranging the parameters */

        int[]visited;

        int[][]child;
        int[]num_parents;
        int num_inserted;

        /* Temporary variable(s) */

        switch (type) {
            case DIST_FACTOR:
                for (int i = 0; i < dim; i++)
                    subdist[0][i].PostProcess();
                break;
            case DIST_BERNOULLI:
                if (subdist != null)
                    for (int i = 0; i < dim; i++)
                        for (int j = 0; j < num_states; j++)
                            subdist[i][j].PostProcess();
                break;
            case DIST_CONDCHOWLIU:
                if (subdist != null)
                    for (int i = 0; i < dim; i++)
                        for (int j = 0; j < num_states; j++)
                            subdist[i][j].PostProcess();

                break;
            case DIST_CHOWLIU:
                if (subdist != null)
                    for (int i = 0; i < dim; i++)
                        for (int j = 0; j < num_states; j++)
                            subdist[i][j].PostProcess();

                break;
            case DIST_BN_ME_BIVAR:
            case DIST_BN_CME_BIVAR:
                /* Determining the order of nodes to simulate the data */

                /* Partial order sorting */

                /* Building a matrix of children */
                num_parents = new int[dim];
                for (int i = 0; i < dim; i++)
                    num_parents[i] = 0;

                child = new int[dim][];
                for (int i = 0; i < dim; i++)
                    child[i] = new int[dim];
                for (int i = 0; i < dim; i++)
                    for (int j = 0; j < dim; j++)
                        child[i][j] = 0;

                for (int i = 0; i < dim; i++)
                    for (int j = 1; j < num_features[i]; j++) {
                        if (feature_index[i][j] < dim) {
                            child[feature_index[i][j]][i] = 1;
                            num_parents[i]++;
                        }
                    }

                visited = new int[dim];
                for (int i = 0; i < dim; i++)
                    visited[i] = 0;

                /* Building an ordering */
                num_inserted = 0;
                while (num_inserted < dim) {
                    /* Looking for the non-visited node with 0 parents */
                    int i = 0;
                    while (visited[i] != 0 || num_parents[i] > 0)
                        i++;

                    /* Adding i to the list */
                    sim_order[num_inserted] = i;
                    num_inserted++;

                    visited[i] = 1;
                    for (int j = 0; j < dim; j++)
                        if (child[i][j]!= 0)
                            num_parents[j]--;
                }

                visited = null;
                num_parents = null;
                for (int i = 0; i < dim; i++)
                    child[i] = null;
                child = null;

                break;
            default:
                ;
        }
    }

    void InformationGain(int num_points, int[][]data, double[]w, double norm_const, double[]sw,
                    int[]candidate, double[]gain, double[]alpha, int index) {
        /* Implementation of the Information Gain score */
        int max_num_features;

        double[]param_sum_f;
        double[]p;

        double d1, d2; // First and second derivatives

        int iteration;

        /* Determining the dimensionality of the candidate array */
        if (type == DIST_BN_ME_BIVAR)
            max_num_features = dim;
        else if (type == DIST_BN_CME_BIVAR)
            max_num_features = 2 * dim;

        /* Initializing the array of linear sums for each datum */
        param_sum_f = new double[num_points];
        for (int n = 0; n < num_points; n++)
            param_sum_f[n] = sigma[index][0];

        /* Allocating the array of current probability values */
        p = new double[num_points];

        /* Computing the linear sums in the exponents for each vectors in the data set */
        for (int n = 0; n < num_points; n++)
            for (int i = 1; i < num_features[index]; i++)
                if (data[n][feature_index[index][i]] != 0)
                    param_sum_f[n] += sigma[index][i];

        for (int i = 0; i < max_num_features; i++)
            if (candidate[i] != 0) { /* Computing Information Gain and corresponding coefficient for each of candidate features */
                alpha[i] = 0;

                iteration = 0;
                d1 = norm_const; // Not to trigger compiler warnings
                while ((iteration == 0 || abs(d1) / norm_const > NR_EPSILON) && iteration < MAX_NR_ITERATIONS) { /* Newton-Raphson */
                    iteration++;

                    /* Computing the probabilities P(1|params,new parameter) */
                    for (int n = 0; n < num_points; n++)
                        if (data[n][i] != 0)
                            p[n] = 1.0 - 1.0 / (1.0 + exp(alpha[i] + param_sum_f[n]));

                    /* Computing the first derivative */
                    d1 = sw[i];
                    for (int n = 0; n < num_points; n++)
                        if (data[n][i] != 0)
                            d1 -= w[n] * p[n];

                    /* Computing the second derivative */
                    d2 = 0.0;
                    for (int n = 0; n < num_points; n++)
                        if (data[n][i]!= 0)
                            d2 -= w[n] * p[n] * (1.0 - p[n]);

                    /* Updating the parameter */
                    alpha[i] -= d1 / d2;
                } /* Newton-Raphson */

                /* Computing gain */
                gain[i] = alpha[i] * sw[i];
                for (int n = 0; n < num_points; n++)
                    if (data[n][i]!= 0)
                        gain[i] -= w[n] * (log(1.0 + exp(alpha[i] + param_sum_f[n])) - log(1.0 + exp(param_sum_f[n])));
                gain[i] /= norm_const;

            } /* Computing Information Gain and corresponding coefficient for each of candidate features */

        /* Deallocating the arrays */
        p = null;
        param_sum_f = null;

        return;
    }

    double learn_univariate_conditional_maxent(int num_points, int[][]data, double[]w, double norm_const, int index,
                                               double[]sw, int num_features, double[]delta, int[]feature) {

        double wll = 0.0;                        // Weighted log-likelihood
        double prev_wll = NEG_INF;               // Previous value of weighted log-likelihood
        double[]delta_sum_f;                   // Linear sums in the exponents (contribution from base)
        int iteration = 0;                      // Iteration of the conjugate gradient algorithm
        double []gradient, old_gradient;       // Gradient of the weighted log-likelihood
        double[]xi;                            // Direction of the ascent
        double gamma, gamma_pr, gamma_fr;      // Coefficient used in determining the direction of the ascent
        double[]xi_sum_f;                      // Linear sums in the exponents (contribution from the direction)
        double nu;                             // The solution of the linear search problem
        int iteration_nr;                     // Newton-Raphson iteration index
        double d1, d2;                          // First and second derivative of the log-likelihood
        double[]p;                             // Probability P(1|other variables, parameters)
        
        /* Temporary variables */
        double gg, og, oo;

  /*
  for (int i=0;i<num_features; i++ )
    delta[i]=0.0;
  */

        /* Initializing the arrays of linear sums for each datum */
        delta_sum_f = new double[num_points];
        for (int n = 0; n < num_points; n++)
            delta_sum_f[n] = delta[0];

        xi_sum_f = new double[num_points];

        xi = new double[num_features];

        /* Allocating the array of current probability values */
        p = new double[num_points];

        /* Precomputing values */
        /* Computing the linear sums in the exponents */
        for (int n = 0; n < num_points; n++)
            for (int i = 1; i < num_features; i++)
                if (data[n][feature[i]]!= 0)
                    delta_sum_f[n] += delta[i];

        /* Computing weighted log-likelihood */
        for (int n = 0; n < num_points; n++)
            if (data[n][index]!= 0)
                wll += w[n] * (delta_sum_f[n] - log(1.0 + exp(delta_sum_f[n])));
            else
                wll -= w[n] * log(1.0 + exp(delta_sum_f[n]));

        old_gradient = null;
        gradient = null; // Not to trigger compiler warnings
        while ((wll - prev_wll) / norm_const > cg_epsilon) {
            prev_wll = wll;

            /* Computing the gradient */
            if (iteration > 0) {
                if (iteration > 1)
                    old_gradient = null;
                old_gradient = gradient;
            }

            gradient = new double[num_features];
            for (int i = 0; i < num_features; i++)
                gradient[i] = sw[feature[i]];

            for (int n = 0; n < num_points; n++)
                gradient[0] -= w[n] * exp(delta_sum_f[n]) / (1.0 + exp(delta_sum_f[n]));
            gradient[0] /= norm_const;

            for (int i = 1; i < num_features; i++) {
                for (int n = 0; n < num_points; n++)
                    if (data[n][feature[i]] != 0)
                        gradient[i] -= w[n] * exp(delta_sum_f[n]) / (1.0 + exp(delta_sum_f[n]));
                gradient[i] /= norm_const;
            }

            /* Computing the new direction */
            if (iteration == 0)
                for (int i = 0; i < num_features; i++)
                    xi[i] = -gradient[i];
            else {
                gg = 0.0;
                og = 0.0;
                oo = 0.0;
                for (int i = 0; i < num_features; i++) {
                    gg += gradient[i] * gradient[i];
                    og += gradient[i] * old_gradient[i];
                    oo += old_gradient[i] * old_gradient[i];
                }

                gamma_pr = (gg - og) / oo;  // Polak-Ribiere
                gamma_fr = gg / oo;       // Fletcher-Reeves

                if (gamma_pr < -gamma_fr)
                    gamma = -gamma_fr;
                else if (gamma_pr > gamma_fr)
                    gamma = gamma_fr;
                else
                    gamma = gamma_pr;

                for (int i = 0; i < num_features; i++)
                    xi[i] = gradient[i] - gamma * old_gradient[i];
            }

            /* Line search optimization algorithm */

            /* Pre-computing commonly used values */
            /* Exponent contribution from the new direction */
            for (int n = 0; n < num_points; n++)
                xi_sum_f[n] = xi[0];

            for (int n = 0; n < num_points; n++)
                for (int i = 1; i < num_features; i++)
                    if (data[n][feature[i]] != 0)
                        xi_sum_f[n] += xi[i];

            nu = 0.0;
            iteration_nr = 0;

            /* Newton-Raphson */
            d1 = norm_const;   // Not to trigger the compiler warning
            while ((iteration_nr == 0 || abs(d1) / norm_const > NR_EPSILON) && iteration_nr < MAX_NR_ITERATIONS) {
                iteration_nr++;

                /* First derivative */
                d1 = 0.0;
                for (int i = 0; i < num_features; i++)
                    d1 += sw[feature[i]] * xi[i];

                for (int n = 0; n < num_points; n++) {
                    p[n] = exp(delta_sum_f[n] + nu * xi_sum_f[n]) / (1.0 + exp(delta_sum_f[n] + nu * xi_sum_f[n]));
                    d1 -= w[n] * xi_sum_f[n] * p[n];
                }

                d2 = 0.0;
                for (int n = 0; n < num_points; n++)
                    d2 -= w[n] * xi_sum_f[n] * xi_sum_f[n] * p[n] * (1.0 - p[n]);

                nu -= d1 / d2;
            }

            /* Updating the parameters */
            for (int i = 0; i < num_features; i++)
                delta[i] += nu * xi[i];

            /* Updating sums of features */
            for (int n = 0; n < num_points; n++)
                delta_sum_f[n] += nu * xi_sum_f[n];

            wll = 0.0;
            /* Computing weighted log-likelihood */
            for (int n = 0; n < num_points; n++)
                if (data[n][index])
                    wll += w[n] * (delta_sum_f[n] - log(1.0 + exp(delta_sum_f[n])));
                else
                    wll -= w[n] * log(1.0 + exp(delta_sum_f[n]));

            iteration++;
        }

        /* Deallocating the arrays */
        if (iteration > 0) {
            gradient = null;
            if (iteration > 1)
                old_gradient = null;
        }

        p = null;
        xi = null;
        xi_sum_f = null;
        delta_sum_f = null;

        return (wll);
    }

    void normalize_logistic(double[]log_un_prob, double[]prob, int num_states, int first) {
        /* Calculating normalized softmax function given unnormalized exponent expressions */
        
        /* Temporary variable(s) */
        double sum;
        double max;

        if (first != 0)
            /* First exponent is assumed 0.0 */
            max = 0.0;
        else
            max = NEG_INF;

        for (int i = first; i < num_states; i++)
            if (max < log_un_prob[i])
                max = log_un_prob[i];

        if (first != 0)
            sum = exp(-max);
        else
            sum = 0.0;

        for (int i = first; i < num_states; i++) {
            prob[i] = exp(log_un_prob[i] - max);
            sum += prob[i];
        }

        /* Normalizing */
        for (int i = first; i < num_states; i++)
            prob[i] /= sum;
    }

    Distribution Project() {
        /* Converting HMM models with inputs to models without inputs */

        Distribution transition;

        /* Temporary variable(s) */
        double sum;

        /* Converting the structures */
        switch (type) {
            case DIST_TRANSLOGISTIC:
                /* Allocating collapsed transition distribution */
                transition = new Distribution(DIST_CONDBERNOULLI, num_states, 1);

                /* Retrieving the distribution values from the base transition distribution */
                sum = 0.0;
                for (int i = 0; i < num_states; i++) { /* First state value i */
                    transition.state_prob[i] = exp(lambda[i]);
                    sum += transition.state_prob[i];
                } /* First state value i */

                for (int i = 0; i < num_states; i++)
                    transition.state_prob[i] /= sum;
                for (int i = 0; i < num_states; i++)
                    transition.log_state_prob[i] = log(transition.state_prob[i]);

                for (int i = 0; i < num_states; i++) { /* Previous state i */
                    sum = 0.0;
                    for (int j = 0; j < num_states; j++) {
                        transition.cond_state_prob[i][j] = exp(sigma[i][j]);
                        sum += transition.cond_state_prob[i][j];
                    }

                    /* Normalizing */
                    for (int j = 0; j < num_states; j++)
                        transition.cond_state_prob[i][j] /= sum;
                    for (int j = 0; j < num_states; j++)
                        transition.log_cond_state_prob[i][j] = log(transition.cond_state_prob[i][j]);
                } /* Previous state i */

                /* Initializing priors for first state and transition matrix */
                /* !!! A slight hack !!! */
                for (int i = 0; i < transition.num_states; i++)
                    for (int j = 0; j < transition.num_states; j++)
                        transition.pcount_uni[i][j] = 0.00001;
                for (int i = 0; i < transition.num_states; i++)
                    transition.pcount_single[i] = 0.00001 * transition.num_states;
                transition.pcount = 0.00001 * transition.num_states * transition.num_states;

                break;
            case DIST_LOGISTIC:
                /* Allocating collapsed transition distribution */
                transition = new Distribution(DIST_BERNOULLI, num_states, 1);

                /* Retrieving the distribution values from the base transition distribution */
                sum = 0.0;
                for (int i = 0; i < num_states; i++) { /* First state value i */
                    transition.state_prob[i] = exp(lambda[i]);
                    sum += transition.state_prob[i];
                } /* First state value i */

                for (int i = 0; i < num_states; i++)
                    transition.state_prob[i] /= sum;
                for (int i = 0; i < num_states; i++)
                    transition.log_state_prob[i] = log(transition.state_prob[i]);

                /* Initializing priors for first state and transition matrix */
                /* !!! A slight hack !!! */
                for (int i = 0; i < transition.num_states; i++)
                    transition.pcount_single[i] = 0.00001 * transition.num_states;
                transition.pcount = 0.00001 * transition.num_states;

                break;
            default:
                transition = null;
        }

        return (transition);
    }

    void  Expand(Distribution transition) {
        /* Expands a model without inputs to a model with inputs of dimension d */

        /* Temporary variable(s) */
        double temp;

        /* Converting the structures */
        switch (type) {
            case DIST_CONDBERNOULLI:
                /* Converting parameter values */

                /* First entry in the sequence */
                temp = log_state_prob[0];
                transition.lambda[0] = 0.0;
                for (int i = 1; i < num_states; i++)
                    transition.lambda[i] = log_state_prob[i] - temp;

                for (int i = 0; i < num_states; i++) { /* Transition from state i */
                    temp = log_cond_state_prob[i][0];
                    transition.sigma[i][0] = 0.0;
                    for (int j = 1; j < num_states; j++)
                        transition.sigma[i][j] = log_cond_state_prob[i][j] - temp;
                } /* Transition from state i */

                /* Zeroing out the linear term */
                for (int i = 0; i < num_states; i++)
                    for (int j = 0; j < transition.dim; j++)
                        transition.rho[i][j] = 0.0;

                break;
            case DIST_BERNOULLI:
                /* Converting parameter values */

                /* First entry in the sequence */
                temp = log_state_prob[0];
                transition.lambda[0] = 0.0;
                for (int i = 1; i < num_states; i++)
                    transition.lambda[i] = log_state_prob[i] - temp;

                /* Zeroing out the linear term */
                for (int i = 0; i < num_states; i++)
                    for (int j = 0; j < transition.dim; j++)
                        transition.rho[i][j] = 0.0;

                break;
            default:
                ;
        }

    }

    double[] TrueLogLikelihood(Distribution Q[], int num_Q) {
        double[] ll = new double[num_Q];  // Log-likelihood

        double[][][][] pair_P;             // Pairwise probability table for distribution P
        boolean[] visited;                 // Indicators of visited nodes

        /* Initializing the log-likelihoods */
        for (int i = 0; i < num_Q; i++)
            ll[i] = 0.0;

        switch (type) {
            case DIST_CHOWLIU:
                /* Allocating the array of pairwise probabilities for P(x) */
                pair_P = new double[dim][][][];
                for (int i1 = 0; i1 < dim; i1++) {
                    pair_P[i1] = new double [i1][][];
                    for (int i2 = 0; i2 < i1; i2++) {
                        pair_P[i1][i2] = new double[num_states][];
                        for (int b1 = 0; b1 < num_states; b1++)
                            pair_P[i1][i2][b1] = new double[num_states];
                    }
                }

                /* Allocating the array of indicators for visited nodes */
                visited = new boolean[dim];

                /* Initializing the indicators */
                for (int i1 = 0; i1 < dim; i1++)
                    visited[i1] = false;

                /* Initializing all pairs to conditional independence */
                for (int i1 = 0; i1 < dim; i1++)
                    for (int i2 = 0; i2 < i1; i2++)
                        for (int b1 = 0; b1 < num_states; b1++)
                            for (int b2 = 0; b2 < num_states; b2++)
                                pair_P[i1][i2][b1][b2] = ind_prob[i1][b1] * ind_prob[i2][b2];

                for (int e = 0; e < num_edges; e++) { /* For each edge in E_P */
                    int node1 = edge[e][0];
                    int node2 = edge[e][1];
                    if (FLAG_KL) {
                        out.format("Considering edge %d %d\n", node1, node2);
                    }
                    /* Updating the edge probabilities for edge e */
                    for (int b1 = 0; b1 < num_states; b1++)
                        for (int b2 = 0; b2 < num_states; b2++)
                            pair_P[node1][node2][b1][b2] = edge_prob[e][b1][b2];

                    if (visited[node1]) { /* node2 is not visited */
                        if (FLAG_KL) {
                            out.format("First node (%d) visited\n", node1);
                        }
                        /* Updating node2 probabilities */
                        visited[node2] = true;
                        /* node1>node2 */

                        /* node1>node2>i1 */
                        for (int i1 = 0; i1 < node2; i1++)
                            if (visited[i1]) {
                                if (FLAG_KL) {
                                    out.format("Updating pair %d %d\n", node2, i1);
                                }
                                for (int b2 = 0; b2 < num_states; b2++)
                                    for (int b1 = 0; b1 < num_states; b1++) {
                                        pair_P[node2][i1][b2][b1] = 0.0;
                                        for (int b = 0; b < num_states; b++)
                                            pair_P[node2][i1][b2][b1] +=
                                                    (pair_P[node1][node2][b][b2] * pair_P[node1][i1][b][b1] / ind_prob[node1][b]);
                                    }
                            }

                        /* node1>i1>node2 */
                        for (int i1 = node2 + 1; i1 < node1; i1++)
                            if (visited[i1]) {
                                if (FLAG_KL) {
                                    out.format("Updating pair %d %d\n", i1, node2);
                                }
                                for (int b2 = 0; b2 < num_states; b2++)
                                    for (int b1 = 0; b1 < num_states; b1++) {
                                        pair_P[i1][node2][b1][b2] = 0.0;
                                        for (int b = 0; b < num_states; b++)
                                            pair_P[i1][node2][b1][b2] +=
                                                    (pair_P[node1][node2][b][b2] * pair_P[node1][i1][b][b1] / ind_prob[node1][b]);
                                    }
                            }

                        /* i1>node1>node2 */
                        for (int i1 = node1 + 1; i1 < dim; i1++)
                            if (visited[i1]) {
                                if (FLAG_KL) {
                                    out.format("Updating pair %d %d\n", i1, node2);
                                }
                                for (int b2 = 0; b2 < num_states; b2++)
                                    for (int b1 = 0; b1 < num_states; b1++) {
                                        pair_P[i1][node2][b1][b2] = 0.0;
                                        for (int b = 0; b < num_states; b++)
                                            pair_P[i1][node2][b1][b2] +=
                                                    (pair_P[node1][node2][b][b2] * pair_P[i1][node1][b1][b] / ind_prob[node1][b]);
                                    }
                            }
                    } /* node2 is not visited */ else if (visited[node2]) { /* node1 is not visited */
                        if (FLAG_KL) {
                            out.format("Second node (%d) visited\n", node2);
                        }
                        /* Updating node1 probabilities */
                        visited[node1] = true;
                        /* node1>node2 */

                        /* node1>node2>i2 */
                        for (int i2 = 0; i2 < node2; i2++)
                            if (visited[i2]) {
                                if (FLAG_KL) {
                                    out.format("Updating pair %d %d\n", node1, i2);
                                }
                                for (int b1 = 0; b1 < num_states; b1++)
                                    for (int b2 = 0; b2 < num_states; b2++) {
                                        pair_P[node1][i2][b1][b2] = 0.0;
                                        for (int b = 0; b < num_states; b++)
                                            pair_P[node1][i2][b1][b2] +=
                                                    (pair_P[node1][node2][b1][b] * pair_P[node2][i2][b][b2] / ind_prob[node2][b]);
                                    }
                            }

                        /* node1>i2>node2 */
                        for (int i2 = node2 + 1; i2 < node1; i2++)
                            if (visited[i2]) {
                                if (FLAG_KL) {
                                    out.format("Updating pair %d %d\n", node1, i2);
                                }
                                for (int b1 = 0; b1 < num_states; b1++)
                                    for (int b2 = 0; b2 < num_states; b2++) {
                                        pair_P[node1][i2][b1][b2] = 0.0;
                                        for (int b = 0; b < num_states; b++)
                                            pair_P[node1][i2][b1][b2] +=
                                                    (pair_P[node1][node2][b1][b] * pair_P[i2][node2][b2][b] / ind_prob[node2][b]);
                                    }
                            }

                        /* i2>node1>node2 */
                        for (int i2 = node1 + 1; i2 < dim; i2++)
                            if (visited[i2]) {
                                if (FLAG_KL) {
                                    out.format("Updating pair %d %d\n", i2, node1);
                                }
                                for (int b1 = 0; b1 < num_states; b1++)
                                    for (int b2 = 0; b2 < num_states; b2++) {
                                        pair_P[i2][node1][b2][b1] = 0.0;
                                        for (int b = 0; b < num_states; b++)
                                            pair_P[i2][node1][b2][b1] +=
                                                    (pair_P[node1][node2][b1][b] * pair_P[i2][node2][b2][b] / ind_prob[node2][b]);
                                    }
                            }
                    } /* node1 is not visited */ else { /* Neither node has been visited */
                        if (FLAG_KL) {
                            out.format("Neither node (%d %d) visited\n", node1, node2);
                        }
                        visited[node1] = true;
                        visited[node2] = true;
                    } /* Neither node has been visited */
                } /* For each edge in E_P */

                /* Deallocating the array of indicators for visited nodes */
                visited = null;
                if (FLAG_KL) {
                    /* !!! START !!! */
                    for (int i1 = 0; i1 < dim; i1++)
                        for (int i2 = 0; i2 < i1; i2++)
                            out.format("%d\t%d\t%0.3le\t%0.3le\t%0.3le\t%0.3le\n", i1, i2,
                                    pair_P[i1][i2][0][0], pair_P[i1][i2][0][1], pair_P[i1][i2][1][0], pair_P[i1][i2][1][1]);
                    out.format("\n");

                    double[][] MI;

                    MI = new double[dim][];
                    for (int i1 = 0; i1 < dim; i1++) {
                        MI[i1] = new double[dim];
                        for (int i2 = 0; i2 < dim; i2++)
                            MI[i1][i2] = 0.0;
                    }

                    for (int i1 = 0; i1 < dim; i1++)
                        for (int i2 = 0; i2 < i1; i2++) {
                            for (int b1 = 0; b1 < num_states; b1++)
                                MI[i1][i2] -= xlogx(ind_prob[i1][b1]);
                            for (int b2 = 0; b2 < num_states; b2++)
                                MI[i1][i2] -= xlogx(ind_prob[i2][b2]);
                            for (int b1 = 0; b1 < num_states; b1++)
                                for (int b2 = 0; b2 < num_states; b2++)
                                    MI[i1][i2] += xlogx(pair_P[i1][i2][b1][b2]);
                            MI[i2][i1] = MI[i1][i2];
                        }

                    /* Displaying */
                    for (int i1 = 0; i1 < dim; i1++) {
                        for (int i2 = 0; i2 < dim; i2++)
                            out.format("\t%0.3le", MI[i1][i2]);
                        out.format("\n");
                    }

                    for (int i1 = 0; i1 < dim; i1++)
                        MI[i1] = null;
                    MI = null;

                    /* !!! END !!! */
                }
                for (int i = 0; i < num_Q; i++)
                    /* Distribution Q[i] */
                    if (Q[i].type == DIST_CHOWLIU) { /* Computing log-likelihood */
                        /* Computing contributions of the marginals */
                        for (int node1 = 0; node1 < dim; node1++)
                            for (int b1 = 0; b1 < num_states; b1++)
                                ll[i] += ind_prob[node1][b1] * log(Q[i].ind_prob[node1][b1]);

                        /* Computing contributions from the edges */
                        for (int e = 0; e < Q[i].num_edges;
                             e++) { /* Edge e for Q[i] */
                            int node1 = Q[i].edge[e][0];
                            int node2 = Q[i].edge[e][1];
                            for (int b1 = 0; b1 < num_states; b1++)
                                ll[i] -= ind_prob[node1][b1] * log(Q[i].ind_prob[node1][b1]);
                            for (int b2 = 0; b2 < num_states; b2++)
                                ll[i] -= ind_prob[node2][b2] * log(Q[i].ind_prob[node2][b2]);
                            for (int b1 = 0; b1 < num_states; b1++)
                                for (int b2 = 0; b2 < num_states; b2++)
                                    ll[i] += pair_P[node1][node2][b1][b2] * log(Q[i].edge_prob[e][b1][b2]);
                        } /* Edge e for Q[i] */
                    } /* Computing log-likelihood */

                /* Deallocating the array of pairwise probabilities for P(x) */
                for (int i1 = 0; i1 < dim; i1++) {
                    for (int i2 = 0; i2 < i1; i2++) {
                        for (int b1 = 0; b1 < num_states; b1++)
                            pair_P[i1][i2][b1] = null;
                        pair_P[i1][i2] = null;
                    }
                    pair_P[i1] = null;
                }
                pair_P = null;

                break;
            default:
                ;
        }

        return (ll);
    }

    DataPoint FillIn(DataPoint v) {
        /* Computing the most likely sequence for the missing entries */
        DataPoint new_v = null;

        double[][] mult;                // Total update from instantiated nodes
        double[][] max_prob;            // Max prob of the portion of the envelope
        int[][] max_index;             // Index corresponding to the max_prob

//        int node, node1, node2, root_node;
//        int b, b1, b2;
//        int env;
//        int e;

        /* Temporary variable(s) */
        double root_prob;
        double current_prob;

        switch (type) {
            case DIST_CHOWLIU:
                /* Multiplication factor for each node */
                mult = new double[dim][];
                for (int node = 0; node < dim; node++)
                    mult[node] = new double[num_states];

                /* Computing multiplicative factor for each node */
                if (subdist != null)
                    /* Mixture */
                    for (int node = 0; node < dim; node++)
                        for (int b = 0; b < num_states; b++)
                            mult[node][b] = exp(subdist[node][b].log_prob(v, null));

                if (subdist == null)
                    /* Computing the envelopes */
                    num_envelopes = compute_envelopes(v, mult);

                new_v = new DataPoint(v.ddim, v.rdim);

                /* Computing posterior probabilities for missing values */
                for (int env = 0; env < num_envelopes; env++)
                    /* For each envelope */
                    if (envelope[env].is_missing && envelope[env].num_nodes == 1) { /* Envelope contains exactly one missing variable. */
                        root_node = envelope[env].node[0];

                        new_v.ddata[root_node] = 0;
                        root_prob = ind_prob[root_node][0] * mult[root_node][0];
                        for (int b = 1; b < num_states; b++) { /* Picking the most probable missing variable */
                            current_prob = ind_prob[root_node][b] * mult[root_node][b];
                            if (root_prob < current_prob) {
                                new_v.ddata[root_node] = b;
                                root_prob = current_prob;
                            }
                        } /* Picking the most probable missing variable */
                    } /* Envelope contains exactly one missing variable. */ else if (envelope[env].is_missing && envelope[env].num_nodes > 1) { /* Envelope contains more than one missing variable. */
                        /* Allocating arrays of maximum probabilities for this envelope */
                        max_prob = new double[envelope[env].num_nodes - 1];
                        max_index = new int[envelope[env].num_nodes - 1];

                        for (int e = envelope[env].num_edges - 1;
                             e >= 0;
                             e--) { /* Edges in the backward pass */
                            node1 = edge[envelope[env].edge[e]][0];
                            node2 = edge[envelope[env].edge[e]][1];

                            /* Allocating the part of the maximum probability arrays */
                            max_prob[e] = new double[num_states];
                            max_index[e] = new int[num_states];

                            if (envelope[env].node[e + 1] == node2)
                                /* node1 is the parent of node2 */
                                for (int b1 = 0; b1 < num_states; b1++) {
                                    max_prob[e][b1] = mult[node2][0] * edge_prob[envelope[env].edge[e]][b1][0];
                                    max_index[e][b1] = 0;
                                    for (int b2 = 1; b2 < num_states; b2++) {
                                        current_prob = mult[node2][b2] * edge_prob[envelope[env].edge[e]][b1][b2];
                                        if (current_prob > max_prob[e][b1]) {
                                            max_prob[e][b1] = current_prob;
                                            max_index[e][b1] = b2;
                                        }
                                        /* Updating the parent's multiplication factor */
                                        mult[node1][b1] *= max_prob[e][b1] / ind_prob[node1][b1];
                                    }
                                }
                            else
                                /* node2 is the parent of node1 */
                                for (int b2 = 0; b2 < num_states; b2++) {
                                    max_prob[e][b2] = mult[node1][0] * edge_prob[envelope[env].edge[e]][0][b2];
                                    max_index[e][b2] = 0;
                                    for (int b1 = 1; b1 < num_states; b1++) {
                                        current_prob = mult[node1][b1] * edge_prob[envelope[env].edge[e]][b1][b2];
                                        if (current_prob > max_prob[e][b2]) {
                                            max_prob[e][b2] = current_prob;
                                            max_index[e][b2] = b1;
                                        }
                                        /* Updating the parent's multiplication factor */
                                        mult[node2][b2] *= max_prob[e][b2] / ind_prob[node2][b2];
                                    }
                                }
                        } /* Edges in the backward pass */

                        /* Updating the root of the envelope */
                        root_node = envelope[env].node[0];

                        /* Finding the max for the root node */
                        new_v.ddata[root_node] = 0;
                        root_prob = ind_prob[root_node][0] * mult[root_node][0];
                        for (int b = 1; b < num_states; b++) { /* Picking the most probable missing variable */
                            current_prob = ind_prob[root_node][b] * mult[root_node][b];
                            if (root_prob < current_prob) {
                                new_v.ddata[root_node] = b;
                                root_prob = current_prob;
                            }
                        } /* Picking the most probable missing variable */

                        /* Forward pass */
                        for (int e = 0; e < envelope[env].num_edges;
                             e++) {
                            node1 = edge[envelope[env].edge[e]][0];
                            node2 = edge[envelope[env].edge[e]][1];

                            if (envelope[env].node[e + 1] == node2)
                                /* node1 is the parent of node2 */
                                new_v.ddata[node2] = max_index[e][new_v.ddata[node1]];
                            else
                                /* node2 is the parent of node1 */
                                new_v.ddata[node1] = max_index[e][new_v.ddata[node2]];
                        }

                        /* Deallocating arrays of maximum probabilities for this envelope */
                        for (int node = 0; node < envelope[env].num_edges;
                             node++) {
                            max_prob[node] = null;
                            max_index[node] = null;
                        }
                        max_prob = null;
                        max_index = null;
                    } /* Envelope contains more than one missing variable */

                /* Copying instantiated components of v */
                for (int node = 0; node < dim; node++)
                    if (!is_missing(v.ddata[node]))
                        new_v.ddata[node] = v.ddata[node];

                /* Deallocating count collection structures */
                for (int node = 0; node < dim; node++)
                    mult[node] = null;
                mult = null;
                break;
            default:
                ;
        }

        /* Removing envelopes */
        for (int env = 0; env < num_envelopes; env++)
            envelope[env] = null;
        num_envelopes = 0;

        return (new_v);
    }






    /* Code acknowledgement */
    /* gammaln, digamma, and trigammma code by Thomas Minka */
    /* http://www.stat.cmu.edu/~minka/papers/dirichlet/minka-digamma.zip */

    /* Logarithm of the gamma function.

   References:

   1) W. J. Cody and K. E. Hillstrom, 'Chebyshev Approximations for
      the Natural Logarithm of the Gamma Function,' Math. Comp. 21,
      1967, pp. 198-203.

   2) K. E. Hillstrom, ANL/AMD Program ANLC366S, DGAMMA/DLGAMA, May,
      1969.

   3) Hart, Et. Al., Computer Approximations, Wiley and sons, New
      York, 1968.

   From matlab/gammaln.m
*/
    double gammaln(double x) {
        double result, y, xnum, xden;
        int i;
        double d1 = -5.772156649015328605195174e-1;
        double p1[] = {
                4.945235359296727046734888e0, 2.018112620856775083915565e2,
                2.290838373831346393026739e3, 1.131967205903380828685045e4,
                2.855724635671635335736389e4, 3.848496228443793359990269e4,
                2.637748787624195437963534e4, 7.225813979700288197698961e3
        };
        double q1[] = {
                6.748212550303777196073036e1, 1.113332393857199323513008e3,
                7.738757056935398733233834e3, 2.763987074403340708898585e4,
                5.499310206226157329794414e4, 6.161122180066002127833352e4,
                3.635127591501940507276287e4, 8.785536302431013170870835e3
        };
        double d2 = 4.227843350984671393993777e-1;
        double p2[] = {
                4.974607845568932035012064e0, 5.424138599891070494101986e2,
                1.550693864978364947665077e4, 1.847932904445632425417223e5,
                1.088204769468828767498470e6, 3.338152967987029735917223e6,
                5.106661678927352456275255e6, 3.074109054850539556250927e6
        };
        double q2[] = {
                1.830328399370592604055942e2, 7.765049321445005871323047e3,
                1.331903827966074194402448e5, 1.136705821321969608938755e6,
                5.267964117437946917577538e6, 1.346701454311101692290052e7,
                1.782736530353274213975932e7, 9.533095591844353613395747e6
        };
        double d4 = 1.791759469228055000094023e0;
        double p4[] = {
                1.474502166059939948905062e4, 2.426813369486704502836312e6,
                1.214755574045093227939592e8, 2.663432449630976949898078e9,
                2.940378956634553899906876e10, 1.702665737765398868392998e11,
                4.926125793377430887588120e11, 5.606251856223951465078242e11
        };
        double q4[] = {
                2.690530175870899333379843e3, 6.393885654300092398984238e5,
                4.135599930241388052042842e7, 1.120872109616147941376570e9,
                1.488613728678813811542398e10, 1.016803586272438228077304e11,
                3.417476345507377132798597e11, 4.463158187419713286462081e11
        };
        double c[] = {
                -1.910444077728e-03, 8.4171387781295e-04,
                -5.952379913043012e-04, 7.93650793500350248e-04,
                -2.777777777777681622553e-03, 8.333333333333333331554247e-02,
                5.7083835261e-03
        };
        double a = 0.6796875;

        if ((x <= 0.5) || ((x > a) && (x <= 1.5))) {
            if (x <= 0.5) {
                result = -log(x);
                /*  Test whether X < machine epsilon. */
                //      if(x+1 == 1) {
                if (abs(x) < COMP_EPSILON) {
                    return result;
                }
            } else {
                result = 0;
                x = (x - 0.5) - 0.5;
            }
            xnum = 0;
            xden = 1;
            for (int i = 0; i < 8; i++) {
                xnum = xnum * x + p1[i];
                xden = xden * x + q1[i];
            }
            result += x * (d1 + x * (xnum / xden));
        } else if ((x <= a) || ((x > 1.5) && (x <= 4))) {
            if (x <= a) {
                result = -log(x);
                x = (x - 0.5) - 0.5;
            } else {
                result = 0;
                x -= 2;
            }
            xnum = 0;
            xden = 1;
            for (int i = 0; i < 8; i++) {
                xnum = xnum * x + p2[i];
                xden = xden * x + q2[i];
            }
            result += x * (d2 + x * (xnum / xden));
        } else if (x <= 12) {
            x -= 4;
            xnum = 0;
            xden = -1;
            for (int i = 0; i < 8; i++) {
                xnum = xnum * x + p4[i];
                xden = xden * x + q4[i];
            }
            result = d4 + x * (xnum / xden);
        }
        /*  X > 12  */
        else {
            y = log(x);
            result = x * (y - 1) - y * 0.5 + .9189385332046727417803297;
            x = 1 / x;
            y = x * x;
            xnum = c[6];
            for (int i = 0; i < 6; i++) {
                xnum = xnum * y + c[i];
            }
            xnum *= x;
            result += xnum;
        }
        return result;
    }

    /* The digamma function is the derivative of gammaln.

   Reference:
    J Bernardo,
    Psi ( Digamma ) Function,
    Algorithm AS 103,
    Applied Statistics,
    Volume 25, Number 3, pages 315-317, 1976.

    From http://www.psc.edu/~burkardt/src/dirichlet/dirichlet.f
    (with modifications for negative numbers and extra precision)
*/
    double digamma(double x) {
        //  static const double neginf = -1.0/0,
        //  static const double neginf = -1e+300,
        static const double
                c = 12,
                d1 = -0.57721566490153286,
                d2 = 1.6449340668482264365, /* pi^2/6 */
                s = 1e-6,
                s3 = 1. / 12,
                s4 = 1. / 120,
                s5 = 1. / 252,
                s6 = 1. / 240,
                s7 = 1. / 132,
                s8 = 691 / 32760,
                s9 = 1 / 12,
                s10 = 3617 / 8160;
        double result;
        if (false) {
            double cache_x = 0;
            int hits = 0, total = 0;
            total++;
            if (x == cache_x) {
                hits++;
            }
            if (total % 1000 == 1) {
                out.format("hits = %d, total = %d, hits/total = %g\n", hits, total,
                        ((double) hits) / total);
            }
            cache_x = x;
        }
        /* Illegal arguments */
        //  if((x == neginf) || isNaN(x)) {
        //    return( neginf );
        if (x == Double.NEGATIVE_INFINITY || isNaN(x)) {
            return (NEG_INF);
            //    return nan("");
            //    return 0.0/0;
        }
        /* Singularities */
        //  if((x <= 0) && (floor(x) == x)) {
        /* Dealing with comparing double-precision floats (SK) */
        if (x <= 0.0 && abs(floor(x) - x) <= COMP_THRESHOLD) {
            return (NEG_INF);
        }
        /* Negative values */
        /* Use the reflection formula (Jeffrey 11.1.6):
         * digamma(-x) = digamma(x+1) + pi*cot(pi*x)
         *
         * This is related to the identity
         * digamma(-x) = digamma(x+1) - digamma(z) + digamma(1-z)
         * where z is the fractional part of x
         * For example:
         * digamma(-3.1) = 1/3.1 + 1/2.1 + 1/1.1 + 1/0.1 + digamma(1-0.1)
         *               = digamma(4.1) - digamma(0.1) + digamma(1-0.1)
         * Then we use
         * digamma(1-z) - digamma(z) = pi*cot(pi*z)
         */
        if (x < 0) {
            return digamma(1 - x) + PI / tan(-PI * x);
        }
        /* Use Taylor series if argument <= S */
        if (x <= s) return d1 - 1 / x + d2 * x;
        /* Reduce to digamma(X + N) where (X + N) >= C */
        result = 0;
        while (x < c) {
            result -= 1 / x;
            x++;
        }
        /* Use de Moivre's expansion if argument >= C */
        /* This expansion can be computed in Maple via asympt(Psi(x),x) */
        if (x >= c) {
            double r = 1 / x;
            result += log(x) - 0.5 * r;
            r *= r;
            result -= r * (s3 - r * (s4 - r * (s5 - r * (s6 - r * s7))));
        }
        return result;
    }

    /* The trigamma function is the derivative of the digamma function.

   Reference:

    B Schneider,
    Trigamma Function,
    Algorithm AS 121,
    Applied Statistics,
    Volume 27, Number 1, page 97-99, 1978.

    From http://www.psc.edu/~burkardt/src/dirichlet/dirichlet.f
    (with modification for negative arguments and extra precision)
*/
    double trigamma(double x) {
        //  double neginf = -1.0/0,
        double //neginf = -1e+300,
                small = 1e-4,
                large = 8,
                c = 1.6449340668482264365, /* pi^2/6 = Zeta(2) */
                c1 = -2.404113806319188570799476,  /* -2 Zeta(3) */
                b2 = 1. / 6,
                b4 = -1. / 30,
                b6 = 1. / 42,
                b8 = -1. / 30,
                b10 = 5. / 66;
        double result;
        /* Illegal arguments */
        //  if((x == neginf) || isNaN(x)) {
        //    return 0.0/0;
        //    return nan("");
        //    return( neginf );
        if (x == Double.NEGATIVE_INFINITY || isNaN(x)) {
            return (NEG_INF);
        }
        /* Singularities */
        //  if((x <= 0) && (floor(x) == x)) {
        //    return -neginf;
        /* Fixing comparison of double-precision numbers */
        if ((x <= 0.0) && abs(floor(x) - x) <= COMP_THRESHOLD) {
            return (POS_INF);
        }
        /* Negative values */
        /* Use the derivative of the digamma reflection formula:
         * -trigamma(-x) = trigamma(x+1) - (pi*csc(pi*x))^2
         */
        if (x < 0) {
            result = PI / sin(-PI * x);
            return -trigamma(1 - x) + result * result;
        }
        /* Use Taylor series if argument <= small */
        if (x <= small) {
            return 1 / (x * x) + c + c1 * x;
        }
        result = 0;
        /* Reduce to trigamma(x+n) where ( X + N ) >= B */
        while (x < large) {
            result += 1 / (x * x);
            x++;
        }
        /* Apply asymptotic formula when X >= B */
        /* This expansion can be computed in Maple via asympt(Psi(1,x),x) */
        if (x >= large) {
            double r = 1 / (x * x);
            result += 0.5 * r + (1 + r * (b2 + r * (b4 + r * (b6 + r * (b8 + r * b10))))) / x;
        }
        return result;
    }


    double exp_dist(double x, double alpha) {
        return (alpha * exp(-alpha * x));
    }

    double gamma_dist(double x, double alpha, double beta) {
        return (exp((alpha - 1.0) * log(x) - beta * x + alpha * log(beta) - gammaln(alpha)));
    }

    double lognormal_dist(double x, double M, double S2) {
        return (exp(-(log(x) - M) * (log(x) - M) / (2.0 * S2)) / (x * sqrt(2 * PI * S2)));
    }

    static Distribution ReadDistribution(File input) throws IOException {
        /* Reads parameters of the sub-components of the distribution and creates them */

        int line_number = 0;

        /* Distribution */
        Distribution dist;

        /* Parameters */
        int t, param1, param2;
        double mdl_prior;         // Penalty term for parameters
        double pcount;            // Pseudo-count -- Dirichlet prior

        /* Temporary variable(s) */
        String temp_char;
        int[] dim_index;

        ReadFile readFile = new ReadFile(input);
        temp_char = readFile.read_word();
        if (temp_char.compareTo("factored") == 0)
            t = DIST_FACTOR;
        else if (temp_char.compareTo("independent") == 0)
            t = DIST_ALIAS_CI;
        else if (temp_char.compareTo("mixture-distinct") == 0)
            t = DIST_MIXTURE;
        else if (temp_char.compareTo("mixture-distinct-prior") == 0)
            t = DIST_MIXTURE_PRIOR;
        else if (temp_char.compareTo("mixture") == 0)
            t = DIST_ALIAS_MIXTURE;
        else if (temp_char.compareTo("mixture-prior") == 0)
            t = DIST_ALIAS_MIXTURE_PRIOR;
        else if (temp_char.compareTo("cl-mixture-distinct") == 0)
            t = DIST_CLMIXTURE;
        else if (temp_char.compareTo("cl-mixture") == 0)
            t = DIST_ALIAS_CLMIXTURE;
        else if (temp_char.compareTo("cl-mixture-prior") == 0)
            t = DIST_ALIAS_CLMIXTURE_PRIOR;
        else if (temp_char.compareTo("bernoulli") == 0)
            t = DIST_BERNOULLI;
        else if (temp_char.compareTo("bernoulli-prior") == 0)
            t = DIST_BERNOULLI_PRIOR;
        else if (temp_char.compareTo("chain-bernoulli") == 0)
            t = DIST_CONDBERNOULLI;
        else if (temp_char.compareTo("chain-bernoulli-prior") == 0)
            t = DIST_CONDBERNOULLI_PRIOR;
        else if (temp_char.compareTo("chain-bernoulli-global") == 0)
            t = DIST_CONDBERNOULLIG;
        else if (temp_char.compareTo("chain-bernoulli-global-prior") == 0)
            t = DIST_CONDBERNOULLIG_PRIOR;
        else if (temp_char.compareTo("chow-liu") == 0)
            t = DIST_CHOWLIU;
        else if (temp_char.compareTo("chow-liu-mdl") == 0)
            t = DIST_CHOWLIU_MDL;
        else if (temp_char.compareTo("chow-liu-prior") == 0)
            t = DIST_CHOWLIU_DIR_MDL;
        else if (temp_char.compareTo("conditional-chow-liu") == 0)
            t = DIST_CONDCHOWLIU;
        else if (temp_char.compareTo("conditional-chow-liu-mdl") == 0)
            t = DIST_CONDCHOWLIU_MDL;
        else if (temp_char.compareTo("maxent-full") == 0)
            t = DIST_ME_BIVAR;
        else if (temp_char.compareTo("BN-maxent") == 0)
            t = DIST_BN_ME_BIVAR;
        else if (temp_char.compareTo("BN-cond-maxent") == 0)
            t = DIST_BN_CME_BIVAR;
        else if (temp_char.compareTo("delta-exponential") == 0)
            t = DIST_DELTAEXP;
        else if (temp_char.compareTo("delta-gamma") == 0)
            t = DIST_DELTAGAMMA;
        else if (temp_char.compareTo("delta") == 0)
            t = DIST_DIRACDELTA;
        else if (temp_char.compareTo("exponential") == 0)
            t = DIST_EXP;
        else if (temp_char.compareTo("gamma") == 0)
            t = DIST_GAMMA;
        else if (temp_char.compareTo("log-normal") == 0)
            t = DIST_LOGNORMAL;
        else if (temp_char.compareTo("gaussian") == 0)
            t = DIST_NORMAL;
        else if (temp_char.compareTo("chain-gaussian") == 0)
            t = DIST_NORMALCHAIN;
        else if (temp_char.compareTo("tree-gaussian") == 0)
            t = DIST_NORMALCL;
        else if (temp_char.compareTo("logistic") == 0)
            t = DIST_LOGISTIC;
        else if (temp_char.compareTo("transition-logistic") == 0)
            t = DIST_TRANSLOGISTIC;
        else {
            System.err.format("Unknown distribution type %s on line %d.  Skipping.\n", temp_char, line_number);
        }

        temp_char = null;

        switch (t) {
            case DIST_FACTOR:
                /* Product distribution (Conditional independence) */
                param1 = readFile.read_long();
                if (param1 < 1) {
                    System.err.format("Error on line %d: the number of conditionally independent distributions must be at least 1.  Aborting.\n",
                            line_number);
                    System.exit(-1);
                }

                dist = new Distribution(DIST_FACTOR, 0, param1);
                if (INPUT_VERBOSE) {
                    System.out.format("%d-component product distribution\n", dist.dim);
                }
                for (int i = 0; i < dist.dim; i++)
                    dist.subdist[0][i] = ReadDistribution(input);
                dim_index = new int[param1];
                for (int i = 0; i < param1; i++)
                    dim_index[i] = i;
                dist.UpdateIndex(dim_index);
                dim_index = null;

                break;
            case DIST_ALIAS_CI:
                /* Conditionally independent distributions of the same type (shortcut) */
                param1 = readFile.read_long();
                if (param1 < 1) {
                    System.err.format("Error on line %d: the number of conditionally independent distributions must be at least 1.  Aborting.\n",
                            line_number);
                    System.exit(-1);
                }

                dist = new Distribution(DIST_FACTOR, 0, param1);
                if (INPUT_VERBOSE) {
                    System.out.format("%d-component product distribution\n", dist.dim);
                }
                dist.subdist[0][0] = ReadDistribution(input);
                for (int i = 1; i < dist.dim; i++)
                    dist.subdist[0][i] = dist.subdist[0][0].copy();
                dim_index = new int[param1];
                for (int i = 0; i < param1; i++)
                    dim_index[i] = i;
                dist.UpdateIndex(dim_index);
                dim_index = null;

                break;
            case DIST_MIXTURE:
            case DIST_MIXTURE_PRIOR:
                /* Mixture */
                param1 = readFile.read_long();
                if (param1 < 1) {
                    System.err.format("Error on line %d: the number of components in a mixture model must be at least 1.  Aborting.\n",
                            line_number);
                }
                dist = new Distribution(DIST_BERNOULLI, param1, 1);

                if (t == DIST_MIXTURE_PRIOR) { /* Reading in the parameters for the prior */
                    pcount = readFile.read_double();
                    if (pcount < 0.0) {
                        System.err.format("Error on line %d: the pseudo-counts for Dirichlet prior for mixing probabilities of a mixture must be non-negative.  Setting them to zero.\n",
                                line_number);
                        pcount = 0.0;
                    }
                    for (int i = 0; i < dist.num_states; i++)
                        dist.pcount_single[i] = pcount;
                    dist.pcount = pcount * dist.num_states;
                } /* Reading in the parameters for the prior */
                if (INPUT_VERBOSE) {
                    System.out.format("%d-component mixture distribution via a multinomial distribution\n", dist.num_states);
                    if (t == DIST_MIXTURE_PRIOR)
                       System. out.format("Dirichlet prior with pseudo-counts %f\n", pcount);
                }

                dist.subdist = new Distribution **[1];
                dist.subdist[0] = new Distribution *[dist.num_states];
                for (int i = 0; i < dist.num_states; i++)
                    dist.subdist[0][i] = ReadDistribution(input);
                break;
            case DIST_ALIAS_MIXTURE:
            case DIST_ALIAS_MIXTURE_PRIOR:
                /* Mixture of distributions of the same type (shortcut) */
                param1 = readFile.read_long();
                if (param1 < 1) {
                    System.err.format("Error on line %d: the number of mixture components must be at least 1.  Aborting.\n",
                            line_number);
                    System.exit(-1);
                }

                dist = new Distribution(DIST_BERNOULLI, param1, 1);

                if (t == DIST_ALIAS_MIXTURE_PRIOR) { /* Reading in the parameters for the prior */
                    pcount = readFile.read_double();
                    if (pcount < 0.0) {
                        System.err.format("Error on line %d: the pseudo-counts for Dirichlet prior for mixing probabilities of a mixture must be non-negative.  Setting them to zero.\n",
                                line_number);
                        pcount = 0.0;
                    }
                    for (int i = 0; i < dist.num_states; i++)
                        dist.pcount_single[i] = pcount;
                    dist.pcount = pcount * dist.num_states;
                } /* Reading in the parameters for the prior */
                if (INPUT_VERBOSE) {
                   System. out.format("%d-component mixture distribution defined via a multinomial distribution\n", dist.num_states);
                    if (t == DIST_ALIAS_MIXTURE_PRIOR)
                        System.  out.format("Dirichlet prior with pseudo-counts %f\n", pcount);
                }
                dist.subdist = new Distribution **[1];
                dist.subdist[0] = new Distribution *[dist.num_states];
                dist.subdist[0][0] = ReadDistribution(input);
                for (int i = 1; i < dist.num_states; i++)
                    dist.subdist[0][i] = dist.subdist[0][0].copy();
                break;
            case DIST_CLMIXTURE:
                /* Tree-dependent mixture */
                /* First parameter -- number of variables in each output node */
                param1 = readFile.read_long();
                if (param1 < 2) {
                    System.err.format("Error on line %d: the number of variables in the Chow-Liu tree must be at least 2.  Aborting\n",
                            line_number);
                    System.exit(-1);
                }

                /* !!! Assuming that each of the variables has the same range !!! */
                /* Second parameter -- number of values for each variable */
                param2 = readFile.read_long();
                if (param2 < 2) {
                    System.err.format("Error on line %d: the number of possible values for each variable in the Chow-Liu tree distribution must be at least 2. Aborting.\n",
                            line_number);
                    System.exit(-1);
                }
                dist = new Distribution(DIST_CHOWLIU, param2, param1);
                pcount = readFile.read_double();
                if (pcount < 0.0) {
                    System.err.format("Error on line %d: the pseudo-counts for Dirichlet prior for Chow-Liu tree parameters must be non-negative.  Setting them to zero.\n", line_number);
                    pcount = 0.0;
                }
                /* !!! No safety checking for the priors !!! */
                for (int i = 0; i < dist.dim; i++)
                    for (int j = 0; j < i; j++)
                        for (int b1 = 0; b1 < dist.num_states; b1++)
                            for (int b2 = 0; b2 < dist.num_states; b2++)
                                dist.pcount_bi[i][j][b1][b2] = pcount;
                /* !!! Shortcut for the computation of the marginals !!! */
                for (int i = 0; i < dist.dim; i++)
                    for (int b1 = 0; b1 < dist.num_states; b1++)
                        dist.pcount_uni[i][b1] = pcount * dist.num_states;
                dist.pcount = pcount * dist.num_states * dist.num_states;
                mdl_prior = readFile.read_double();
                dist.mdl_beta = mdl_prior;
                if (INPUT_VERBOSE) {
                    System.  out.format("%d-variable %d-component tree-structured mixture with Dirichlet prior with %f pseudo-counts and MDL prior with factor %f\n", dist.dim, dist.num_states, pcount, dist.mdl_beta);
                }
                dist.subdist = new Distribution **[dist.dim];
                for (int i = 0; i < dist.dim; i++)
                    dist.subdist[i] = new Distribution *[dist.num_states];
                for (int i = 0; i < dist.dim; i++)
                    for (int j = 0; j < dist.num_states; j++)
                        dist.subdist[i][j] = ReadDistribution(input);
                dim_index = new int[param1];
                for (int i = 0; i < param1; i++)
                    dim_index[i] = i;
                dist.UpdateIndex(dim_index);
                dim_index = null;
                break;
            case DIST_ALIAS_CLMIXTURE:
            case DIST_ALIAS_CLMIXTURE_PRIOR:
                /* Tree-structured mixture of distributions of the same type (shortcut) */
                /* First parameter -- number of variables in each output node */
                param1 = readFile.read_long();
                if (param1 < 2) {
                    System.err.format("Error on line %d: the number of variables in the Chow-Liu tree must be at least 2.  Aborting\n",
                            line_number);
                    System.exit(-1);
                }

                /* !!! Assuming that each of the variables has the same range !!! */
                /* Second parameter -- number of values for each variable */
                param2 = readFile.read_long();
                if (param2 < 2) {
                    System.err.format("Error on line %d: the number of possible values for each variable in the Chow-Liu tree distribution must be at least 2. Aborting.\n",
                            line_number);
                    System.exit(-1);
                }
                dist = new Distribution(DIST_CHOWLIU, param2, param1);
                if (t == DIST_ALIAS_CLMIXTURE) {
                    mdl_prior = readFile.read_double();
                    dist.mdl_beta = mdl_prior;
                    if (INPUT_VERBOSE) {
                        System.  out.format("%d-variable %d-component tree-structured mixture with MDL factor %0.6le\n",
                                dist.dim, dist.num_states, dist.mdl_beta);
                    }
                } else if (t == DIST_ALIAS_CLMIXTURE_PRIOR) {
                    pcount = readFile.read_double();
                    if (pcount < 0.0) {
                        System.err.format("Error on line %d: the pseudo-counts for Dirichlet prior for Chow-Liu tree parameters must be non-negative.  Setting them to zero.\n", line_number);
                        pcount = 0.0;
                    }
                    /* !!! No safety checking for the priors !!! */
                    for (int i = 0; i < dist.dim; i++)
                        for (int j = 0; j < i; j++)
                            for (int b1 = 0; b1 < dist.num_states; b1++)
                                for (int b2 = 0; b2 < dist.num_states; b2++)
                                    dist.pcount_bi[i][j][b1][b2] = pcount;
                    /* !!! Shortcut for the computation of the marginals !!! */
                    for (int i = 0; i < dist.dim; i++)
                        for (int b1 = 0; b1 < dist.num_states; b1++)
                            dist.pcount_uni[i][b1] = pcount * dist.num_states;
                    dist.pcount = pcount * dist.num_states * dist.num_states;
                    mdl_prior = readFile.read_double();
                    dist.mdl_beta = mdl_prior;
                    if (INPUT_VERBOSE) {
                       System. out.format("%d-variable %d-component tree-structured mixture with Dirichlet prior with %f pseudo-counts and MDL prior with factor %f\n", dist.dim, dist.num_states, pcount, dist.mdl_beta);
                    }
                }
                dist.subdist = new Distribution **[dist.dim];
                for (int i = 0; i < dist.dim; i++)
                    dist.subdist[i] = new Distribution *[dist.num_states];
                for (int j = 0; j < dist.num_states; j++)
                    dist.subdist[0][j] = ReadDistribution(input);
                for (int i = 1; i < dist.dim; i++)
                    for (int j = 0; j < dist.num_states; j++)
                        dist.subdist[i][j] = dist.subdist[0][j].copy();
                dim_index = new int[param1];
                for (int i = 0; i < param1; i++)
                    dim_index[i] = i;
                dist.UpdateIndex(dim_index);
                dim_index = null;
                break;
            case DIST_BERNOULLI:
            case DIST_BERNOULLI_PRIOR:
                /* One-dimensional Bernoulli */

                /* One parameter -- number of possible states */
                param1 = readFile.read_long();
                if (param1 < 1) {
                    System.err.format("Error on line %d: the number of states for a 1-d Bernoulli distribution must be at least 1.  Aborting.\n",
                            line_number);
                    System.exit(-1);
                }

                dist = new Distribution(DIST_BERNOULLI, param1, 1);

                if (t == DIST_BERNOULLI_PRIOR) { /* Reading in the parameters for the prior */
                    pcount = readFile.read_double();
                    if (pcount < 0.0) {
                        System.err.format("Error on line %d: the pseudo-counts for Dirichlet prior of 1-d multinomial distribution must be non-negative.  Setting them to zero.\n",
                                line_number);
                        pcount = 0.0;
                    }
                    for (int i = 0; i < dist.num_states; i++)
                        dist.pcount_single[i] = pcount;
                    dist.pcount = 0.0;
                    for (int i = 0; i < dist.num_states; i++)
                        dist.pcount += dist.pcount_single[i];
                } /* Reading in the parameters for the prior */

                for (int i = 0; i < dist.dim; i++)
                    dist.dim_index[i] = i;
                if (INPUT_VERBOSE) {
                    System.  out.format("Univariate %d-valued Bernoulli\n", dist.num_states);
                    if (t == DIST_BERNOULLI_PRIOR)
                        System.  out.format("Dirichlet prior with pseudo-counts %f\n", pcount);
                }
                break;
            case DIST_CONDBERNOULLI:
            case DIST_CONDBERNOULLIG:
            case DIST_CONDBERNOULLI_PRIOR:
            case DIST_CONDBERNOULLIG_PRIOR:
                /* Conditional Bernoulli */

                /* One parameter -- number of possible states */
                param1 = readFile.read_long();
                if (param1 < 1) {
                    System.err.format("Error on line %d: the number of states for a conditional Bernoulli distribution must be at least 1. Aborting.\n",
                            line_number);
                    System.exit(-1);
                }

                switch (t) {
                    case DIST_CONDBERNOULLI:
                    case DIST_CONDBERNOULLI_PRIOR:
                        dist = new Distribution(DIST_CONDBERNOULLI, param1, 1);
                        break;
                    case DIST_CONDBERNOULLIG:
                    case DIST_CONDBERNOULLIG_PRIOR:
                        dist = new Distribution(DIST_CONDBERNOULLIG, param1, 1);
                        break;
                }

                if (t == DIST_CONDBERNOULLI_PRIOR || t == DIST_CONDBERNOULLIG_PRIOR) { /* Reading in the parameters for the prior */
                    pcount = readFile.read_double();
                    if (pcount < 0.0) {
                        System.err.format("Error on line %d: the pseudo-counts for Dirichlet prior of 1-d conditional multinomial distribution must be non-negative.  Setting them to zero.\n",
                                line_number);
                        pcount = 0.0;
                    }
                    for (int i = 0; i < dist.num_states; i++)
                        for (int j = 0; j < dist.num_states; j++)
                            dist.pcount_uni[i][j] = pcount;
                    for (int i = 0; i < dist.num_states; i++)
                        dist.pcount_single[i] = pcount * dist.num_states;
                    dist.pcount = pcount * dist.num_states * dist.num_states;
                } /* Reading in the parameters for the prior */

                for (int i = 0; i < dist.dim; i++)
                    dist.dim_index[i] = i;
                if (INPUT_VERBOSE) {
                    if (t == DIST_CONDBERNOULLI)
                        System.  out.format("Univariate %d-valued conditional Bernoulli\n", dist.num_states);
                    else if (t == DIST_CONDBERNOULLIG)
                        System.   out.format("Univariate %d-valued conditional Bernoulli with averaged first sequence value\n", dist.num_states);
                    else if (t == DIST_CONDBERNOULLI_PRIOR)
                        System.    out.format("Univariate %d-valued conditional Bernoulli with Dirichlet prior with pseudo-counts %f\n", dist.num_states, pcount);
                    else if (t == DIST_CONDBERNOULLIG_PRIOR)
                        System.   out.format("Univariate %d-valued conditional Bernoulli with averaged first sequence value with Dirichlet prior with pseudo-counts %f\n", dist.num_states, pcount);
                }
                break;
            case DIST_UNICONDMVME:
                /* Conditional multi-valued MaxEnt */

                /* !!! Not to be read in directly !!! */
                break;
            case DIST_CHOWLIU:
            case DIST_CHOWLIU_MDL:
            case DIST_CHOWLIU_DIR_MDL:
                /* Chow-Liu tree */

                /* First parameter -- number of variables in each output node */
                param1 = readFile.read_long();
                if (param1 < 2) {
                    System.err.format("Error on line %d: the number of variables in the Chow-Liu tree must be at least 2.  Aborting\n",
                            line_number);
                    System.exit(-1);
                }

                /* !!! Assuming that each of the variables has the same range !!! */
                /* Second parameter -- number of values for each variable */
                param2 = readFile.read_long();
                if (param2 < 2) {
                    System.err.format("Error on line %d: the number of possible values for each variable in the Chow-Liu tree distribution must be at least 2. Aborting.",
                            line_number);
                    System.exit(-1);
                }

                dist = new Distribution(DIST_CHOWLIU, param2, param1);
                for (int i = 0; i < dist.dim; i++)
                    dist.dim_index[i] = i;

                /* Priors */
                if (t == DIST_CHOWLIU_MDL) {
                    mdl_prior = readFile.read_double();
                    dist.mdl_beta = mdl_prior;
                    if (INPUT_VERBOSE) {
                        System.  out.format("%d-variate %d-valued Chow-Liu tree with MDL prior with factor %0.6le\n",
                                dist.dim, dist.num_states, dist.mdl_beta);
                    }
                } else if (t == DIST_CHOWLIU_DIR_MDL) {
                    pcount = readFile.read_double();
                    if (pcount < 0.0) {
                        System.err.format("Error on line %d: the pseudo-counts for Dirichlet prior for Chow-Liu tree parameters must be non-negative.  Setting them to zero.\n", line_number);
                        pcount = 0.0;
                    }
                    /* !!! No safety checking for the priors !!! */
                    for (int i = 0; i < dist.dim; i++)
                        for (int j = 0; j < i; j++)
                            for (int b1 = 0; b1 < dist.num_states; b1++)
                                for (int b2 =  0; b2 < dist.num_states; b2++)
                                    dist.pcount_bi[i][j][b1][b2] = pcount;
                    /* !!! Shortcut for the computation of the marginals !!! */
                    for (int i = 0; i < dist.dim; i++)
                        for (int b1 = 0; b1 < dist.num_states; b1++)
                            dist.pcount_uni[i][b1] = pcount * dist.num_states;
                    dist.pcount = pcount * dist.num_states * dist.num_states;
                    mdl_prior = readFile.read_double();
                    dist.mdl_beta = mdl_prior;
                    if (INPUT_VERBOSE) {
                        System.   out.format("%d-variate %d-valued Chow-Liu tree with Dirichlet prior with %f pseudo-counts and MDL prior with factor %f\n", dist.dim, dist.num_states, pcount, dist.mdl_beta);
                    }
                } else if (t == DIST_CHOWLIU) {
                    if (INPUT_VERBOSE) {
                        System.   out.format("%d-variate %d-valued Chow-Liu tree\n", dist.dim, dist.num_states);
                    }
                    dist.mdl_beta = 0.0;
                }
                break;
            case DIST_CONDCHOWLIU:
            case DIST_CONDCHOWLIU_MDL:
                /* Conditional Chow-Liu tree */

                /* First parameter -- number of variables in each output node */
                param1 = readFile.read_long();
                if (param1 < 2) {
                    System.err.format("Error on line %d: the number of variables in the conditional Chow-Liu tree must be at least 2.  Aborting\n",
                            line_number);
                    System.exit(-1);
                }

                /* !!! Assuming that each of the variables has the same range !!! */
                /* Second parameter -- number of values for each variable */
                param2 = readFile.read_long();
                if (param2 < 2) {
                    System.err.format("Error on line %d: the number of possible values for each variable in the conditional Chow-Liu tree distribution must be at least 2. Aborting.",
                            line_number);
                    System.exit(-1);
                }

                dist = new Distribution(DIST_CONDCHOWLIU, param2, param1);
                for (int i = 0; i < dist.dim; i++)
                    dist.dim_index[i] = i;
                /* MDL factor */
                if (t == DIST_CONDCHOWLIU_MDL) {
                    mdl_prior = readFile.read_double();
                    dist.mdl_beta = mdl_prior;
                    if (INPUT_VERBOSE) {
                        System.  out.format("%d-variate %d-valued conditional Chow-Liu tree with MDL prior with factor %0.6le\n",
                                dist.dim, dist.num_states, dist.mdl_beta);
                    }
                } else if (t == DIST_CONDCHOWLIU) {
                    dist.mdl_beta = 0.0;
                    if (INPUT_VERBOSE) {
                        System.  out.format("%d-variate %d-valued conditional Chow-Liu tree\n", dist.dim, dist.num_states);
                    }
                }
                break;
            case DIST_ME_BIVAR:
                /* Binary bivariate MaxEnt distribution */

                /* First parameter -- number of stations */
                param1 = readFile.read_long();
                if (param1 < 2) {
                    System.err.format("Error on line %d: the number of nodes in the full bivariate MaxEnt distribution must be at least 2.  Aborting\n",
                            line_number);
                    System.exit(-1);
                }

                dist = new Distribution(t, 2, param1);
                for (int i = 0; i < dist.dim; i++)
                    dist.dim_index[i] = i;
                if (INPUT_VERBOSE) {
                    System.  out.format("%d-variate binary full bivariate MaxEnt\n", dist.dim);
                }
                break;
            case DIST_BN_ME_BIVAR:
                /* Bayesian network binary bivariate MaxEnt distribution */

                /* First parameter -- number of stations */
                param1 = readFile.read_long();
                if (param1 < 2) {
                    System.err.format("Error on line %d: the number of nodes in the BN bivariate MaxEnt distribution must be at least 2.  Aborting\n",
                            line_number);
                    System.exit(-1);
                }

                dist = new Distribution(t, 2, param1);
                mdl_prior = readFile.read_double();
                dist.mdl_beta = mdl_prior;
                for (int i = 0; i < dist.dim; i++)
                    dist.dim_index[i] = i;
                if (INPUT_VERBOSE) {
                    System.  out.format("%d-variate binary BN bivariate MaxEnt with MDL factor %0.6le\n", dist.dim, dist.mdl_beta);
                }
                break;
            case DIST_BN_CME_BIVAR:
                /* Bayesian network binary bivariate conditional MaxEnt distribution */

                /* First parameter -- number of stations */
                param1 = readFile.read_long();
                if (param1 < 2) {
                    System.err.format("Error on line %d: the number of nodes in the BN bivariate conditional MaxEnt distribution must be at least 2.  Aborting\n",
                            line_number);
                    System.exit(-1);
                }

                dist = new Distribution(t, 2, param1);
                mdl_prior = readFile.read_double();
                dist.mdl_beta = mdl_prior;
                for (int i = 0; i < dist.dim; i++)
                    dist.dim_index[i] = i;
                if (INPUT_VERBOSE) {
                    System.  out.format("%d-variate binary BN bivariate conditional MaxEnt with MDL factor %0.6le\n", dist.dim, dist.mdl_beta);
                }
                break;
            case DIST_DELTAEXP:
                /* Number of components */
                param1 = readFile.read_long();
                if (param1 < 2) {
                    System.err.format("Error on line %d: the number of components of the delta-exponential distribution must be at least 2. Aborting.\n",
                            line_number);
                    System.exit(-1);
                }

                dist = new Distribution(t, param1, 1);
                for (int i = 0; i < dist.dim; i++)
                    dist.dim_index[i] = i;
                if (INPUT_VERBOSE) {
                    System.  out.format("Univariate %d-component delta-exponential mixture\n", dist.num_states);
                }
                break;
            case DIST_DELTAGAMMA:
                /* Number of components */
                param1 = readFile.read_long();
                if (param1 < 2) {
                    System.err.format("Error on line %d: the number of components of the delta-gamma distribution must be at least 2. Aborting.",
                            line_number);
                    System.exit(-1);
                }

                dist = new Distribution(t, param1, 1);
                for (int i = 0; i < dist.dim; i++)
                    dist.dim_index[i] = i;
                if (INPUT_VERBOSE) {
                    System.  out.format("Univariate %d-component delta-gamma mixture\n", dist.num_states);
                }
                break;
            case DIST_DIRACDELTA:
                /* Dirac's delta function */

                /* Location of the delta function */
                mdl_prior = readFile.read_double();

                dist = new Distribution(t, 0, 1);

                /* Updating the value */
                dist.delta_value = mdl_prior;

                /* By default, the component index is 0 */
                dist.dim_index[0] = 0;
                if (INPUT_VERBOSE) {
                    System.   out.format("Dirac's delta function at %f\n", dist.delta_value);
                }
                break;
            case DIST_EXP:
                /* Number of components */
                dist = new Distribution(t, 0, 1);
                dist.dim_index[0] = 0;
                if (INPUT_VERBOSE) {
                    System.   out.format("Univariate exponential (geometric) distribution\n");
                }
                break;
            case DIST_GAMMA:
                dist = new Distribution(t, 0, 1);
                dist.dim_index[0] = 0;
                if (INPUT_VERBOSE) {
                    System.   out.format("Univariate gamma distribution\n");
                }
                break;
            case DIST_LOGNORMAL:
                dist = new Distribution(t, 0, 1);
                dist.dim_index[0] = 0;
                if (INPUT_VERBOSE) {
                    System.   out.format("Univariate log-normal distribution\n");
                }
                break;
            case DIST_NORMAL:
                /* Gaussian distribution */

                /* One parameter -- the dimension of the space */
                param1 = readFile.read_long();
                if (param1 < 1) {
                    System.err.format("Error on line %d: the dimensionality of the Gaussian distribution must be at least 1. Aborting.\n",
                            line_number);
                    System.exit(-1);
                }

                dist = new Distribution(t, 0, param1);
                for (int i = 0; i < dist.dim; i++)
                    dist.dim_index[i] = i;
                if (INPUT_VERBOSE) {
                    System.  out.format("%d-variate Gaussian\n", dist.dim);
                }
                break;
            case DIST_NORMALCHAIN:
                /* AR-Gaussian distribution */

                /* One parameter -- the dimension of the space */
                param1 = readFile.read_long();
                if (param1 < 1) {
                    System.err.format("Error on line %d: the dimensionality of the AR-Gaussian distribution must be at least 1. Aborting.\n",
                            line_number);
                    System.exit(-1);
                }

                dist = new Distribution(t, 0, param1);
                for (int i = 0; i < dist.dim; i++)
                    dist.dim_index[i] = i;
                if (INPUT_VERBOSE) {
                    System.   out.format("%d-variate AR-Gaussian\n", dist.dim);
                }
                break;
            case DIST_NORMALCL:
                /* Tree-structured Gaussian distribution */

                /* One parameter -- the dimension of the space */
                param1 = readFile.read_long();
                if (param1 < 1) {
                    System.err.format("Error on line %d: the dimensionality of the tree-structured Gaussian distribution must be at least 1. Aborting.\n",
                            line_number);
                    System.exit(-1);
                }

                dist = new Distribution(t, 0, param1);
                for (int i = 0; i < dist.dim; i++)
                    dist.dim_index[i] = i;
                if (INPUT_VERBOSE) {
                    System.  out.format("%d-variate tree-structured Gaussian\n", dist.dim);
                }
                break;
            case DIST_LOGISTIC:
                /* Logistic function */
                /* Two parameters -- number of states and the dimension of the input space */
                param1 = readFile.read_long();
                if (param1 < 2) {
                    System.err.format("Error on line %d: the number of states in a logistic distribution must be at least 2. Aborting.\n",
                            line_number);
                    System.exit(-1);
                }
                param2 = readFile.read_long();
                if (param2 < 1) {
                    System.err.format("Error on line %d: the dimensionality of a logistic distribution must be at least 1. Aborting.\n",
                            line_number);
                    System.exit(-1);
                }

                dist = new Distribution(t, param1, param2);
                if (INPUT_VERBOSE) {
                    System.  out.format("%d-dimensional %d-valued logistic\n", dist.dim, dist.num_states);
                }
                break;
            case DIST_TRANSLOGISTIC:
                /* Trans-logistic function */

                /* Two parameters -- number of states and the dimension of the input space */
                param1 = readFile.read_long();
                if (param1 < 2) {
                    System.err.format("Error on line %d: the number of states in a trans-logistic distribution must be at least 2. Aborting.\n",
                            line_number);
                    System.exit(-1);
                }
                param2 = readFile.read_long();
                if (param2 < 1) {
                    System.err.format("Error on line %d: the dimensionality of a trans-logistic distribution must be at least 1. Aborting.\n",
                            line_number);
                    System.exit(-1);
                }

                dist = new Distribution(t, param1, param2);
                if (INPUT_VERBOSE) {
                    System.   out.format("%d-dimensional %d-valued trans-logistic\n", dist.dim, dist.num_states);
                }
                break;
            default:
                System.err.format("Error on line %d: unknown distribution code: %d. Aborting.\n", line_number, t);
                System.exit(-1);
        }

        return (dist);
    }

    void  UpdateIndex(int[]index) {

        switch (type) {
            case DIST_FACTOR:
                /* Product distribution */
                //FIXME
                throw new UnsupportedOperationException("FIXME");
//                for (int i = 0; i < dim; i++)
//                    subdist[0][i].UpdateIndex( & index[i] );
//                break;
            case DIST_BERNOULLI:
                if (subdist!= null)
                    /* Mixture */
                    for (int i = 0; i < num_states; i++)
                        subdist[0][i].UpdateIndex(index);
                else
                    dim_index[0] = index[0];
                break;
            case DIST_CHOWLIU:
                if (subdist!=null)
                    /* Mixture */
                    //FIXME
                    throw new UnsupportedOperationException("FIXME");
//                for (int i = 0; i < dim; i++)
//                        for (int j = 0; j < num_states; j++)
//                            subdist[i][j].UpdateIndex( & index[i] );
    else
                for (int i = 0; i < dim; i++)
                    dim_index[i] = index[i];
                break;
            case DIST_CONDBERNOULLI:
            case DIST_CONDBERNOULLIG:
            case DIST_DELTAEXP:
            case DIST_DELTAGAMMA:
            case DIST_DIRACDELTA:
            case DIST_EXP:
            case DIST_GAMMA:
            case DIST_LOGNORMAL:
                dim_index[0] = index[0];
                break;
            case DIST_UNICONDMVME:
                break;
            case DIST_CONDCHOWLIU:
            case DIST_ME_BIVAR:
            case DIST_BN_ME_BIVAR:
            case DIST_BN_CME_BIVAR:
            case DIST_NORMAL:
            case DIST_NORMALCHAIN:
            case DIST_NORMALCL:
            case DIST_LOGISTIC:
            case DIST_TRANSLOGISTIC:
                for (int i = 0; i < dim; i++)
                    dim_index[i] = index[i];
                break;
            default:
        }

    }

    void ReadParameters(File input) throws IOException {
        /* Reads the values of parameters */

        ReadFile readFile = new ReadFile(input);
        switch (type) {
            case DIST_FACTOR:
                /* Reading parameters for sub-components */
                for (int i = 0; i < dim; i++)
                    subdist[0][i].ReadParameters(input);
                break;
            case DIST_BERNOULLI:
                /* Initializing Bernoulli parameters */
                if (subdist == null)
                    for (int i = 0; i < dim; i++)
                        dim_index[i] = readFile.read_long();
                for (int i = 0; i < num_states; i++)
                    state_prob[i] = readFile.read_double();
                for (int i = 0; i < num_states; i++)
                    log_state_prob[i] = log(state_prob[i]);
                if (subdist != null)
                    /* Reading mixture components */
                    for (int i = 0; i < dim; i++)
                        for (int j = 0; j < num_states; j++)
                            subdist[i][j].ReadParameters(input);
                break;
            case DIST_CONDBERNOULLI:
            case DIST_CONDBERNOULLIG:
                /* Conditional Bernoulli distribution */
                for (int i = 0; i < dim; i++)
                    dim_index[i] = readFile.read_long();

                /* First sequence entry probabilities */
                for (int i = 0; i < num_states; i++)
                    state_prob[i] = readFile.read_double();
                for (int i = 0; i < num_states; i++)
                    log_state_prob[i] = log(state_prob[i]);

                /* Conditional probabilities */
                for (int i = 0; i < num_states; i++)
                    for (int j = 0; j < num_states; j++)
                        cond_state_prob[i][j] = readFile.read_double();
                for (int i = 0; i < num_states; i++)
                    for (int j = 0; j < num_states; j++)
                        log_cond_state_prob[i][j] = log(cond_state_prob[i][j]);
                break;
            case DIST_UNICONDMVME:
                /* Multinomial MaxEnt distribution */

                /* Number of functions */
                dim = readFile.read_long();

                /* Number of features for each function */
                for (int i = 0; i < dim; i++)
                    num_features[i] = readFile.read_long();

                /* Indices of features for functions */
                for (int i = 0; i < dim; i++)
                    for (int j = 0; j < num_features[i]; j++)
                        feature_index[i][j] = readFile.read_long();

                /* Corresponding feature values */
                for (int i = 0; i < dim; i++)
                    for (int j = 0; j < num_features[i]; j++)
                        feature_value[i][j] = readFile.read_long();

                /* Weights */
                for (int i = 0; i < dim; i++)
                    lambda[i] = readFile.read_double();
                break;
            case DIST_CHOWLIU:
                /* Chow-Liu tree */
                if (subdist == null)
                    /* Not a mixture */
                    for (int i = 0; i < dim; i++)
                        dim_index[i] = readFile.read_long();

                /* Marginal probabilities */
                for (int i = 0; i < dim; i++)
                    for (int s = 0; s < num_states; s++)
                        ind_prob[i][s] = readFile.read_double();

                /* Number of edges */
                num_edges = readFile.read_long();

                /* Nodes on the edges */
                for (int i = 0; i < num_edges; i++) {
                    edge[i][0] = readFile.read_long();
                    edge[i][1] = readFile.read_long();
                }

                /* Probabilities on the edges */
                for (int i = 0; i < num_edges; i++)
                    for (int j = 0; j < num_states; j++)
                        for (int s = 0; s < num_states; s++)
                            edge_prob[i][j][s] = readFile.read_double();

                if (subdist!= null) { /* Mixture */
                    /* Reading mixing components */
                    for (int i = 0; i < dim; i++)
                        for (int j = 0; j < num_states; j++)
                            subdist[i][j].ReadParameters(input);

                    /* Computing the envelopes */
                    for (int i = 0; i < num_envelopes; i++)
                        envelope[i] = null;

                    num_envelopes = compute_envelopes_full();
                } /* Mixture */

                break;
            case DIST_CONDCHOWLIU:
                /* Conditional Chow-Liu tree */
                for (int i = 0; i < dim; i++)
                    dim_index[i] = readFile.read_long();

                /* Marginal probabilities */
                for (int i = 0; i < 2 * dim; i++)
                    for (int s = 0; s < num_states; s++)
                        ind_prob[i][s] = readFile.read_double();

                /* Number of edges */
                num_edges = readFile.read_long();

                /* Nodes on the edges */
                for (int i = 0; i < num_edges; i++) {
                    edge[i][0] = readFile.read_long();
                    edge[i][1] = readFile.read_long();
                }

                /* Probabilities on the edges */
                for (int i = 0; i < num_edges; i++)
                    for (int j = 0; j < num_states; j++)
                        for (int s = 0; s < num_states; s++)
                            edge_prob[i][j][s] = readFile.read_double();

                break;
            case DIST_ME_BIVAR:
                /* Binary full bivariate MaxEnt distribution */
                for (int i = 0; i < dim; i++)
                    dim_index[i] = readFile.read_long();

                /* Coefficients */
                for (int i = 0; i < dim; i++)
                    for (int j = 0; j < i + 1; j++)
                        sigma[i][j] = readFile.read_double();

                /* Normalization constant exponent */
                det = readFile.read_double();
                break;
            case DIST_BN_ME_BIVAR:
                /* Bayesian network binary bivariate MaxEnt distribution */
                for (int i = 0; i < dim; i++)
                    dim_index[i] = readFile.read_long();

                for (int i = 0; i < dim; i++)
                    num_features[i] = readFile.read_long();

                for (int i = 0; i < dim; i++)
                    for (int j = 0; j < num_features[i]; j++)
                        feature_index[i][j] = readFile.read_long();

                for (int i = 0; i < dim; i++)
                    for (int j = 0; j < num_features[i]; j++)
                        sigma[i][j] = readFile.read_double();


                for (int i = 0; i < dim; i++)
                    sim_order[i] = readFile.read_long();
                break;
            case DIST_BN_CME_BIVAR:
                /* Bayesian network binary bivariate MaxEnt conditional distribution */
                for (int i = 0; i < dim; i++)
                    dim_index[i] = readFile.read_long();

                for (int i = 0; i < dim; i++)
                    state_prob[i] = readFile.read_double();

                for (int i = 0; i < dim; i++)
                    num_features[i] = readFile.read_long();

                for (int i = 0; i < dim; i++)
                    for (int j = 0; j < num_features[i]; j++)
                        feature_index[i][j] = readFile.read_long();

                for (int i = 0; i < dim; i++)
                    for (int j = 0; j < num_features[i]; j++)
                        sigma[i][j] = readFile.read_double();


                for (int i = 0; i < dim; i++)
                    sim_order[i] = readFile.read_long();
                break;
            case DIST_DELTAEXP:
                /* Delta-exponential distribution */
                for (int i = 0; i < dim; i++)
                    dim_index[i] = readFile.read_long();

                /* Reading mixing probabilities */
                for (int i = 0; i < num_states; i++)
                    mix_prob[i] = readFile.read_double();

                /* Reading the parameters for the exponential components */
                for (int i = 0; i < num_states - 1; i++)
                    exp_param[i] = readFile.read_double();

                break;
            case DIST_DELTAGAMMA:
                /* Delta-gamma distribution */
                for (int i = 0; i < dim; i++)
                    dim_index[i] = readFile.read_long();

                /* Reading mixing probabilities */
                for (int i = 0; i < num_states; i++)
                    mix_prob[i] = readFile.read_double();

                /* Reading the parameters for the gamma components */
                for (int i = 0; i < num_states - 1; i++) {
                    gamma_param1[i] = readFile.read_double();
                    gamma_param2[i] = readFile.read_double();
                }
                break;
            case DIST_DIRACDELTA:
                /* Univariate Dirac delta */
                dim_index[0] = readFile.read_long();
                break;
            case DIST_EXP:
                /* Univariate exponential distribution */
                dim_index[0] = readFile.read_long();

                /* Reading the parameter */
                exp_param1 = readFile.read_double();

                break;
            case DIST_GAMMA:
                /* Univariate gamma distribution */
                dim_index[0] = readFile.read_long();

                /* Reading in two parameters */
                gamma1 = readFile.read_double();
                gamma2 = readFile.read_double();
                break;
            case DIST_LOGNORMAL:
                /* Univariate gamma distribution */
                dim_index[0] = readFile.read_long();

                /* Reading in two parameters */
                log_normal1 = readFile.read_double();
                log_normal2 = readFile.read_double();
                break;
            case DIST_NORMAL:
                /* Normal distribution */
                for (int i = 0; i < dim; i++)
                    dim_index[i] = readFile.read_long();

                /* Reading in the mean vector */
                for (int i = 0; i < dim; i++)
                    mu[i] = readFile.read_double();

                /* Reading in the covariance matrix */
                for (int i = 0; i < dim; i++)
                    for (int j = 0; j < dim; j++)
                        sigma[i][j] = readFile.read_double();

                /* Need to find the inverse and the determinant */
                find_inv(sigma, dim, inv_sigma);
                det = find_det(sigma, dim);
                break;
            case DIST_NORMALCHAIN:
                /* Normal distribution */
                for (int i = 0; i < dim; i++)
                    dim_index[i] = readFile.read_long();

                /* Reading in the first state mean vector */
                for (int i = 0; i < dim; i++)
                    first_mu[i] = readFile.read_double();

                /* Reading in the first state covariance matrix */
                for (int i = 0; i < dim; i++)
                    for (int j = 0; j < dim; j++)
                        first_sigma[i][j] = readFile.read_double();

                /* Need to find the inverse and the determinant */
                find_inv(first_sigma, dim, inv_first_sigma);
                first_det = find_det(first_sigma, dim);

                /* Reading in the transformation matrix */
                for (int i = 0; i < dim; i++)
                    for (int j = 0; j < dim; j++)
                        W[i][j] = readFile.read_double();

                /* Reading in translation vector */
                for (int i = 0; i < dim; i++)
                    mu[i] = readFile.read_double();

                /* Reading in noise covariance matrix */
                for (int i = 0; i < dim; i++)
                    for (int j = 0; j < dim; j++)
                        sigma[i][j] = readFile.read_double();

                /* Inverting */
                find_inv(sigma, dim, inv_sigma);
                det = find_det(sigma, dim);

                break;
            case DIST_NORMALCL:
                /* Normal distribution */
                for (int i = 0; i < dim; i++)
                    dim_index[i] = readFile.read_long();

                /* Reading in the mean vector */
                for (int i = 0; i < dim; i++)
                    mu[i] = readFile.read_double();

                /* Reading in the covariance matrix */
                for (int i = 0; i < dim; i++)
                    for (int j = 0; j < dim; j++)
                        sigma[i][j] = readFile.read_double();

                /* Need to find the inverse and the determinant */
                find_inv(sigma, dim, inv_sigma);
                det = find_det(sigma, dim);

                /* Number of edges */
                num_edges = readFile.read_long();

                /* Nodes on the edges */
                for (int i = 0; i < num_edges; i++) {
                    edge[i][0] = readFile.read_long();
                    edge[i][1] = readFile.read_long();
                }
                break;
            case DIST_LOGISTIC:
                /* Logistic distribution */
                for (int s = 0; s < num_states; s++) { /* Parameters by state */
                    /* Coefficients for the constant term */
                    lambda[s] = readFile.read_double();

                    /* Coefficients for the linear term */
                    for (int i = 0; i < dim; i++)
                        rho[s][i] = readFile.read_double();
                } /* Parameters by state */
                break;
            case DIST_TRANSLOGISTIC:
                /* Logistic distribution with transitions */
                for (int s = 0; s < num_states; s++) { /* Parameters by state */
                    /* Coefficients for the constant term */
                    lambda[s] = readFile.read_double();

                    /* Coefficients for the transition term */
                    for (int i = 0; i < num_states; i++)
                        sigma[i][s] = readFile.read_double();

                    /* Coefficients for the linear term */
                    for (int i = 0; i < dim; i++)
                        rho[s][i] = readFile.read_double();
                } /* Parameters by state */
                break;
            default:
                ;
        }
        return;
    }

    void  ReadParameters2(File input) throws IOException {
        /* Reads the values of parameters */

        /* Not reading in the dimension index */

        ReadFile readFile = new ReadFile(input);
        switch (type) {
            case DIST_FACTOR:
                /* Reading parameters for sub-components */
                for (int i = 0; i < dim; i++)
                    subdist[0][i].ReadParameters2(input);
                break;
            case DIST_BERNOULLI:
                /* Initializing Bernoulli parameters */
                for (int i = 0; i < num_states; i++)
                    state_prob[i] = readFile.read_double();
                for (int i = 0; i < num_states; i++)
                    log_state_prob[i] = log(state_prob[i]);
                if (subdist != null)
                    /* Mixture components */
                    for (int i = 0; i < dim; i++)
                        for (int j = 0; j < num_states; j++)
                            subdist[i][j].ReadParameters2(input);
                break;
            case DIST_CONDBERNOULLI:
            case DIST_CONDBERNOULLIG:
                /* Conditional Bernoulli distribution */
                /* First sequence entry probabilities */
                for (int i = 0; i < num_states; i++)
                    state_prob[i] = readFile.read_double();
                for (int i = 0; i < num_states; i++)
                    log_state_prob[i] = log(state_prob[i]);

                /* Conditional probabilities */
                for (int i = 0; i < num_states; i++)
                    for (int j = 0; j < num_states; j++)
                        cond_state_prob[i][j] = readFile.read_double();
                for (int i = 0; i < num_states; i++)
                    for (int j = 0; j < num_states; j++)
                        log_cond_state_prob[i][j] = log(cond_state_prob[i][j]);
                break;
            case DIST_UNICONDMVME:
                /* Multinomial MaxEnt distribution */

                /* Number of functions */
                dim = readFile.read_long();

                /* Number of features for each function */
                for (int i = 0; i < dim; i++)
                    num_features[i] = readFile.read_long();

                /* Indices of features for functions */
                for (int i = 0; i < dim; i++)
                    for (int j = 0; j < num_features[i]; j++)
                        feature_index[i][j] = readFile.read_long();

                /* Corresponding feature values */
                for (int i = 0; i < dim; i++)
                    for (int j = 0; j < num_features[i]; j++)
                        feature_value[i][j] = readFile.read_long();

                /* Weights */
                for (int i = 0; i < dim; i++)
                    lambda[i] = readFile.read_double();
                break;
            case DIST_CHOWLIU:
                /* Chow-Liu tree */
                /* Marginal probabilities */
                for (int i = 0; i < dim; i++)
                    for (int s = 0; s < num_states; s++)
                        ind_prob[i][s] = readFile.read_double();

                /* Number of edges */
                num_edges = readFile.read_long();

                /* Nodes on the edges */
                for (int i = 0; i < num_edges; i++) {
                    edge[i][0] = readFile.read_long();
                    edge[i][1] = readFile.read_long();
                }

                /* Probabilities on the edges */
                for (int i = 0; i < num_edges; i++)
                    for (int j = 0; j < num_states; j++)
                        for (int s = 0; s < num_states; s++)
                            edge_prob[i][j][s] = readFile.read_double();

                if (subdist != null) { /* Mixture */
                    /* Reading mixing components */
                    for (int i = 0; i < dim; i++)
                        for (int j = 0; j < num_states; j++)
                            subdist[i][j].ReadParameters2(input);

                    /* Computing the envelopes */
                    for (int i = 0; i < num_envelopes; i++)
                        envelope[i] = null;

                    num_envelopes = compute_envelopes_full();
                } /* Mixture */

                break;
            case DIST_CONDCHOWLIU:
                /* Conditional Chow-Liu tree */
                /* Marginal probabilities */
                for (int i = 0; i < 2 * dim; i++)
                    for (int s = 0; s < num_states; s++)
                        ind_prob[i][s] = readFile.read_double();

                /* Number of edges */
                num_edges = readFile.read_long();

                /* Nodes on the edges */
                for (int i = 0; i < num_edges; i++) {
                    edge[i][0] = readFile.read_long();
                    edge[i][1] = readFile.read_long();
                }

                /* Probabilities on the edges */
                for (int i = 0; i < num_edges; i++)
                    for (int j = 0; j < num_states; j++)
                        for (int s = 0; s < num_states; s++)
                            edge_prob[i][j][s] = readFile.read_double();

                break;
            case DIST_ME_BIVAR:
                /* Binary full bivariate MaxEnt distribution */
                /* Coefficients */
                for (int i = 0; i < dim; i++)
                    for (int j = 0; j < i + 1; j++)
                        sigma[i][j] = readFile.read_double();

                /* Normalization constant exponent */
                det = readFile.read_double();
                break;
            case DIST_BN_ME_BIVAR:
                /* Bayesian network binary bivariate MaxEnt distribution */
                for (int i = 0; i < dim; i++)
                    num_features[i] = readFile.read_long();

                for (int i = 0; i < dim; i++)
                    for (int j = 0; j < num_features[i]; j++)
                        feature_index[i][j] = readFile.read_long();

                for (int i = 0; i < dim; i++)
                    for (int j = 0; j < num_features[i]; j++)
                        sigma[i][j] = readFile.read_double();


                for (int i = 0; i < dim; i++)
                    sim_order[i] = readFile.read_long();
                break;
            case DIST_BN_CME_BIVAR:
                /* Bayesian network binary bivariate MaxEnt conditional distribution */
                for (int i = 0; i < dim; i++)
                    state_prob[i] = readFile.read_double();

                for (int i = 0; i < dim; i++)
                    num_features[i] = readFile.read_long();

                for (int i = 0; i < dim; i++)
                    for (int j = 0; j < num_features[i]; j++)
                        feature_index[i][j] = readFile.read_long();

                for (int i = 0; i < dim; i++)
                    for (int j = 0; j < num_features[i]; j++)
                        sigma[i][j] = readFile.read_double();


                for (int i = 0; i < dim; i++)
                    sim_order[i] = readFile.read_long();
                break;
            case DIST_DELTAEXP:
                /* Delta-exponential distribution */
                /* Reading mixing probabilities */
                for (int i = 0; i < num_states; i++)
                    mix_prob[i] = readFile.read_double();

                /* Reading the parameters for the exponential components */
                for (int i = 0; i < num_states - 1; i++)
                    exp_param[i] = readFile.read_double();

                break;
            case DIST_DELTAGAMMA:
                /* Delta-gamma distribution */
                /* Reading mixing probabilities */
                for (int i = 0; i < num_states; i++)
                    mix_prob[i] = readFile.read_double();

                /* Reading the parameters for the gamma components */
                for (int i = 0; i < num_states - 1; i++) {
                    gamma_param1[i] = readFile.read_double();
                    gamma_param2[i] = readFile.read_double();
                }
                break;
            case DIST_DIRACDELTA:
                break;
            case DIST_EXP:
                /* Univariate exponential distribution */
                /* Reading the parameter */
                exp_param1 = readFile.read_double();

                break;
            case DIST_GAMMA:
                /* Univariate gamma distribution */
                /* Reading in two parameters */
                gamma1 = readFile.read_double();
                gamma2 = readFile.read_double();
                break;
            case DIST_LOGNORMAL:
                /* Univariate gamma distribution */
                /* Reading in two parameters */
                log_normal1 = readFile.read_double();
                log_normal2 = readFile.read_double();
                break;
            case DIST_NORMAL:
                /* Normal distribution */
                /* Reading in the mean vector */
                for (int i = 0; i < dim; i++)
                    mu[i] = readFile.read_double();

                /* Reading in the covariance matrix */
                for (int i = 0; i < dim; i++)
                    for (int j = 0; j < dim; j++)
                        sigma[i][j] = readFile.read_double();

                /* Need to find the inverse and the determinant */
                find_inv(sigma, dim, inv_sigma);
                det = find_det(sigma, dim);
                break;
            case DIST_NORMALCHAIN:
                /* Normal distribution */
                /* Reading in the first state mean vector */
                for (int i = 0; i < dim; i++)
                    first_mu[i] = readFile.read_double();

                /* Reading in the first state covariance matrix */
                for (int i = 0; i < dim; i++)
                    for (int j = 0; j < dim; j++)
                        first_sigma[i][j] = readFile.read_double();

                /* Need to find the inverse and the determinant */
                find_inv(first_sigma, dim, inv_first_sigma);
                first_det = find_det(first_sigma, dim);

                /* Reading in the transformation matrix */
                for (int i = 0; i < dim; i++)
                    for (int j = 0; j < dim; j++)
                        W[i][j] = readFile.read_double();

                /* Reading in translation vector */
                for (int i = 0; i < dim; i++)
                    mu[i] = readFile.read_double();

                /* Reading in noise covariance matrix */
                for (int i = 0; i < dim; i++)
                    for (int j = 0; j < dim; j++)
                        sigma[i][j] = readFile.read_double();

                /* Inverting */
                find_inv(sigma, dim, inv_sigma);
                det = find_det(sigma, dim);

                break;
            case DIST_NORMALCL:
                /* Normal distribution with tree-structured covariance matrix */
                /* Reading in the mean vector */
                for (int i = 0; i < dim; i++)
                    mu[i] = readFile.read_double();

                /* Reading in the covariance matrix */
                for (int i = 0; i < dim; i++)
                    for (int j = 0; j < dim; j++)
                        sigma[i][j] = readFile.read_double();

                /* Need to find the inverse and the determinant */
                find_inv(sigma, dim, inv_sigma);
                det = find_det(sigma, dim);

                /* Number of edges */
                num_edges = readFile.read_long();

                /* Nodes on the edges */
                for (int i = 0; i < num_edges; i++) {
                    edge[i][0] = readFile.read_long();
                    edge[i][1] = readFile.read_long();
                }
                break;
            case DIST_LOGISTIC:
                /* Logistic distribution */
                for (int s = 0; s < num_states; s++) { /* Parameters by state */
                    /* Coefficients for the constant term */
                    lambda[s] = readFile.read_double();

                    /* Coefficients for the linear term */
                    for (int i = 0; i < dim; i++)
                        rho[s][i] = readFile.read_double();
                } /* Parameters by state */
                break;
            case DIST_TRANSLOGISTIC:
                /* Logistic distribution with transitions */
                for (int s = 0; s < num_states; s++) { /* Parameters by state */
                    /* Coefficients for the constant term */
                    lambda[s] = readFile.read_double();

                    /* Coefficients for the transition term */
                    for (int i = 0; i < num_states; i++)
                        sigma[i][s] = readFile.read_double();

                    /* Coefficients for the linear term */
                    for (int i = 0; i < dim; i++)
                        rho[s][i] = readFile.read_double();
                } /* Parameters by state */
                break;
            default:
        }
    }

    void AllocateEMStructures(Data data, Data input) {

        switch (type) {
            case DIST_CONDBERNOULLI:
            case DIST_CONDBERNOULLIG:
            case DIST_TRANSLOGISTIC:
                for (int i = 0; i < num_states; i++)
                    for (int j = 0; j < num_states; j++) {
                        joint_prob[i][j] = new double[data.num_seqs][];
                        for (int n = 0; n < data.num_seqs; n++)
                            joint_prob[i][j][n] = new double[data.sequence[n].seq_length];
                    }

                for (int i = 0; i < num_states; i++) {
                    log_bkd_update[i] = new double[data.num_seqs][];
                    for (int n = 0; n < data.num_seqs; n++)
                        log_bkd_update[i][n] = new double[data.sequence[n].seq_length];
                }

                break;
            default:
        }
    }

    void AllocateForwardPassStructures(Data data, Data input) {

        /* Allocating array of prosterior univariate probabilities */
        for (int i = 0; i < num_states; i++) {
            uni_prob[0][i] = new double[data.num_seqs][];
            for (int n = 0; n < data.num_seqs; n++)
                uni_prob[0][i][n] = new double[data.sequence[n].seq_length];
        }

        /* Allocating inverse scaling factors */
        log_upd_scale = new double[data.num_seqs][];
        for (int n = 0; n < data.num_seqs; n++)
            log_upd_scale[n] = new double[data.sequence[n].seq_length];

        switch (type) {
            case DIST_BERNOULLI:
                break;
            case DIST_CONDBERNOULLI:
            case DIST_CONDBERNOULLIG:
                /* Allocating scaled forward pass updates */
                for (int i = 0; i < num_states; i++) { /* For each state */
                    log_fwd_update[i] = new double[data.num_seqs][];
                    for (int n = 0; n < data.num_seqs; n++)
                        log_fwd_update[i][n] = new double[data.sequence[n].seq_length];
                } /* For each state */
                break;
            case DIST_LOGISTIC:
                /* Allocating the array of probabilities P(S_nt|S_n,t-1,X_nt) */
                for (int i = 0; i < num_states; i++) {
                    log_p_tr[i][0] = new double[input.num_seqs][];
                    for (int n = 0; n < input.num_seqs; n++)
                        log_p_tr[i][0][n] = new double[input.sequence[n].seq_length];
                }

                /* Allocating the array of log-unnormalized P(S_nt|S_n,t-1,X_nt) */
                for (int i = 1; i < num_states; i++) {
                    log_un_prob[i][0] = new double[input.num_seqs][];
                    for (int n = 0; n < input.num_seqs; n++)
                        log_un_prob[i][0][n] = new double[input.sequence[n].seq_length];
                }
                break;
            case DIST_TRANSLOGISTIC:
                /* Allocating scaled forward pass updates */
                for (int i = 0; i < num_states; i++) { /* For each state */
                    log_fwd_update[i] = new double *[data.num_seqs];
                    for (int n = 0; n < data.num_seqs; n++)
                        log_fwd_update[i][n] = new double[data.sequence[n].seq_length];
                } /* For each state */

                /* Allocating the array of probabilities P(S_nt|S_n,t-1,X_nt) */
                for (int i = 0; i < num_states; i++)
                    for (int j = 0; j < num_states; j++) {
                        log_p_tr[i][j] = new double[input.num_seqs][];
                        for (int n = 0; n < input.num_seqs; n++)
                            /* For each sequence */
                            log_p_tr[i][j][n] = new double[input.sequence[n].seq_length];
                    }

                /* Allocating the array of log-unnormalized P(S_nt|S_n,t-1,X_nt) */
                for (int i = 1; i < num_states; i++)
                    for (int j = 0; j < num_states; j++) {
                        log_un_prob[i][j] = new double[input.num_seqs][];
                        for (int n = 0; n < input.num_seqs; n++)
                            /* For each sequence */
                            log_un_prob[i][j][n] = new double[input.sequence[n].seq_length];
                    }
                break;
            default:
        }

    }

    void CalculateTransitionValuesSequence(Sequence input, int n, int start_index) {

        /* Temporary variable(s) */
        double sum;
        double max;

        /* Calculating P(S_t|S_{t-1}, X_t ) */

        /* !!! Hack speeding up the calculation of transition probabilities !!! */
        switch (type) {
            case DIST_LOGISTIC:
                for (int t = start_index; t < input.seq_length; t++) {
                    /* Calculating P(S_nt|X_nt) */
                    max = 0.0;
                    for (int i = 1; i < num_states; i++) {
                        log_un_prob[i][0][n][t] = log_un_probLogistic(input.entry[t], i, -1);
                        if (max < log_un_prob[i][0][n][t])
                            max = log_un_prob[i][0][n][t];
                    }

                    sum = exp(-max);
                    for (int i = 1; i < num_states; i++)
                        sum += exp(log_un_prob[i][0][n][t] - max);

                    log_p_tr[0][0][n][t] = -max - log(sum);
                    for (int i = 1; i < num_states; i++)
                        log_p_tr[i][0][n][t] = log_un_prob[i][0][n][t] - max - log(sum);
                }

                break;
            case DIST_TRANSLOGISTIC:
                if (start_index == 0) {
                    /* Calculating P(S_n1|X_n1) */
                    max = 0.0;
                    for (int i = 1; i < num_states; i++) {
                        log_un_prob[i][0][n][0] = log_un_probLogistic(input.entry[0], i, -1);
                        if (max < log_un_prob[i][0][n][0])
                            max = log_un_prob[i][0][n][0];
                    }

                    sum = exp(-max);
                    for (int i = 1; i < num_states; i++)
                        sum += exp(log_un_prob[i][0][n][0] - max);

                    log_p_tr[0][0][n][0] = -max - log(sum);
                    for (int i = 1; i < num_states; i++)
                        log_p_tr[i][0][n][0] = log_un_prob[i][0][n][0] - max - log(sum);

                    /* Updating start_index */
                    start_index = 1;
                }

                /* Calculating P(S_{nt}|S_{n,t-1},X_{nt}) */
                for (int t = start_index; t < input.seq_length; t++)
                    for (int j = 0; j < num_states; j++) {
                        max = 0.0;
                        for (int i = 1; i < num_states; i++) {
                            log_un_prob[i][j][n][t] = log_un_probLogistic(input.entry[t], i, j);
                            if (max < log_un_prob[i][j][n][t])
                                max = log_un_prob[i][j][n][t];
                        }

                        sum = exp(-max);
                        for (int i = 1; i < num_states; i++)
                            sum += exp(log_un_prob[i][j][n][t] - max);

                        log_p_tr[0][j][n][t] = -max - log(sum);
                        for (int i = 1; i < num_states; i++)
                            log_p_tr[i][j][n][t] = log_un_prob[i][j][n][t] - max - log(sum);
                    }
                break;
            default:
                ;
        }
    }

    void CalculateForwardUpdatesSequence(Sequence data, int n, int start_index, double[][][][] log_pb) {

        /* Temporary variable(s) */
        double sum;
        double[] value_contrib;
        double max_value, max_value2;

        value_contrib = new double[num_states];

        switch (type) {
            case DIST_BERNOULLI:
                for (int t = start_index; t < data.seq_length; t++) {
                    max_value = NEG_INF;
                    for (int i = 0; i < num_states; i++) {
                        uni_prob[0][i][n][t] = log_state_prob[i] + log_pb[0][i][n][t];
                        if (uni_prob[0][i][n][t] > max_value)
                            max_value = uni_prob[0][i][n][t];
                    }

                    /* Computing log of the rescaling constant */
                    sum = 0.0;
                    for (int i = 0; i < num_states; i++) {
                        uni_prob[0][i][n][t] -= max_value;
                        sum += exp(uni_prob[0][i][n][t]);
                    }
                    for (int i = 0; i < num_states; i++)
                        uni_prob[0][i][n][t] = exp(uni_prob[0][i][n][t] - log(sum));
                    log_upd_scale[n][t] = -max_value - log(sum);
                }
                break;
            case DIST_CONDBERNOULLI:
            case DIST_CONDBERNOULLIG:
                if (start_index == 0) {
                    /* First alpha */
                    max_value = NEG_INF;
                    for (int i = 0; i < num_states; i++) {
                        log_fwd_update[i][n][0] = log_state_prob[i] + log_pb[0][i][n][0];
                        if (log_fwd_update[i][n][0] > max_value)
                            max_value = log_fwd_update[i][n][0];
                    }

                    /* Log of the rescaling constant */
                    sum = 0.0;
                    for (int i = 0; i < num_states; i++) {
                        log_fwd_update[i][n][0] -= max_value;
                        sum += exp(log_fwd_update[i][n][0]);
                    }

                    for (int i = 0; i < num_states; i++)
                        log_fwd_update[i][n][0] -= log(sum);
                    log_upd_scale[n][0] = -max_value - log(sum);

                    /* Updating start_index */
                    start_index = 1;
                }

                /* Calculating the rest of alphas */
                for (int t = start_index; t < data.seq_length; t++) { /* Update for t-th vector */
                    max_value = NEG_INF;
                    for (int i = 0; i < num_states; i++) { /* Summing over S_t */
                        max_value2 = NEG_INF;
                        for (int j = 0; j < num_states; j++) {
                            value_contrib[j] = log_fwd_update[j][n][t - 1] + log_cond_state_prob[j][i];
                            if (value_contrib[j] > max_value2)
                                max_value2 = value_contrib[j];
                        }

                        sum = 0.0;
                        for (int j = 0; j < num_states; j++) {
                            value_contrib[j] -= max_value2;
                            sum += exp(value_contrib[j]);
                        }

                        log_fwd_update[i][n][t] = log(sum) + max_value2 + log_pb[0][i][n][t];
                        if (log_fwd_update[i][n][t] > max_value)
                            max_value = log_fwd_update[i][n][t];
                    } /* Summing over S_t */

                    sum = 0.0;
                    for (int i = 0; i < num_states; i++) {
                        log_fwd_update[i][n][t] -= max_value;
                        sum += exp(log_fwd_update[i][n][t]);
                    }

                    for (int i = 0; i < num_states; i++)
                        log_fwd_update[i][n][t] -= log(sum);
                    log_upd_scale[n][t] = -max_value - log(sum);
                }
                break;
            case DIST_LOGISTIC:
                for (int t = start_index; t < data.seq_length; t++) {
                    max_value = NEG_INF;
                    for (int i = 0; i < num_states; i++) {
                        uni_prob[0][i][n][t] = log_p_tr[i][0][n][t] + log_pb[0][i][n][t];
                        if (uni_prob[0][i][n][t] > max_value)
                            max_value = uni_prob[0][i][n][t];
                    }

                    /* Computing log of the rescaled constant */
                    sum = 0.0;
                    for (int i = 0; i < num_states; i++) {
                        uni_prob[0][i][n][t] -= max_value;
                        sum += exp(uni_prob[0][i][n][t]);
                    }

                    for (int i = 0; i < num_states; i++)
                        uni_prob[0][i][n][t] = exp(uni_prob[0][i][n][t] - log(sum));
                    log_upd_scale[n][t] = -max_value - log(sum);
                }
                break;
            case DIST_TRANSLOGISTIC:
                if (start_index == 0) {
                    /* First alpha */
                    max_value = NEG_INF;
                    for (int i = 0; i < num_states; i++) {
                        log_fwd_update[i][n][0] = log_p_tr[i][0][n][0] + log_pb[0][i][n][0];
                        if (log_fwd_update[i][n][0] > max_value)
                            max_value = log_fwd_update[i][n][0];
                    }

                    /* Log of the rescaling constant */
                    sum = 0.0;
                    for (int i = 0; i < num_states; i++) {
                        log_fwd_update[i][n][0] -= max_value;
                        sum += exp(log_fwd_update[i][n][0]);
                    }

                    for (int i = 0; i < num_states; i++)
                        log_fwd_update[i][n][0] -= log(sum);
                    log_upd_scale[n][0] = -max_value - log(sum);

                    /* Updating start_index */
                    start_index = 1;
                }

                /* Calculating the rest of alphas */
                for (int t = start_index; t < data.seq_length; t++) { /* Update for t-th vector */
                    max_value = NEG_INF;
                    for (int i = 0; i < num_states; i++) { /* Summing over S_t */
                        max_value2 = NEG_INF;
                        for (int j = 0; j < num_states; j++) {
                            value_contrib[j] = log_fwd_update[j][n][t - 1] + log_p_tr[i][j][n][t];
                            if (value_contrib[j] > max_value2)
                                max_value2 = value_contrib[j];
                        }

                        sum = 0.0;
                        for (int j = 0; j < num_states; j++) {
                            value_contrib[j] -= max_value2;
                            sum += exp(value_contrib[j]);
                        }

                        log_fwd_update[i][n][t] = log(sum) + max_value2 + log_pb[0][i][n][t];
                        if (log_fwd_update[i][n][t] > max_value)
                            max_value = log_fwd_update[i][n][t];
                    } /* Summing over S_t */

                    sum = 0.0;
                    for (int i = 0; i < num_states; i++) {
                        log_fwd_update[i][n][t] -= max_value;
                        sum += exp(log_fwd_update[i][n][t]);
                    }

                    for (int i = 0; i < num_states; i++)
                        log_fwd_update[i][n][t] -= log(sum);
                    log_upd_scale[n][t] = -max_value - log(sum);
                } /* Update for t-th vector */
                break;
            default:
                ;
        }

        value_contrib = null;

    }

    void CalculateBackwardUpdatesSequence(Sequence data, int n, double[][][][] log_pb) {

        /* Temporary variable(s) */
        double[] value_contrib;
        double max_value;
        double sum;

        value_contrib = new double[num_states];

        switch (type) {
            case DIST_BERNOULLI:
                break;
            case DIST_CONDBERNOULLI:
            case DIST_CONDBERNOULLIG:
                /* Initializing beta_T */
                for (int i = 0; i < num_states; i++)
                    log_bkd_update[i][n][data.seq_length - 1] = log_upd_scale[n][data.seq_length - 1];

                /* Calculating remaining betas */
                for (int t = data.seq_length - 2; t >= 0; t--) {
                    for (int i = 0; i < num_states; i++) { /* Summing over S_t */
                        max_value = NEG_INF;
                        for (int j = 0; j < num_states; j++) {
                            value_contrib[j] = log_bkd_update[j][n][t + 1] + log_cond_state_prob[i][j] + log_pb[0][j][n][t + 1];
                            if (value_contrib[j] > max_value)
                                max_value = value_contrib[j];
                        }

                        sum = 0.0;
                        for (int j = 0; j < num_states; j++) {
                            value_contrib[j] -= max_value;
                            sum += exp(value_contrib[j]);
                        }

                        log_bkd_update[i][n][t] = log(sum) + max_value + log_upd_scale[n][t];
                    } /* Summing over S_t */
                }
                break;
            case DIST_LOGISTIC:
                break;
            case DIST_TRANSLOGISTIC:
                /* Initializing beta_T */
                for (int i = 0; i < num_states; i++)
                    log_bkd_update[i][n][data.seq_length - 1] = log_upd_scale[n][data.seq_length - 1];

                /* Calculating remaining betas */
                for (int t = data.seq_length - 2; t >= 0; t--) {
                    for (int i = 0; i < num_states; i++) { /* Summing over S_t */
                        max_value = NEG_INF;
                        for (int j = 0; j < num_states; j++) {
                            value_contrib[j] = log_bkd_update[j][n][t + 1] + log_p_tr[j][i][n][t + 1] + log_pb[0][j][n][t + 1];
                            if (value_contrib[j] > max_value)
                                max_value = value_contrib[j];
                        }

                        sum = 0.0;
                        for (int j = 0; j < num_states; j++) {
                            value_contrib[j] -= max_value;
                            sum += exp(value_contrib[j]);
                        }

                        log_bkd_update[i][n][t] = log(sum) + max_value + log_upd_scale[n][t];
                    } /* Summing over S_t */
                }
                break;
            default:
                ;
        }

        value_contrib = null;

    }

    void CalculateSummaries(Data data, double[][][][] log_pb) {

        switch (type) {
            case DIST_BERNOULLI:
                /* Univariate probabilities */
                break;
            case DIST_CONDBERNOULLI:
            case DIST_CONDBERNOULLIG:
                /* Joint probabilities */
                for (int i = 0; i < num_states; i++)
                    for (int j = 0; j < num_states; j++)
                        for (int n = 0; n < data.num_seqs; n++)
                            for (int t = 1; t < data.sequence[n].seq_length; t++)
                                joint_prob[i][j][n][t] = exp(log_fwd_update[j][n][t - 1] + log_bkd_update[i][n][t] + log_pb[0][i][n][t] + log_cond_state_prob[j][i]);

                /* Univariate probabilities */
                for (int i = 0; i < num_states; i++)
                    for (int n = 0; n < data.num_seqs; n++)
                        for (int t = 0; t < data.sequence[n].seq_length; t++)
                            uni_prob[0][i][n][t] = exp(log_fwd_update[i][n][t] + log_bkd_update[i][n][t] - log_upd_scale[n][t]);

                break;
            case DIST_LOGISTIC:
                /* Univariate probabilities */
                break;
            case DIST_TRANSLOGISTIC:
                /* Joint probabilities */
                for (int i = 0; i < num_states; i++)
                    for (int j = 0; j < num_states; j++)
                        for (int n = 0; n < data.num_seqs; n++)
                            for (int t = 1; t < data.sequence[n].seq_length; t++)
                                joint_prob[i][j][n][t] = exp(log_fwd_update[j][n][t - 1] + log_bkd_update[i][n][t] + log_pb[0][i][n][t] + log_p_tr[i][j][n][t]);

                /* Univariate probabilities */
                for (int i = 0; i < num_states; i++)
                    for (int n = 0; n < data.num_seqs; n++)
                        for (int t = 0; t < data.sequence[n].seq_length; t++)
                            uni_prob[0][i][n][t] = exp(log_fwd_update[i][n][t] + log_bkd_update[i][n][t] - log_upd_scale[n][t]);
                break;
            default:
                ;
        }

    }

    void DeallocateEMStructures(Data data, Data input) {

        switch (type) {
            case DIST_CONDBERNOULLI:
            case DIST_CONDBERNOULLIG:
            case DIST_TRANSLOGISTIC:
                for (int i = 0; i < num_states; i++)
                    for (int j = 0; j < num_states; j++) {
                        for (int n = 0; n < data.num_seqs; n++)
                            joint_prob[i][j][n] = null;
                        joint_prob[i][j] = null;
                    }

                for (int i = 0; i < num_states; i++) { /* For each state */
                    for (int n = 0; n < data.num_seqs; n++)
                        log_bkd_update[i][n] = null;
                    log_bkd_update[i] = null;
                } /* For each state */
                break;
        }

    }


    void DeallocateForwardPassStructures(Data data, Data input) {

        /* Deallocating univariate posterior probabilities */
        for (int i = 0; i < num_states; i++) {
            for (int n = 0; n < data.num_seqs; n++)
                uni_prob[0][i][n] = null;
            uni_prob[0][i] = null;
        }

        for (int n = 0; n < data.num_seqs; n++)
            log_upd_scale[n] = null;
        log_upd_scale = null;

        switch (type) {
            case DIST_BERNOULLI:
                break;
            case DIST_CONDBERNOULLI:
            case DIST_CONDBERNOULLIG:
                for (int i = 0; i < num_states; i++) { /* For each state */
                    for (int n = 0; n < data.num_seqs; n++)
                        log_fwd_update[i][n] = null;
                    log_fwd_update[i] = null;
                } /* For each state */
                break;
            case DIST_LOGISTIC:
                /* Normalized transition probability */
                for (int i = 0; i < num_states; i++) {
                    for (int n = 0; n < input.num_seqs; n++)
                        log_p_tr[i][0][n] = null;
                    log_p_tr[i][0] = null;
                }

                /* Log-unnormalized transition probability */
                for (int i = 1; i < num_states; i++) {
                    for (int n = 0; n < input.num_seqs; n++)
                        log_un_prob[i][0][n] = null;
                    log_un_prob[i][0] = null;
                }
                break;
            case DIST_TRANSLOGISTIC:
                for (int i = 0; i < num_states; i++) { /* For each state */
                    for (int n = 0; n < data.num_seqs; n++)
                        log_fwd_update[i][n] = null;
                    log_fwd_update[i] = null;
                } /* For each state */

                /* Normalized transition probability */
                for (int i = 0; i < num_states; i++)
                    for (int j = 0; j < num_states; j++) {
                        for (int n = 0; n < input.num_seqs; n++)
                            log_p_tr[i][j][n] = null;
                        log_p_tr[i][j] = null;
                    }

                /* Log-unnormalized transition probability */
                for (int i = 1; i < num_states; i++)
                    for (int j = 0; j < num_states; j++) {
                        for (int n = 0; n < input.num_seqs; n++)
                            log_un_prob[i][j][n] = null;
                        log_un_prob[i][j] = null;
                    }
                break;
            default:
        }

    }


    void  Simulate(DataPoint sim, DataPoint prev_datum) {
        /* Simulating an observation according to distribution */

        int[]generated;
        double[]prob_vector;
        int num_gibbs_sims;

        /* Temporary variables */
        int temp_value;
        int[]ddata;
        double[]rdata;
        double sum;
        double temp;

        switch (type) {
            case DIST_FACTOR:
                for (int i = 0; i < dim; i++)
                    subdist[0][i].Simulate(sim, prev_datum);
                break;
            case DIST_BERNOULLI:
                if (dim == 1) {
                    int i = generateBernoulli(state_prob, num_states);
                    if (subdist != null)
                        /* Mixture */
                        subdist[0][i].Simulate(sim, prev_datum);
                    else
                        /* Not a mixture */
                        sim.ddata[dim_index[0]] = i;
                } else {
                    System.err.format("Multi-dimensional Bernoulli distribution simulation is not yet implemented!\n");
                    System.exit(-1);
                }
                break;
            case DIST_CONDBERNOULLI:
            case DIST_CONDBERNOULLIG:
                if (dim == 1)
                    if (prev_datum)
                        /* Not the first sequence entry */
                        sim.ddata[dim_index[0]] = generateBernoulli(cond_state_prob[prev_datum.ddata[dim_index[0]]], num_states);
                    else
                        /* First sequence entry */
                        sim.ddata[dim_index[0]] = generateBernoulli(state_prob, num_states);
                else {
                    System.err.format("Multi-dimensional conditional Bernoulli distribution simulation is not yet implemented!\n");
                    System.exit(-1);
                }
                break;
            case DIST_UNICONDMVME:
                prob_vector = new double[num_states];
                for (int i = 0; i < num_states; i++)
                    prob_vector[i] = 0.0;

                /* Computing probabilities for all outcomes */
                for (int i = 0; i < dim; i++) {
                    temp = 1.0;
                    for (int j = 1; j < num_features[i]; j++)
                        if (feature_index[i][j] < prev_datum.ddim) { /* Categorical feature */
                            if (feature_value[i][j] != prev_datum.ddata[feature_index[i][j]]) {
                                temp = 0.0;
                                j = num_features[i];
                            }
                        } /* Categorical feature */ else
                            /* Real-valued feature */
                            temp *= prev_datum.rdata[feature_index[i][j] - prev_datum.ddim];

                    prob_vector[feature_value[i][0]] += temp;
                }

                temp = prob_vector[0];
                for (int i = 1; i < num_states; i++)
                    if (prob_vector[i] > temp)
                        temp = prob_vector[i];

                sum = 0.0;
                for (int i = 0; i < num_states; i++)
                    sum += exp(prob_vector[i] - temp);

                for (int i = 0; i < num_states; i++)
                    prob_vector[i] = exp(prob_vector[i] - temp) / sum;

                sim.ddata[0] = generateBernoulli(prob_vector, num_states);

                prob_vector = null;
                break;
            case DIST_CHOWLIU:
    /* When generating the model, assuming that postprocessing step is run as well.
       Generating the nodes in the order in which they appear in the edges. */

                ddata = new int[dim];

                generated = new int[dim];
                for (int i = 0; i < dim; i++)
                    generated[i] = 0;

                prob_vector = new double[num_states];

                for (int e = 0; e < num_edges; e++) {
	/* Two possibilities:
	   both nodes need to be simulated, or just one */
                    if (!generated[edge[e][0]] && !generated[edge[e][1]]) { /* Both nodes need to be simulated */
                        /* Simulating the first node of the edge */
                        ddata[edge[e][0]] = generateBernoulli(ind_prob[edge[e][0]], num_states);
                        generated[edge[e][0]] = 1;

                        /* Simulating the second node of the edge */
                        /* Creating the probability distribution first */
                        for (int i = 0; i < num_states; i++)
                            prob_vector[i] = edge_prob[e][ddata[edge[e][0]]][i] / ind_prob[edge[e][0]][ddata[edge[e][0]]];

                        ddata[edge[e][1]] = generateBernoulli(prob_vector, num_states);
                        generated[edge[e][1]] = 1;
                    } /* Both nodes need to be simulated */ else { /* Only one node needs to be simulated */
                        if (!generated[edge[e][0]]) { /* Simulating the first node of the edge given the second */
                            /* Creating the probability distribution first */
                            for (int i = 0; i < num_states; i++)
                                prob_vector[i] = edge_prob[e][i][ddata[edge[e][1]]] / ind_prob[edge[e][1]][ddata[edge[e][1]]];

                            ddata[edge[e][0]] = generateBernoulli(prob_vector, num_states);
                            generated[edge[e][0]] = 1;
                        } /* Simulating the first node of the edge given the second */

                        if (!generated[edge[e][1]]) { /* Simulating the second node of the edge given the first */
                            /* Creating the probability distribution first */
                            for (int i = 0; i < num_states; i++)
                                prob_vector[i] = edge_prob[e][ddata[edge[e][0]]][i] / ind_prob[edge[e][0]][ddata[edge[e][0]]];

                            ddata[edge[e][1]] = generateBernoulli(prob_vector, num_states);
                            generated[edge[e][1]] = 1;
                        } /* Simulating the second node of the edge given the first */
                    } /* Only one node needs to be simulated */
                }

                /* Making sure all nodes have been simulated */
                for (int i = 0; i < dim; i++)
                    if (!generated[i])
                        /* Simulating the node form individual probabilities */
                        ddata[i] = generateBernoulli(ind_prob[i], num_states);

                prob_vector = null;

                generated = null;

                for (int i = 0; i < dim; i++)
                    if (subdist!=null)
                        /* Mixture */
                        subdist[i][ddata[i]].Simulate(sim, prev_datum);
                    else
                        /* Not a mixture */
                        sim.ddata[dim_index[i]] = ddata[i];

                ddata = null;

                break;
            case DIST_CONDCHOWLIU:
    /* When generating the model, assuming that postprocessing step is run as well.
       Generating the nodes in the order in which they appear in the edges. */

                ddata = new int[dim];

                generated = new int[2 * dim];
                for (int i = 0; i < dim; i++) {
                    generated[i] = 0;
                    generated[i + dim] = 1;
                }

                prob_vector = new double[num_states];

                for (int e = 0; e < num_edges; e++) {
	/* Two possibilities:
	   both nodes need to be simulated, or just one */
                    if (!generated[edge[e][0]] && !generated[edge[e][1]]) { /* Both nodes need to be simulated */
                        /* Simulating the first node of the edge */
                        ddata[edge[e][0]] = generateBernoulli(ind_prob[edge[e][0]], num_states);
                        generated[edge[e][0]] = 1;

                        /* Simulating the second node of the edge */
                        /* Creating the probability distribution first */
                        for (int i = 0; i < num_states; i++)
                            prob_vector[i] = edge_prob[e][ddata[edge[e][0]]][i] /
                                    ind_prob[edge[e][0]][ddata[edge[e][0]]];

                        ddata[edge[e][1]] = generateBernoulli(prob_vector, num_states);
                        generated[edge[e][1]] = 1;
                    } /* Both nodes need to be simulated */ else { /* Only one node needs to be simulated */
                        if (!generated[edge[e][0]]) { /* Simulating the first node of the edge given the second */
                            /* Creating the probability distribution first */
                            if (edge[e][1] < dim)
                                temp_value = ddata[edge[e][1]];
                            else if (prev_datum)
                                temp_value = prev_datum.ddata[dim_index[edge[e][1] - dim]];
                            else
                                temp_value = missing_value((int) 0);

                            if (!is_missing(temp_value))
                                for (int i = 0; i < num_states; i++)
                                    prob_vector[i] = edge_prob[e][i][temp_value] / ind_prob[edge[e][1]][temp_value];
                            else
                                for (int i = 0; i < num_states; i++)
                                    prob_vector[i] = ind_prob[edge[e][0]][i];

                            ddata[edge[e][0]] = generateBernoulli(prob_vector, num_states);
                            generated[edge[e][0]] = 1;
                        } /* Simulating the first node of the edge given the second */

                        if (!generated[edge[e][1]]) { /* Simulating the second node of the edge given the first */
                            /* Creating the probability distribution first */
                            if (edge[e][0] < dim)
                                temp_value = ddata[edge[e][0]];
                            else if (prev_datum)
                                temp_value = prev_datum.ddata[dim_index[edge[e][0] - dim]];
                            else
                                temp_value = missing_value((int) 0);

                            if (!is_missing(temp_value))
                                for (int i = 0; i < num_states; i++)
                                    prob_vector[i] = edge_prob[e][temp_value][i] / ind_prob[edge[e][0]][temp_value];
                            else
                                for (int i = 0; i < num_states; i++)
                                    prob_vector[i] = ind_prob[edge[e][1]][i];

                            ddata[edge[e][1]] = generateBernoulli(prob_vector, num_states);
                            generated[edge[e][1]] = 1;
                        } /* Simulating the second node of the edge given the first */
                    } /* Only one node needs to be simulated */
                }

                /* Making sure all nodes have been simulated */
                for (int i = 0; i < dim; i++)
                    if (!generated[i])
                        /* Simulating the node form individual probabilities */
                        ddata[i] = generateBernoulli(ind_prob[i], num_states);

                prob_vector = null;

                generated = null;

                for (int i = 0; i < dim; i++)
                    sim.ddata[dim_index[i]] = ddata[i];
                ddata = null;

                break;
            case DIST_ME_BIVAR:
                /* Simulation by Gibbs sampling */
                if (is_sim_initialized)
                    /* Already performed burn-off */
                    num_gibbs_sims = GIBBS_LAG;
                else { /* Need to perform the burn-off first */
                    is_sim_initialized = 1;
                    last_sim = new int[dim];

                    /* Initializing at random */
                    for (int i = 0; i < dim; i++)
                        if  (Constants.drand48() > 0.5)
                            last_sim[i] = 1;
                        else
                            last_sim[i] = 0;

                    num_gibbs_sims = GIBBS_BURNIN;
                } /* Need to perform the burn-off first */

                for (int sim_index = 0; sim_index < num_gibbs_sims; sim_index++)
                    /* Performing Gibbs sampling */
                    for (int i = 0; i < dim; i++) { /* Resampling component i */
                        /* Calculating probability of outcome 0: 1/(exp(sum)+1) */
                        sum = 0.0;
                        for (int j = 0; j < i; j++)
                            if (last_sim[j] == 1)
                                sum += sigma[i][j];
                        sum += sigma[i][i];
                        for (int j = i + 1; j < dim; j++)
                            if (last_sim[j] == 1)
                                sum += sigma[j][i];

                        /* Sampling */
                        if  (Constants.drand48() < 1.0 / (exp(sum) + 1.0))
                            last_sim[i] = 0;
                        else
                            last_sim[i] = 1;
                    } /* Resampling component i */

                /* Copying the last sample into the output */
                for (int i = 0; i < dim; i++)
                    sim.ddata[dim_index[i]] = last_sim[i];

                break;
            case DIST_BN_ME_BIVAR:
                for (int i = 0; i < dim; i++) {
                    /* Computing the probability of outcome 0: 1/(exp(sum)+1) */
                    sum = sigma[sim_order[i]][0];
                    for (int j = 1; j < num_features[sim_order[i]]; j++)
                        if (sim.ddata[dim_index[feature_index[sim_order[i]][j]]] == 1)
                            sum += sigma[sim_order[i]][j];

                    /* Sampling */
                    if  (Constants.drand48() < 1.0 / (1.0 + exp(sum)))
                        sim.ddata[dim_index[sim_order[i]]] = 0;
                    else
                        sim.ddata[dim_index[sim_order[i]]] = 1;
                }
                break;
            case DIST_BN_CME_BIVAR:
                for (int i = 0; i < dim; i++)
                    if (!prev_datum) { /* First entry in a sequence -- using state_prob */
                        if  (Constants.drand48() < state_prob[i])
                            sim.ddata[dim_index[i]] = 1;
                        else
                            sim.ddata[dim_index[i]] = 0;
                    } /* First entry in a sequence -- using state_prob */ else {
                        /* Computing the probability of outcome 0: 1/(exp(sum)+1) */
                        sum = sigma[sim_order[i]][0];
                        for (int j = 1; j < num_features[sim_order[i]]; j++)
                            if (feature_index[sim_order[i]][j] < dim) {
                                if (sim.ddata[dim_index[feature_index[sim_order[i]][j]]] == 1)
                                    sum += sigma[sim_order[i]][j];
                            } else {
                                if (prev_datum.ddata[dim_index[feature_index[sim_order[i]][j] - dim]] == 1)
                                    sum += sigma[sim_order[i]][j];
                            }

                        /* Sampling */
                        if  (Constants.drand48() < 1.0 / (1.0 + exp(sum)))
                            sim.ddata[dim_index[sim_order[i]]] = 0;
                        else
                            sim.ddata[dim_index[sim_order[i]]] = 1;
                    }
                break;
            case DIST_DELTAEXP:
                i = generateBernoulli(mix_prob, num_states);
                if (i == 0)
                    sim.rdata[dim_index[0]] = 0.0;
                else
                    sim.rdata[dim_index[0]] = generateExponential(exp_param[i - 1]);
                break;
            case DIST_DELTAGAMMA:
                i = generateBernoulli(mix_prob, num_states);
                if (i == 0)
                    sim.rdata[dim_index[0]] = 0.0;
                else
                    sim.rdata[dim_index[0]] = generateGamma(gamma_param1[i - 1], gamma_param2[i - 1]);
                break;
            case DIST_DIRACDELTA:
                sim.rdata[dim_index[0]] = delta_value;
                break;
            case DIST_EXP:
                sim.rdata[dim_index[0]] = generateExponential(exp_param1);
                break;
            case DIST_GAMMA:
                sim.rdata[dim_index[0]] = generateGamma(gamma1, gamma2);
                break;
            case DIST_NORMAL:
                /* If Cholesky decomposition is not computed yet, doing so now */
                if (!chol_sigma) {
                    chol_sigma = new double *[dim];
                    for (int i = 0; i < dim; i++)
                        chol_sigma[i] = new double[dim];

                    cholesky(sigma, dim, chol_sigma);
                }

                rdata = new double[dim];
                SimulateNormal(mu, chol_sigma, rdata, dim);
                for (int i = 0; i < dim; i++)
                    sim.rdata[dim_index[i]] = rdata[i];
                rdata = null;
                break;
            case DIST_NORMALCHAIN:
                /* If Cholesky decomposition is not computed yet, doing so now */
                if (!chol_sigma) {
                    chol_sigma = new double *[dim];
                    for (int i = 0; i < dim; i++)
                        chol_sigma[i] = new double[dim];

                    cholesky(sigma, dim, chol_sigma);
                }
                if (!chol_first_sigma) {
                    chol_first_sigma = new double *[dim];
                    for (int i = 0; i < dim; i++)
                        chol_first_sigma[i] = new double[dim];

                    cholesky(first_sigma, dim, chol_first_sigma);
                }

                rdata = new double[dim];

                if (prev_datum) { /* Not the first observation in the sequence */

                    /* Allocating mean vector */
                    prob_vector = new double[dim];

                    /* Calculating the value for the mean vector */
                    for (int i = 0; i < dim; i++) {
                        prob_vector[i] = mu[i];
                        for (int e = 0; e < dim; e++)
                            prob_vector[i] += W[i][e] * prev_datum.rdata[dim_index[e]];
                    }

                    SimulateNormal(prob_vector, chol_sigma, rdata, dim);

                    prob_vector = null;
                } /* Not the first observation in the sequence */ else
                    /* First observation in the sequence */
                    SimulateNormal(first_mu, chol_first_sigma, rdata, dim);

                for (int i = 0; i < dim; i++)
                    sim.rdata[dim_index[i]] = rdata[i];
                rdata = null;

                break;
            case DIST_NORMALCL:
                /* !!! May be replaced by more efficient sampling !!! */
                /* If Cholesky decomposition is not computed yet, doing so now */
                if (!chol_sigma) {
                    chol_sigma = new double *[dim];
                    for (int i = 0; i < dim; i++)
                        chol_sigma[i] = new double[dim];

                    cholesky(sigma, dim, chol_sigma);
                }

                rdata = new double[dim];
                SimulateNormal(mu, chol_sigma, rdata, dim);
                for (int i = 0; i < dim; i++)
                    sim.rdata[dim_index[i]] = rdata[i];
                rdata = null;
                break;
            case DIST_LOGISTIC:
            case DIST_TRANSLOGISTIC:
                System.err.format("Simulation from the logistic data is not yet implemented!\n");
                System.exit(-1);
                break;
            default:
                ;
        }

    }

    void SimulateMissingEntries(DataPoint datum, DataPoint prev_datum) {
        /* Simulating an observation according to distribution */
        if (DISABLED)
            return;

        double[][] mult;
        double[] prob;
        int[] sim_value;

        /* Temporary variables */
        double sum;

        switch (type) {
            case DIST_FACTOR:
                for (int i = 0; i < dim; i++)
                    subdist[0][i].SimulateMissingEntries(datum, prev_datum);
                break;
            case DIST_CLMIXTURE:
                /* Tree-structured mixture */
                /* Computing the envelopes */

                /* Allocating the array of probabilities for the current value of the latent variable */
                prob = new double[num_states];

                /* Allocating additional variables */
                mult = new double [dim][];
                for (int i = 0; i < dim; i++)
                    mult[i] = new double[num_states];

                sim_value = new int[dim];

                /* Precomputing probability values for components */
                for (int i = 0; i < dim; i++)
                    for (int j = 0; j < num_states; j++)
                        mult[i][j] = subdist[i][j].prob(datum, prev_datum);

                for (int env = 0; env < num_envelopes; env++)
                    /* For each envelope */
                    if (envelope[env].num_nodes == 1) { /* Envelope contains exactly one variable. */
                        m1 = envelope[env].node[0];
                        sum = 0.0;
                        for (int b = 0; b < num_states; b++) { /* Summing over the latent variable */
                            prob[b] = ind_prob[m1][b] * mult[m1][b];
                            sum += prob[b];
                        } /* Summing over the latent variable */

                        /* Normalizing the probability */
                        for (int b = 0; b < num_states; b++)
                            prob[b] /= sum;

                        /* Simulating the value for the latent variable */
                        sim_value[m1] = generateBernoulli(prob, num_states);

                        /* Simulating missing values for the appropriate component */
                        subdist[m1][sim_value[m1]].SimulateMissingEntries(datum, prev_datum);
                    } /* Envelope contains exactly one missing variable. */ else { /* Envelope contains more than one variable. */
                        /* Propagating probabilities in a backward pass */
                        for (int e = envelope[env].num_edges - 1; e >= 0; e--) {
                            ei = envelope[env].edge[e];
                            i1 = edge[ei][0];
                            i2 = edge[ei][1];

                            if (envelope[env].node[e + 1] == i2) { /* i1 is the parent of i2 */
                                for (value1 = 0; value1 < num_states; value1++) { /* Computing the contribution to i1 with value value1 from child i2 */
                                    sum = 0.0;
                                    for (value2 = 0; value2 < num_states; value2++)
                                        sum += mult[i2][value2] * edge_prob[ei][value1][value2];
                                    mult[i1][value1] *= sum / ind_prob[i1][value1];
                                } /* Computing the contribution to i1 with value value1 from child i2 */
                            } /* i1 is the parent of i2 */ else { /* i2 is the parent of i1 */
                                for (value2 = 0; value2 < num_states; value2++) { /* Computing the contribution to i2 with value value2 from child i1 */
                                    sum = 0.0;
                                    for (value1 = 0; value1 < num_states; value1++)
                                        sum += mult[i1][value1] * edge_prob[ei][value1][value2];
                                    mult[i2][value2] *= sum / ind_prob[i2][value2];
                                } /* Computing the contribution to i2 with value value2 from child i1 */
                            } /* i2 is the parent of i1 */
                        }

                        /* Summing over the root of the envelope */
                        m1 = envelope[env].node[0];
                        sum = 0.0;
                        for (int b = 0; b < num_states; b++) { /* Summing over the latent variable */
                            prob[b] = mult[m1][b] * ind_prob[m1][b];
                            sum += prob[b];
                        } /* Summing over the latent variable */

                        /* Normalizing the probability */
                        for (int b = 0; b < num_states; b++)
                            prob[b] /= sum;

                        /* Simulating the value for the latent variable */
                        sim_value[m1] = generateBernoulli(prob, num_states);

                        /* Simulating missing values for the appropriate component */
                        subdist[m1][sim_value[m1]].SimulateMissingEntries(datum, prev_datum);

                        /* Backward pass */
                        for (int e = 0; e < envelope[env].num_edges; e++) {
                            ei = envelope[env].edge[e];
                            i1 = edge[ei][0];
                            i2 = edge[ei][1];

                            if (envelope[env].node[e + 1] == i2) { /* i1 is the parent of i2 */
                                sum = 0.0;
                                for (int b = 0; b < num_states; b++) {
                                    prob[b] = mult[i2][b] * edge_prob[ei][sim_value[i1]][b];
                                    sum += prob[b];
                                }

                                /* Normalizing */
                                for (int b = 0; b < num_states; b++)
                                    prob[b] /= sum;

                                /* Simulating value for a latent variable */
                                sim_value[i2] = generateBernoulli(prob, num_states);

                                /* Simulating missing values for the appropriate component */
                                subdist[i2][sim_value[i2]].SimulateMissingEntries(datum, prev_datum);
                            } /* i1 is the parent of i2 */ else { /* i2 is the parent of i1 */
                                sum = 0.0;
                                for (int b = 0; b < num_states; b++) {
                                    prob[b] = mult[i1][b] * edge_prob[ei][b][sim_value[i2]];
                                    sum += prob[b];
                                }

                                /* Normalizing */
                                for (int b = 0; b < num_states; b++)
                                    prob[b] /= sum;

                                /* Simulating value for a latent variable */
                                sim_value[i1] = generateBernoulli(prob, num_states);

                                /* Simulating missing values for the appropriate component */
                                subdist[i1][sim_value[i1]].SimulateMissingEntries(datum, prev_datum);
                            } /* i2 is the parent of i1 */
                        }

                    } /* Envelope contains more than one variable. */

                /* Deallocating */
                sim_value = null;

                for (int i = 0; i < dim; i++)
                    mult[i] = null;
                mult = null;

                prob = null;

                break;
            case DIST_BERNOULLI:
                if (dim == 1) {
                    if (subdist) {
                        prob = new double[num_states];

                        /* Computing posterior probabilities */
                        sum = 0.0;
                        for (int b = 0; b < num_states; b++) {
                            prob[b] = state_prob[b] * subdist[0][b].prob(datum, prev_datum);
                            sum += prob[b];
                        }
                        for (int b = 0; b < num_states; b++)
                            prob[b] /= sum;

                        /* Picking a mixture component */
                        i = generateBernoulli(prob, num_states);

                        /* Simulating a data point from the component */
                        subdist[0][i].SimulateMissingEntries(datum, prev_datum);

                        prob = null;
                    } else if (is_missing(datum.ddata[dim_index[0]]))
                        datum.ddata[dim_index[0]] = generateBernoulli(state_prob, num_states);
                }
                break;
            case DIST_CONDBERNOULLI:
            case DIST_CONDBERNOULLIG:
                if (dim == 1)
                    if (is_missing(datum.ddata[dim_index[0]])) {
                        if (prev_datum)
                            /* Not the first sequence entry */
                            datum.ddata[dim_index[0]] = generateBernoulli(cond_state_prob[prev_datum.ddata[dim_index[0]]], num_states);
                        else
                            /* First sequence entry */
                            datum.ddata[dim_index[0]] = generateBernoulli(state_prob, num_states);
                    }
                break;
            case DIST_UNICONDMVME:
                /* !!! Do not know what to do with this yet !!! */
                break;
            case DIST_CHOWLIU:
                /* Allocating the array of probabilities for the current value of the latent variable */
                prob = new double[num_states];

                /* Allocating additional variables */
                mult = new double *[dim];
                for (int i = 0; i < dim; i++)
                    mult[i] = new double[num_states];

                /* Computing the envelopes */
                num_envelopes = compute_envelopes(datum, mult);

                for (int env = 0; env < num_envelopes; env++)
                    /* For each envelope */
                    if (is_missing(datum.ddata[dim_index[envelope[env].node[0]]])) { /* Envelope contains missing values */
                        if (envelope[env].num_nodes == 1) { /* Envelope contains exactly one missing variable. */
                            m1 = envelope[env].node[0];
                            sum = 0.0;
                            for (int b = 0; b < num_states; b++) { /* Summing over the missing variable */
                                prob[b] = ind_prob[m1][b] * mult[m1][b];
                                sum += prob[b];
                            } /* Summing over the missing variable */

                            /* Normalizing the probability */
                            for (int b = 0; b < num_states; b++)
                                prob[b] /= sum;

                            /* Simulating the value */
                            datum.ddata[dim_index[m1]] = generateBernoulli(prob, num_states);
                        } /* Envelope contains exactly one missing variable. */ else { /* Envelope contains more than one missing variable. */
                            for (int i = 0; i < envelope[env].num_nodes; i++)
                                for (int b = 0; b < num_states; b++)
                                    mult[envelope[env].node[i]][b] = 1.0;

                            /* Propagating probabilities in a backward pass */
                            for (int e = envelope[env].num_edges - 1; e >= 0; e--) {
                                ei = envelope[env].edge[e];
                                i1 = edge[ei][0];
                                i2 = edge[ei][1];

                                /* Both nodes are missing. */
                                if (envelope[env].node[e + 1] == i2) { /* i1 is the parent of i2 */
                                    for (value1 = 0; value1 < num_states; value1++) { /* Computing the contribution to i1 with value value1 from child i2 */
                                        sum = 0.0;
                                        for (value2 = 0; value2 < num_states; value2++)
                                            sum += mult[i2][value2] * edge_prob[ei][value1][value2];
                                        mult[i1][value1] *= sum / ind_prob[i1][value1];
                                    } /* Computing the contribution to i1 with value value1 from child i2 */
                                } /* i1 is the parent of i2 */ else { /* i2 is the parent of i1 */
                                    for (value2 = 0; value2 < num_states; value2++) { /* Computing the contribution to i2 with value value2 from child i1 */
                                        sum = 0.0;
                                        for (value1 = 0; value1 < num_states; value1++)
                                            sum += mult[i1][value1] * edge_prob[ei][value1][value2];
                                        mult[i2][value2] *= sum / ind_prob[i2][value2];
                                    } /* Computing the contribution to i2 with value value2 from child i1 */
                                } /* i2 is the parent of i1 */
                            }

                            /* Summing over the root of the envelope */
                            m1 = envelope[env].node[0];
                            sum = 0.0;
                            for (int b = 0; b < num_states; b++) {
                                prob[b] = mult[m1][b] * ind_prob[m1][b];
                                sum += prob[b];
                            }
                            /* Normalizing */
                            for (int b = 0; b < num_states; b++)
                                prob[b] /= sum;

                            /* Simulating the value */
                            datum.ddata[dim_index[m1]] = generateBernoulli(prob, num_states);

                            /* Backward pass */
                            for (int e = 0; e < envelope[env].num_edges; e++) {
                                ei = envelope[env].edge[e];
                                i1 = edge[ei][0];
                                i2 = edge[ei][1];

                                if (is_missing(datum.ddata[i1])) { /* First node is missing */
                                    sum = 0.0;
                                    for (int b = 0; b < num_states; b++) {
                                        prob[b] = mult[i1][b] * edge_prob[ei][b][datum.ddata[dim_index[i2]]];
                                        sum += prob[b];
                                    }
                                    for (int b = 0; b < num_states; b++)
                                        prob[b] /= sum;

                                    datum.ddata[dim_index[i1]] = generateBernoulli(prob, num_states);
                                } /* First node is missing */ else if (is_missing(datum.ddata[i2])) { /* Second node is missing */
                                    sum = 0.0;
                                    for (int b = 0; b < num_states; b++) {
                                        prob[b] = mult[i2][b] * edge_prob[ei][b][datum.ddata[dim_index[i1]]];
                                        sum += prob[b];
                                    }
                                    for (int b = 0; b < num_states; b++)
                                        prob[b] /= sum;

                                    datum.ddata[dim_index[i2]] = generateBernoulli(prob, num_states);
                                } /* Second node is missing */
                            }
                        } /* Envelope contains more than one missing variable. */
                    } /* Envelope contains missing values */

                /* Deallocating */
                for (int i = 0; i < dim; i++)
                    mult[i] = null;
                mult = null;
                for (int env = 0; env < num_envelopes; env++)
                    envelope[env] = null;

                /* Deallocating temporary array of probabilities */
                prob = null;

                break;

            case DIST_CONDCHOWLIU:
                /* Do not know how to do that yet */
                break;
            if (BLAH) {
    /* When generating the model, assuming that postprocessing step is run as well.
       Generating the nodes in the order in which they appear in the edges. */

                ddata = new int[dim];

                generated = new int[2 * dim];
                for (int i = 0; i < dim; i++) {
                    generated[i] = 1;
                    generated[i + dim] = 0;
                }

                prob_vector = new double[num_states];

                for (int e = 0; e < num_edges; e++) {
	/* Two possibilities:
	   both nodes need to be simulated, or just one */
                    if (!generated[edge[e][0]] && !generated[edge[e][1]]) { /* Both nodes need to be simulated */
                        /* Simulating the first node of the edge */
                        ddata[edge[e][0] - dim] = generateBernoulli(ind_prob[edge[e][0]], num_states);
                        generated[edge[e][0]] = 1;

                        /* Simulating the second node of the edge */
                        /* Creating the probability distribution first */
                        for (int i = 0; i < num_states; i++)
                            prob_vector[i] = edge_prob[e][ddata[edge[e][0] - dim]][i] /
                                    ind_prob[edge[e][0] - dim][ddata[edge[e][0] - dim]];

                        ddata[edge[e][1] - dim] = generateBernoulli(prob_vector, num_states);
                        generated[edge[e][1]] = 1;
                    } /* Both nodes need to be simulated */ else { /* Only one node needs to be simulated */
                        if (!generated[edge[e][0]]) { /* Simulating the first node of the edge given the second */
                            /* Creating the probability distribution first */
                            if (edge[e][1] < dim)
                                if (prev_datum)
                                    temp_value = prev_datum.ddata[dim_index[edge[e][1]]];
                                else
                                    temp_value = missing_value((int) 0);
                            else
                                temp_value = ddata[edge[e][1] - dim];

                            if (!is_missing(temp_value))
                                for (int i = 0; i < num_states; i++)
                                    prob_vector[i] = edge_prob[e][i][temp_value] / ind_prob[edge[e][1]][temp_value];
                            else
                                for (int i = 0; i < num_states; i++)
                                    prob_vector[i] = ind_prob[edge[e][0]][i];

                            ddata[edge[e][0] - dim] = generateBernoulli(prob_vector, num_states);
                            generated[edge[e][0]] = 1;
                        } /* Simulating the first node of the edge given the second */

                        if (!generated[edge[e][1]]) { /* Simulating the second node of the edge given the first */
                            /* Creating the probability distribution first */
                            if (edge[e][0] < dim)
                                if (prev_datum)
                                    temp_value = prev_datum.ddata[dim_index[edge[e][0]]];
                                else
                                    temp_value = missing_value((int) 0);
                            else
                                temp_value = ddata[edge[e][0] - dim];

                            if (!is_missing(temp_value))
                                for (int i = 0; i < num_states; i++)
                                    prob_vector[i] = edge_prob[e][temp_value][i] / ind_prob[edge[e][0]][temp_value];
                            else
                                for (int i = 0; i < num_states; i++)
                                    prob_vector[i] = ind_prob[edge[e][1]][i];

                            ddata[edge[e][1] - dim] = generateBernoulli(prob_vector, num_states);
                            generated[edge[e][1]] = 1;
                        } /* Simulating the second node of the edge given the first */
                    } /* Only one node needs to be simulated */
                }

                /* Making sure all nodes have been simulated */
                for (int i = dim; i < 2 * dim; i++)
                    if (!generated[i])
                        /* Simulating the node form individual probabilities */
                        ddata[i - dim] = generateBernoulli(ind_prob[i], num_states);

                prob_vector = null;

                generated = null;

                for (int i = 0; i < dim; i++)
                    sim.ddata[dim_index[i]] = ddata[i];
                ddata = null;

                break;
            }
            case DIST_ME_BIVAR:
                /* Not yet implemented */
                break;
            if (BLAH) {
                /* Simulation by Gibbs sampling */
                if (is_sim_initialized)
                    /* Already performed burn-off */
                    num_gibbs_sims = GIBBS_LAG;
                else { /* Need to perform the burn-off first */
                    is_sim_initialized = 1;
                    last_sim = new int[dim];

                    /* Initializing at random */
                    for (int i = 0; i < dim; i++)
                        if  (Constants.drand48() > 0.5)
                            last_sim[i] = 1;
                        else
                            last_sim[i] = 0;

                    num_gibbs_sims = GIBBS_BURNIN;
                } /* Need to perform the burn-off first */

                for (int sim_index = 0; sim_index < num_gibbs_sims; sim_index++)
                    /* Performing Gibbs sampling */
                    for (int i = 0; i < dim; i++) { /* Resampling component i */
                        /* Calculating probability of outcome 0: 1/(exp(sum)+1) */
                        sum = 0.0;
                        for (int j = 0; j < i; j++)
                            if (last_sim[j] == 1)
                                sum += sigma[i][j];
                        sum += sigma[i][i];
                        for (int j = i + 1; j < dim; j++)
                            if (last_sim[j] == 1)
                                sum += sigma[j][i];

                        /* Sampling */
                        if  (Constants.drand48() < 1.0 / (exp(sum) + 1.0))
                            last_sim[i] = 0;
                        else
                            last_sim[i] = 1;
                    } /* Resampling component i */

                /* Copying the last sample into the output */
                for (int i = 0; i < dim; i++)
                    sim.ddata[dim_index[i]] = last_sim[i];
            }
            case DIST_BN_ME_BIVAR:
                /* Not yet implemented */
                break;
            if (BLAH) {
                for (int i = 0; i < dim; i++) {
                    /* Computing the probability of outcome 0: 1/(exp(sum)+1) */
                    sum = sigma[sim_order[i]][0];
                    for (int j = 1; j < num_features[sim_order[i]]; j++)
                        if (sim.ddata[dim_index[feature_index[sim_order[i]][j]]] == 1)
                            sum += sigma[sim_order[i]][j];

                    /* Sampling */
                    if  (Constants.drand48() < 1.0 / (1.0 + exp(sum)))
                        sim.ddata[dim_index[sim_order[i]]] = 0;
                    else
                        sim.ddata[dim_index[sim_order[i]]] = 1;
                }
                break;
            }
            case DIST_BN_CME_BIVAR:
                /* Do not know how to perform this */
                break;
            if (BLAH) {
                for (int i = 0; i < dim; i++)
                    if (!prev_datum) { /* First entry in a sequence -- using state_prob */
                        if  (Constants.drand48() < state_prob[i])
                            sim.ddata[dim_index[i]] = 1;
                        else
                            sim.ddata[dim_index[i]] = 0;
                    } /* First entry in a sequence -- using state_prob */ else {
                        /* Computing the probability of outcome 0: 1/(exp(sum)+1) */
                        sum = sigma[sim_order[i]][0];
                        for (int j = 1; j < num_features[sim_order[i]]; j++)
                            if (feature_index[sim_order[i]][j] < dim) {
                                if (sim.ddata[dim_index[feature_index[sim_order[i]][j]]] == 1)
                                    sum += sigma[sim_order[i]][j];
                            } else {
                                if (prev_datum.ddata[dim_index[feature_index[sim_order[i]][j] - dim]] == 1)
                                    sum += sigma[sim_order[i]][j];
                            }

                        /* Sampling */
                        if  (Constants.drand48() < 1.0 / (1.0 + exp(sum)))
                            sim.ddata[dim_index[sim_order[i]]] = 0;
                        else
                            sim.ddata[dim_index[sim_order[i]]] = 1;
                    }
                break;
                case DIST_DELTAEXP:
                    i = generateBernoulli(mix_prob, num_states);
                    if (i == 0)
                        sim.rdata[dim_index[0]] = 0.0;
                    else
                        sim.rdata[dim_index[0]] = generateExponential(exp_param[i - 1]);
                    break;
                case DIST_DELTAGAMMA:
                    i = generateBernoulli(mix_prob, num_states);
                    if (i == 0)
                        sim.rdata[dim_index[0]] = 0.0;
                    else
                        sim.rdata[dim_index[0]] = generateGamma(gamma_param1[i - 1], gamma_param2[i - 1]);
                    break;
                case DIST_DIRACDELTA:
                    sim.rdata[dim_index[0]] = delta_value;
                    break;
                case DIST_EXP:
                    sim.rdata[dim_index[0]] = generateExponential(exp_param1);
                    break;
                case DIST_GAMMA:
                    sim.rdata[dim_index[0]] = generateGamma(gamma1, gamma2);
                    break;
                case DIST_NORMAL:
                    /* If Cholesky decomposition is not computed yet, doing so now */
                    if (!chol_sigma) {
                        chol_sigma = new double *[dim];
                        for (int i = 0; i < dim; i++)
                            chol_sigma[i] = new double[dim];

                        cholesky(sigma, dim, chol_sigma);
                    }

                    rdata = new double[dim];
                    SimulateNormal(mu, chol_sigma, rdata, dim);
                    for (int i = 0; i < dim; i++)
                        sim.rdata[dim_index[i]] = rdata[i];
                    rdata = null;
                    break;
                case DIST_NORMALCHAIN:
                    /* If Cholesky decomposition is not computed yet, doing so now */
                    if (!chol_sigma) {
                        chol_sigma = new double *[dim];
                        for (int i = 0; i < dim; i++)
                            chol_sigma[i] = new double[dim];

                        cholesky(sigma, dim, chol_sigma);
                    }
                    if (!chol_first_sigma) {
                        chol_first_sigma = new double *[dim];
                        for (int i = 0; i < dim; i++)
                            chol_first_sigma[i] = new double[dim];

                        cholesky(first_sigma, dim, chol_first_sigma);
                    }

                    rdata = new double[dim];

                    if (prev_datum) { /* Not the first observation in the sequence */

                        /* Allocating mean vector */
                        prob_vector = new double[dim];

                        /* Calculating the value for the mean vector */
                        for (int i = 0; i < dim; i++) {
                            prob_vector[i] = mu[i];
                            for (int e = 0; e < dim; e++)
                                prob_vector[i] += W[i][e] * prev_datum.rdata[dim_index[e]];
                        }

                        SimulateNormal(prob_vector, chol_sigma, rdata, dim);

                        prob_vector = null;
                    } /* Not the first observation in the sequence */ else
                        /* First observation in the sequence */
                        SimulateNormal(first_mu, chol_first_sigma, rdata, dim);

                    for (int i = 0; i < dim; i++)
                        sim.rdata[dim_index[i]] = rdata[i];
                    rdata = null;

                    break;
                case DIST_NORMALCL:
                    /* !!! May be replaced by more efficient sampling !!! */
                    /* If Cholesky decomposition is not computed yet, doing so now */
                    if (!chol_sigma) {
                        chol_sigma = new double *[dim];
                        for (int i = 0; i < dim; i++)
                            chol_sigma[i] = new double[dim];

                        cholesky(sigma, dim, chol_sigma);
                    }

                    rdata = new double[dim];
                    SimulateNormal(mu, chol_sigma, rdata, dim);
                    for (int i = 0; i < dim; i++)
                        sim.rdata[dim_index[i]] = rdata[i];
                    rdata = null;
                    break;
                case DIST_LOGISTIC:
                case DIST_TRANSLOGISTIC:
                    System.err.format("Simulation from the logistic data is not yet implemented!\n");
                    System.exit(-1);
                    break;
            }
            default:
        }

    }

    void UpdateEmissionBernoulli(Data data,double[][]prob_s){
        /* !!! Univariate Bernoulli !!! */

        double[][][]mix_comp_prob;

        double[]state_count;
        double weight_missing=0.0;

        double[]current_prob;    // Probability weight for the current example

        /* Flags */
        //  int leave_unchanged=0;  // Underflow flag

        /* Temporary variables */
        double sum;
        double[]value_contrib;
        double max_value;

        state_count=new double[num_states];
        current_prob=new double[num_states];
        value_contrib=new double[num_states];

        for (int b =0;b<num_states; b++)
            state_count[b]=0.0;

        if(subdist != null)
        { /* Mixture components present */
            /* Allocating P(mixing component|data) */
            mix_comp_prob=new double[num_states][][];
            for (int b =0;b<num_states; b++)
            {
                mix_comp_prob[b]=new double[data.num_seqs][];
                for (int n =0;n<data.num_seqs;n++)
                    mix_comp_prob[b][n]=new double[data.sequence[n].seq_length];
            }
        } /* Mixture components present */

        for (int n =0;n<data.num_seqs;n++)
            for (int t =0;t<data.sequence[n].seq_length;t++)
                if(subdist!=null)
                { /* Mixture components */
                    /* !!! Assuming univariate multinomial !!! */
                    max_value=NEG_INF;
                    for (int b =0;b<num_states; b++)
                    {
                        value_contrib[b]=log_state_prob[b];
                        if(t==0)
                            value_contrib[b]+=subdist[0][b].log_prob(data.sequence[n].entry[t],null);
                        else
                            value_contrib[b]+=subdist[0][b].log_prob(data.sequence[n].entry[t],data.sequence[n].entry[t-1]);
                        if(value_contrib[b]>max_value)
                            max_value=value_contrib[b];
                    }

                    sum=0.0;
                    for (int b =0;b<num_states; b++)
                    {
                        mix_comp_prob[b][n][t]=exp(value_contrib[b]-max_value);
                        sum+=mix_comp_prob[b][n][t];
                    }

                    for (int b =0;b<num_states; b++)
                    {
                        mix_comp_prob[b][n][t]/=sum;
                        mix_comp_prob[b][n][t]*=prob_s[n][t];
                        state_count[b]+=mix_comp_prob[b][n][t];
                    }
                } /* Mixture components */
                else
                if(!is_missing(data.sequence[n].entry[t].ddata[dim_index[0]]))
                    /* Not missing */
                    state_count[data.sequence[n].entry[t].ddata[dim_index[0]]]+=prob_s[n][t];
                else
                    /* Missing */
                    weight_missing+=prob_s[n][t];

        if(subdist !=null)
            for (int b =0;b<num_states; b++)
                /* Passing unnormalized weight to the mixture components */
                if(abs(state_count[b])>COMP_THRESHOLD)
                    /* Updating mixture components */
                    subdist[0][b].UpdateEmissionParameters(data,mix_comp_prob[b],state_count[b]);

        /* Incorporating the prior and normalizing the probabilities */
        sum=0.0;
        for (int b =0;b<num_states; b++)
        {
            state_count[b]+=pcount_single[b];
            state_prob[b]=state_count[b]+state_prob[b]*weight_missing;
            sum+=state_prob[b];
        }

        for (int b =0;b<num_states; b++)
            state_prob[b]/=sum;
        for (int b =0;b<num_states; b++)
            log_state_prob[b]=log(state_prob[b]);

        value_contrib = null;
        current_prob = null;
        state_count = null;

        if(subdist!=null)
        { /* Mixture components present */

            /* Deallocating P(mixing component|X,S) */
            for (int b =0;b<num_states; b++)
            {
                for (int n =0;n<data.num_seqs;n++)
                    mix_comp_prob[b][n] = null;
                mix_comp_prob[b] = null;
            }
            mix_comp_prob = null;
        } /* Mixture components present */

        return;
    }

    void UpdateEmissionConditionalBernoulli(Data data,double[][]prob_s,int is_averaged){
        /* Unvariate conditional Bernoulli */

        /* Temporary variable(s) */
        double sum;

        for (int b =0;b<num_states; b++)
        {
            /* First sequence entry probabilities */
            state_prob[b]=0.0;
            for (int n =0;n<data.num_seqs;n++)
            {
                if(data.sequence[n].entry[0].ddata[dim_index[0]]==b)
                    state_prob[b]+=prob_s[n][0];

                for (int t =1;t<data.sequence[n].seq_length;t++)
                {
                    if(data.sequence[n].entry[t].ddata[dim_index[0]]==b)
                        cond_state_prob[data.sequence[n].entry[t-1].ddata[dim_index[0]]][b]+=prob_s[n][t];

                    if(is_averaged)
                        if(data.sequence[n].entry[t].ddata[dim_index[0]]==b)
                            state_prob[b]+=prob_s[n][t];
                }
            }
        }

        /* Normalizing */
        sum=0.0;
        for (int b =0;b<num_states; b++)
        {
            /* Adding the prior counts */
            state_prob[b]+=pcount_single[b];
            sum+=state_prob[b];
        }
        for (int b =0;b<num_states; b++)
            state_prob[b]/=sum;
        for (int b =0;b<num_states; b++)
            log_state_prob[b]=log(state_prob[b]);

        for (int i =0;i<num_states; i++)
        {
            sum=0.0;
            for (int j =0;j<num_states; j++)
            {
                /* Adding the prior counts */
                cond_state_prob[i][j]+=pcount_uni[i][j];
                sum+=cond_state_prob[i][j];
            }
            for (int j =0;j<num_states; j++)
                cond_state_prob[i][j]/=sum;
        }

        for (int i =0;i<num_states; i++)
            for (int j =0;j<num_states; j++)
                log_cond_state_prob[i][j]=log(cond_state_prob[i][j]);

        return;
    }

    void UpdateEmissionChowLiu(Data data,double[][]prob_s){

        double[][][][]pair_prob;         // Projection on the pair of the variables   
        double[][]MI;                  // Mutual information
        double best_MI;               // Best value of mutual information in the current pass
        int[]best_edge;              // The list of current best edges

        double[][]cond_prob;
        double[][]temp_ind_prob;       // Future individual probabilities

        /* Updates */
        double[][]mult;                // Total update from instantiated nodes
        double[][]sent_backward;       // Update sent in a backward pass (to the parent)
        double[][]received_backward;   // Total update received from uninstantiated children
        double[][]received_forward;    // Update sent in a forward pass (from the parent)

        int[]visited;                // Indicator of visited variables for bivariate probabilities

        double[][][][]mix_weight;        // Weights for mixture components
        double norm_const;            // Normalization constant

//        int node,node1,node2;        // Node (variable) indices
//        int node1_bi;                // Node (variable) indices for bivariate marginal computation
//        int node_index,last_edge;
//        int root_node;               // Index of the root node
//        int b,b1,b2,b_bi;            // Variable values
//        int n;                       // Sequence index
//        int t;                       // Entry index
//        int env,env1,env2;           // Envelope indices
//        int e;                       // Edge index

        /* Temporary variable(s) */
        double sum;
        DataPoint datum;
        double[]temp_cond_prob;
        double[][]temp_received_backward;
        double[][]temp_sent_backward;

        /* Allocating the array of individual probabilities */
        temp_ind_prob=new double[dim][];
        for (int node =0;node<dim; node++)
            temp_ind_prob[node]=new double[num_states];

        /* Allocating the array of pairwise probabilities */
        pair_prob=new double[dim][][][];
        for(int node1=0;node1<dim; node1++)
        {
            pair_prob[node1]=new double[node1][][];
            for(int node2=0;node2<node1; node2++)
            {
                pair_prob[node1][node2]=new double[num_states][];
                for (int b =0;b<num_states; b++)
                    pair_prob[node1][node2][b]=new double[num_states];
            }
        }

        /* Array of best edges */
        best_edge=new int[dim-1];

        if(subdist != null)
        { /* Mixture */
            /* Allocating weights for mixture components */
            mix_weight=new double[dim][][][];
            for (int node =0;node<dim; node++)
            {
                mix_weight[node]=new double[num_states][][];
                for (int b =0;b<num_states; b++)
                {
                    mix_weight[node][b]=new double[data.num_seqs][];
                    for (int n =0;n<data.num_seqs;n++)
                        mix_weight[node][b][n]=new double[data.sequence[n].seq_length];
                }
            }
        } /* Mixture */

        /* Initializing */

        /* Individual probability projections */
        for (int node =0;node<dim; node++)
            for (int b =0;b<num_states; b++)
                temp_ind_prob[node][b]=0.0;

        /* Pairwise probability projections */
        for(int node1=0;node1<dim; node1++)
            for(int node2=0;node2<node1; node2++)
                for(int b1=0;b1<num_states; b1++)
                    for(int b2=0;b2<num_states; b2++)
                        pair_prob[node1][node2][b1][b2]=0.0;

        /* Initializing structures for count collection */

        /* Multiplication factor for each node */
        mult=new double[dim][];
        for (int node =0;node<dim; node++)
            mult[node]=new double[num_states];

        sent_backward=new double[dim][];
        for (int node =0;node<dim; node++)
            sent_backward[node]=new double[num_states];
        received_backward=new double[dim][];
        for (int node =0;node<dim; node++)
            received_backward[node]=new double[num_states];
        received_forward=new double[dim][];
        for (int node =0;node<dim; node++)
            received_forward[node]=new double[num_states];
        cond_prob=new double[dim][];
        for (int node =0;node<dim; node++)
            cond_prob[node]=new double[num_states];
        temp_cond_prob=new double[num_states];
        temp_sent_backward=new double[dim][];
        for (int node =0;node<dim; node++)
            temp_sent_backward[node]=new double[num_states];
        temp_received_backward=new double[dim][];
        for (int node =0;node<dim; node++)
            temp_received_backward[node]=new double[num_states];
        visited=new int[dim];

        for (int n =0;n<data.num_seqs;n++)
            for (int t =0;t<data.sequence[n].seq_length;t++)
            { /* For each example, computing its contribution to pairwise probabilities */
                datum=data.sequence[n].entry[t];

                /* Computing multiplicative factor for each node */
                if(subdist != null)
                    /* Mixture */
                    for (int node =0;node<dim; node++)
                        for (int b =0;b<num_states; b++)
                            if(t==0)
                                mult[node][b]=exp(subdist[node][b].log_prob(datum,null));
                            else
                                mult[node][b]=exp(subdist[node][b].log_prob(datum,data.sequence[n].entry[t-1]));

                if(subdist==null)
                    /* Computing the envelopes */
                    num_envelopes=compute_envelopes(datum,mult);

                /* Computing posterior probabilities for missing values */
                for (int env =0;env<num_envelopes; env++)
                    /* For each envelope */
                    if(envelope[env].is_missing&&envelope[env].num_nodes==1)
                    { /* Envelope contains exactly one missing variable. */
                        root_node=envelope[env].node[0];
                        sum=0.0;
                        for (int b =0;b<num_states; b++)
                        {/* Summing over the missing variable */
                            sum+=ind_prob[root_node][b]*mult[root_node][b];
                        } /* Summing over the missing variable */

                        for (int b =0;b<num_states; b++)
                            cond_prob[root_node][b]=ind_prob[root_node][b]*mult[root_node][b]/sum;
                    } /* Envelope contains exactly one missing variable. */
                    else if(envelope[env].is_missing&&envelope[env].num_nodes>1)
                    { /* Envelope contains more than one missing variable. */
	      
	      /* Initializing multiplicative contributions of leaves connected to each
		 missing variable */
                        for (int node =0;node<envelope[env].num_nodes;node++)
                            for (int b =0;b<num_states; b++)
                                /* No need to initialize sent messages. */
                                received_backward[envelope[env].node[node]][b]=1.0;
	      /* No need to initialize received message in the forward pass 
		 since it is received from one source. */

                        /* Propagating contributions in a backward pass */
                        for (int e =envelope[env].num_edges-1;e>=0;e--)
                        { /* Edges in a backwards pass */
                            int node1=edge[envelope[env].edge[e]][0];
                            int node2=edge[envelope[env].edge[e]][1];

                            /* Both nodes are missing. */
                            if(envelope[env].node[e+1]==node2)
                            { /* node1 is the parent of node2 */
                                /* Rescaling received_backward for node2 */
                                sum=0.0;
                                for(int b2=0;b2<num_states; b2++)
                                    sum+=received_backward[node2][b2];
                                for(int b2=0;b2<num_states; b2++)
                                    received_backward[node2][b2]/=sum;

                                /* Updating contributions to node1 */
                                for(int b1=0;b1<num_states; b1++)
                                { /* Computing the contribution to node1 with value b1 from child node2 */
                                    sent_backward[node2][b1]=0.0;
                                    for(int b2=0;b2<num_states; b2++)
                                        sent_backward[node2][b1]+=
                                                mult[node2][b2]*received_backward[node2][b2]*edge_prob[envelope[env].edge[e]][b1][b2]/ind_prob[node1][b1];
                                    /* Updating the parent */
                                    received_backward[node1][b1]*=sent_backward[node2][b1];
                                } /* Computing the contribution to node1 with value b1 from child node2 */
                            } /* node1 is the parent of node2 */
                            else
                            { /* node2 is the parent of node1 */
                                /* Rescaling received_backward for node1 */
                                sum=0.0;
                                for(int b1=0;b1<num_states; b1++)
                                    sum+=received_backward[node1][b1];
                                for(int b1=0;b1<num_states; b1++)
                                    received_backward[node1][b1]/=sum;

                                /* Updating contributions to node2 */
                                for(int b2=0;b2<num_states; b2++)
                                { /* Computing the contribution to node2 with value b2 from child node1 */
                                    sent_backward[node1][b2]=0.0;
                                    for(int b1=0;b1<num_states; b1++)
                                        sent_backward[node1][b2]+=
                                                mult[node1][b1]*received_backward[node1][b1]*edge_prob[envelope[env].edge[e]][b1][b2]/ind_prob[node2][b2];
                                    /* Updating the parent */
                                    received_backward[node2][b2]*=sent_backward[node1][b2];
                                } /* Computing the contribution to node2 with value b2 from child node1 */
                            } /* node2 is the parent of node1 */
                        } /* Edges in a backwards pass */

                        /* Updating the root of the envelope */
                        int root_node=envelope[env].node[0];
                        /* Rescaling received_backward for m1 */
                        sum=0.0;
                        for (int b =0;b<num_states; b++)
                            sum+=received_backward[root_node][b];
                        for (int b =0;b<num_states; b++)
                            received_backward[root_node][b]/=sum;

                        sum=0.0;
                        for (int b =0;b<num_states; b++)
                        {
                            received_forward[root_node][b]=
                                    ind_prob[root_node][b]*mult[root_node][b]*received_backward[root_node][b];
                            cond_prob[root_node][b]=received_forward[root_node][b];
                            sum+=cond_prob[root_node][b];
                        }

                        /* Normalizing */
                        for (int b =0;b<num_states; b++)
                            cond_prob[root_node][b]/=sum;

                        /* Forwards pass */
                        for (int e =0;e<envelope[env].num_edges;e++)
                        { /* Edges in a forward pass */
                            int node1=edge[envelope[env].edge[e]][0];
                            int node2=edge[envelope[env].edge[e]][1];

                            /* Both variables are missing */
                            if(envelope[env].node[e+1]==node2)
                            { /* node1 is the parent of node2 */
                                /* Computing update from node1 to node2 */
                                for(int b2=0;b2<num_states; b2++)
                                {
                                    received_forward[node2][b2]=0.0;
                                    for(int b1=0;b1<num_states; b1++)
                                        received_forward[node2][b2]+=
                                                received_forward[node1][b1]*(edge_prob[envelope[env].edge[e]][b1][b2]/(ind_prob[node2][b2]*ind_prob[node1][b1]))
                                                        /sent_backward[node2][b1];
                                }

                                /* Computing the conditional probability for node2 */
                                sum=0.0;
                                for(int b2=0;b2<num_states; b2++)
                                {
                                    received_forward[node2][b2]*=ind_prob[node2][b2]*mult[node2][b2]*received_backward[node2][b2];
                                    cond_prob[node2][b2]=received_forward[node2][b2];
                                    sum+=cond_prob[node2][b2];
                                }

                                /* Normalizing */
                                for(int b2=0;b2<num_states; b2++)
                                {
                                    cond_prob[node2][b2]/=sum;
                                    /* Rescaling received_forward[node2][b2] */
                                    received_forward[node2][b2]=cond_prob[node2][b2];
                                }
                            } /* node1 is the parent of node2 */
                            else
                            { /* node2 is the parent of node1 */
                                /* Computing update from node2 to node1 */
                                for(int b1=0;b1<num_states; b1++)
                                {
                                    received_forward[node1][b1]=0.0;
                                    for(int b2=0;b2<num_states; b2++)
                                        received_forward[node1][b1]+=
                                                received_forward[node2][b2]*(edge_prob[envelope[env].edge[e]][b1][b2]/(ind_prob[node1][b1]*ind_prob[node2][b2]))
                                                        /sent_backward[node1][b2];
                                }

                                /* Computing the conditional probability for node1 */
                                sum=0.0;
                                for(int b1=0;b1<num_states; b1++)
                                {
                                    received_forward[node1][b1]*=ind_prob[node1][b1]*mult[node1][b1]*received_backward[node1][b1];
                                    cond_prob[node1][b1]=received_forward[node1][b1];
                                    sum+=cond_prob[node1][b1];
                                }

                                /* Normalizing */
                                for(int b1=0;b1<num_states; b1++)
                                {
                                    cond_prob[node1][b1]/=sum;
                                    /* Rescaling received forward[node1][b1] */
                                    received_forward[node1][b1]=cond_prob[node1][b1];
                                }
                            } /* node2 is the parent of node1 */
                        } /* Edges in a forward pass */

                        /* Computing bivariate probabilities by setting a value of each node and completing the inference */

                        for (int node_index =envelope[env].num_nodes-1;node_index>=1;node_index--)
                        { /* Looping over nodes of the envelope */

                            /* One of the node for which to compute bivariate probabilities */
                            int node1_bi=envelope[env].node[node_index];

                            /* Finding the edge with node1_bi as a child */
                            int last_edge=node_index-1;

                            for (int b_bi =0;b_bi<num_states; b_bi++)
                            { /* Looping on the value of the current missing node */
                                /* Assigning the value for the last node */
                                /* Resetting visited indicator */
                                for (int node =0;node<node_index; node++)
                                    visited[envelope[env].node[node]]=0;

                                int ei=envelope[env].edge[last_edge];
                               int  node1=edge[envelope[env].edge[last_edge]][0];
                               int  node2=edge[envelope[env].edge[last_edge]][1];

                                /* Updating the parent of the current node */
                                if(node2==node1_bi)
                                { /* node1 is the parent of the current node */
                                    /* Computing the contributions from the current node */
                                    for(int b1=0;b1<num_states; b1++)
                                    {
                                        temp_sent_backward[node2][b1]=edge_prob[ei][b1][b_bi]/ind_prob[node1][b1];

                                        /* Recomputing parent's received_backward value */
                                        temp_received_backward[node1][b1]=
                                                received_backward[node1][b1]*temp_sent_backward[node2][b1]/sent_backward[node2][b1];
                                    }
                                    visited[node1]=1;
                                } /* node1 is the parent of the current node */
                                else
                                { /* node2 is the parent of the current node */
                                    /* Computing the contributions from the current node */
                                    for(int b2=0;b2<num_states; b2++)
                                    {
                                        temp_sent_backward[node1][b2]=edge_prob[ei][b_bi][b2]/ind_prob[node2][b2];

                                        /* Recomputing parent's received_backward value */
                                        temp_received_backward[node2][b2]=
                                                received_backward[node2][b2]*temp_sent_backward[node1][b2]/sent_backward[node1][b2];
                                    }

                                    visited[node2]=1;
                                } /* node2 is the parent of the current node */

                                /* Propagating contributions in a backward pass */
                                for (int e =last_edge-1;e>=0;e--)
                                { /* Edges in a backwards pass */
                                   int  ei=envelope[env].edge[e];
                                  int   node1=edge[ei][0];
                                  int   node2=edge[ei][1];

                                    /* Both nodes are missing. */
                                    if(envelope[env].node[e+1]==node2)
                                        /* node1 is the parent of node2 */

                                        /* Updating contributions to node1 */
                                        if(visited[node2])
                                        { /* node2 is on the path from changed node */
                                            /* Rescaling temp_received_backward for node2 */
                                            sum=0.0;
                                            for(int b2=0;b2<num_states; b2++)
                                                sum+=temp_received_backward[node2][b2];
                                            for(int b2=0;b2<num_states; b2++)
                                                temp_received_backward[node2][b2]/=sum;

                                            /* Updating contributions to node1 */
                                            for(int b1=0;b1<num_states; b1++)
                                            { /* Updating received_backward for node1 */
                                                temp_sent_backward[node2][b1]=0.0;
                                                for(int b2=0;b2<num_states; b2++)
                                                    temp_sent_backward[node2][b1]+=
                                                            mult[node2][b2]*temp_received_backward[node2][b2]*edge_prob[ei][b1][b2]/ind_prob[node1][b1];
                                                /* Updating the parent */
                                                temp_received_backward[node1][b1]=
                                                        received_backward[node1][b1]*temp_sent_backward[node2][b1]/sent_backward[node2][b1];
                                            } /* Updating received_backward for node1 */
                                            visited[node1]=1;
                                        } /* node2 is on the path from changed node */
                                        else
                                        { /* node2 is not on the path from changed node */
                                            /* Setting temp_sent_backward and temp_received_backward */
                                            for(int b1=0;b1<num_states; b1++)
                                                temp_sent_backward[node2][b1]=sent_backward[node2][b1];
                                            for(int b2=0;b2<num_states; b2++)
                                                temp_received_backward[node2][b2]=received_backward[node2][b2];
                                        } /* node2 is not on the path from changed node */
                                    else
                                        /* node2 is the parent of node1 */

                                        /* Updating contributions to node2 */
                                        if(visited[node1])
                                        { /* node1 is on the path from changed node */
                                            /* Rescaling temp_received_backward for node1 */
                                            sum=0.0;
                                            for(int b1=0;b1<num_states; b1++)
                                                sum+=temp_received_backward[node1][b1];
                                            for(int b1=0;b1<num_states; b1++)
                                                temp_received_backward[node1][b1]/=sum;

                                            /* Updating contributions to node2 */
                                            for(int b2=0;b2<num_states; b2++)
                                            { /* Updating received_backward for node2 */
                                                temp_sent_backward[node1][b2]=0.0;
                                                for(int b1=0;b1<num_states; b1++)
                                                    temp_sent_backward[node1][b2]+=
                                                            mult[node1][b1]*temp_received_backward[node1][b1]*edge_prob[ei][b1][b2]/ind_prob[node2][b2];
                                                /* Updating the parent */
                                                temp_received_backward[node2][b2]=
                                                        received_backward[node2][b2]*temp_sent_backward[node1][b2]/sent_backward[node1][b2];
                                            } /* Updating received_backward for node2 */
                                            visited[node2]=1;
                                        } /* node1 is on the path from changed node */
                                        else
                                        { /* node1 is not on the path from changed node */
                                            /* Setting temp_sent_backward and temp_received_backward */
                                            for(int b2=0;b2<num_states; b2++)
                                                temp_sent_backward[node1][b2]=sent_backward[node1][b2];
                                            for(int b1=0;b1<num_states; b1++)
                                                temp_received_backward[node1][b1]=received_backward[node1][b1];
                                        } /* node1 is not on the path from changed node */

                                } /* Edges in a backwards pass */

                                /* Updating the root of the envelope */
                                root_node=envelope[env].node[0];
                                /* Rescaling temp_received_backward for m1 */
                                sum=0.0;
                                for (int b =0;b<num_states; b++)
                                    sum+=temp_received_backward[root_node][b];
                                for (int b =0;b<num_states; b++)
                                    temp_received_backward[root_node][b]/=sum;

                                sum=0.0;
                                for (int b =0;b<num_states; b++)
                                {
                                    received_forward[root_node][b]=
                                            ind_prob[root_node][b]*mult[root_node][b]*temp_received_backward[root_node][b];
                                    temp_cond_prob[b]=received_forward[root_node][b];
                                    sum+=temp_cond_prob[b];
                                }

                                /* Normalizing */
                                for (int b =0;b<num_states; b++)
                                    temp_cond_prob[b]/=sum;

                                /* Updating the bivariate count table */
                                for (int b =0;b<num_states; b++)
                                    if(root_node>node1_bi)
                                        pair_prob[root_node][node1_bi][b][b_bi]+=cond_prob[node1_bi][b_bi]*temp_cond_prob[b]*prob_s[n][t];
                                    else
                                        pair_prob[node1_bi][root_node][b_bi][b]+=cond_prob[node1_bi][b_bi]*temp_cond_prob[b]*prob_s[n][t];

                                /* Forwards pass */
                                for (int e =0;e<last_edge; e++)
                                { /* Edges in a forward pass */
                                    ei=envelope[env].edge[e];
                                    node1=edge[ei][0];
                                    node2=edge[ei][1];

                                    /* Both variables are missing */
                                    if(envelope[env].node[e+1]==node2)
                                    { /* node1 is the parent of node2 */
                                        /* Computing update from node1 to node2 */
                                        for(int b2=0;b2<num_states; b2++)
                                        {
                                            received_forward[node2][b2]=0.0;
                                            for(int b1=0;b1<num_states; b1++)
                                                received_forward[node2][b2]+=
                                                        received_forward[node1][b1]*(edge_prob[ei][b1][b2]/(ind_prob[node2][b2]*ind_prob[node1][b1]))
                                                                /temp_sent_backward[node2][b1];
                                        }

                                        /* Computing the conditional probability for node2 */
                                        sum=0.0;
                                        for (int b =0;b<num_states; b++)
                                        {
                                            received_forward[node2][b]*=ind_prob[node2][b]*mult[node2][b]*temp_received_backward[node2][b];
                                            temp_cond_prob[b]=received_forward[node2][b];
                                            sum+=temp_cond_prob[b];
                                        }

                                        /* Normalizing */
                                        for (int b =0;b<num_states; b++)
                                        {
                                            temp_cond_prob[b]/=sum;
                                            /* Rescaling received_forward[node2][b] */
                                            received_forward[node2][b]=temp_cond_prob[b];
                                        }

                                        /* Updating the bivariate count table */
                                        for (int b =0;b<num_states; b++)
                                            if(node2>node1_bi)
                                                pair_prob[node2][node1_bi][b][b_bi]+=cond_prob[node1_bi][b_bi]*temp_cond_prob[b]*prob_s[n][t];
                                            else
                                                pair_prob[node1_bi][node2][b_bi][b]+=cond_prob[node1_bi][b_bi]*temp_cond_prob[b]*prob_s[n][t];
                                    } /* node1 is the parent of node2 */
                                    else
                                    { /* node2 is the parent of node1 */
                                        /* Computing update from node2 to node1 */
                                        for(int b1=0;b1<num_states; b1++)
                                        {
                                            received_forward[node1][b1]=0.0;
                                            for(int b2=0;b2<num_states; b2++)
                                                received_forward[node1][b1]+=
                                                        received_forward[node2][b2]*(edge_prob[ei][b1][b2]/(ind_prob[node1][b1]*ind_prob[node2][b2]))
                                                                /temp_sent_backward[node1][b2];
                                        }

                                        /* Computing the conditional probability for node1 */
                                        sum=0.0;
                                        for (int b =0;b<num_states; b++)
                                        {
                                            received_forward[node1][b]*=ind_prob[node1][b]*mult[node1][b]*temp_received_backward[node1][b];
                                            temp_cond_prob[b]=received_forward[node1][b];
                                            sum+=temp_cond_prob[b];
                                        }

                                        /* Normalizing */
                                        for (int b =0;b<num_states; b++)
                                        {
                                            temp_cond_prob[b]/=sum;
                                            /* Rescaling received_forward[node1][b] */
                                            received_forward[node1][b]=temp_cond_prob[b];
                                        }

                                        /* Updating the bivariate count table */
                                        for (int b =0;b<num_states; b++)
                                            if(node1>node1_bi)
                                                pair_prob[node1][node1_bi][b][b_bi]+=cond_prob[node1_bi][b_bi]*temp_cond_prob[b]*prob_s[n][t];
                                            else
                                                pair_prob[node1_bi][node1][b_bi][b]+=cond_prob[node1_bi][b_bi]*temp_cond_prob[b]*prob_s[n][t];
                                    } /* node2 is the parent of node1 */
                                } /* Edges in a forward pass */
                            } /* Looping on the value of the current missing node */
                        } /* Looping over nodes of the envelope */
                    } /* Envelope contains more than one missing variable. */

                /* Collecting the counts */
                if(subdist != null)
                    /* Mixture */
                    for (int i =0;i<dim; i++)
                        for (int b =0;b<num_states; b++)
                        {
                            /* Computing the weight of each datum for each mixture component */
                            mix_weight[i][b][n][t]=prob_s[n][t]*cond_prob[i][b];

                            temp_ind_prob[i][b]+=mix_weight[i][b][n][t];
                        }
                else
                    /* Not a mixture */
                    for (int i =0;i<dim; i++)
                        if(is_missing(datum.ddata[dim_index[i]]))
                            for (int b =0;b<num_states; b++)
                                temp_ind_prob[i][b]+=prob_s[n][t]*cond_prob[i][b];
                        else
                            temp_ind_prob[i][datum.ddata[dim_index[i]]]+=prob_s[n][t];

                if(subdist == null)
                    /* Not a mixture */
                    for (int i =1;i<dim; i++)
                        for (int j =0;j<i; j++)
                            if(is_missing(datum.ddata[dim_index[i]]))
                            { /* First node of the pair is missing */
                                if(!is_missing(datum.ddata[dim_index[j]]))
                                    /* Second node of the pair is not missing */
                                    for (int b =0;b<num_states; b++)
                                        pair_prob[i][j][b][datum.ddata[dim_index[j]]]+=
                                                prob_s[n][t]*cond_prob[i][b];
                            } /* First node of the pair is missing */
                            else
                            { /* First node of the pair is not missing */
                                if(is_missing(datum.ddata[dim_index[j]]))
                                    /* Second node of the pair is missing */
                                    for (int b =0;b<num_states; b++)
                                        pair_prob[i][j][datum.ddata[dim_index[i]]][b]+=
                                                prob_s[n][t]*cond_prob[j][b];
                                else
                                    /* Both nodes are present */
                                    pair_prob[i][j][datum.ddata[dim_index[i]]]
                                            [datum.ddata[dim_index[j]]]+=prob_s[n][t];
                            } /* First node of the pair is not missing */

                /* Accounting for pairs from different envelopes with missing variables */
                for(int env1=0;env1<num_envelopes; env1++)
                    for(int env2=env1+1;env2<num_envelopes; env2++)
                        if(envelope[env1].is_missing&&envelope[env2].is_missing)
                            for(int node1=0;node1<envelope[env1].num_nodes;node1++)
                                for(int node2=0;node2<envelope[env2].num_nodes;node2++)
                                {
                                    int i=envelope[env1].node[node1];
                                   int  j=envelope[env2].node[node2];
                                    for(int b1=0;b1<num_states; b1++)
                                        for(int b2=0;b2<num_states; b2++)
                                            if(i>j)
                                                pair_prob[i][j][b1][b2]+=prob_s[n][t]*cond_prob[i][b1]*cond_prob[j][b2];
                                            else
                                                pair_prob[j][i][b2][b1]+=prob_s[n][t]*cond_prob[i][b1]*cond_prob[j][b2];
                                }

                /* Deallocating the envelopes */
                if(subdist == null)
                    /* Not a mixture */
                    for (int env =0;env<num_envelopes; env++)
                        envelope[env] = null;
            } /* For each example, computing its contribution to pairwise probabilities */

        /* Deallocating count collection structures */
        for (int i =0;i<dim; i++)
            mult[i] = null;
        mult = null;
        for (int i =0;i<dim; i++)
            sent_backward[i] = null;
        sent_backward = null;
        for (int i =0;i<dim; i++)
            received_backward[i] = null;
        received_backward = null;
        for (int i =0;i<dim; i++)
            received_forward[i] = null;
        received_forward = null;
        for (int i =0;i<dim; i++)
            cond_prob[i] = null;
        cond_prob = null;
        temp_cond_prob = null;
        for (int i =0;i<dim; i++)
            temp_sent_backward[i] = null;
        temp_sent_backward = null;
        for (int i =0;i<dim; i++)
            temp_received_backward[i] = null;
        temp_received_backward = null;
        visited = null;

        /* Calculating probability distribution projections from the counts */
        /* The normalizing factor is the normalizing constant! */
        for (int i =0;i<dim; i++)
        {
            sum=0.0;
            for(int i1=0;i1<num_states; i1++)
                sum+=temp_ind_prob[i][i1];

            for(int i1=0;i1<num_states; i1++)
                ind_prob[i][i1]=(temp_ind_prob[i][i1]+pcount_uni[i][i1])/(sum+pcount);
        }

        norm_const=sum;

        for (int i =0;i<dim; i++)
            for (int j =0;j<i; j++)
            {
                sum=0.0;
                for(int i1=0;i1<num_states; i1++)
                    for(int i2=0;i2<num_states; i2++)
                        sum+=pair_prob[i][j][i1][i2];

                for(int i1=0;i1<num_states; i1++)
                    for(int i2=0;i2<num_states; i2++)
                        pair_prob[i][j][i1][i2]=(pair_prob[i][j][i1][i2]+pcount_bi[i][j][i1][i2])/(sum+pcount);
            }

        /* Allocating mutual information lower triangular matrix */
        MI=new double*[dim];
        for (int i =0;i<dim; i++)
            MI[i]=new double[dim];

        /* Initializing mutual information */
        for (int i =0;i<dim; i++)
            for (int j =0;j<i; j++)
                MI[i][j]=0.0;

        /* Calculating mutual information */
        for (int i =0;i<dim; i++)
            for (int j =0;j<i; j++)
                for(int i1=0;i1<num_states; i1++)
                {
                    MI[i][j]-=xlogx(ind_prob[i][i1]);
                    MI[i][j]-=xlogx(ind_prob[j][i1]);
                    for(int i2=0;i2<num_states; i2++)
                        MI[i][j]+=xlogx(pair_prob[i][j][i1][i2]);
                }

        /* Applying the penalty -- MDL prior*/
        for (int i =0;i<dim; i++)
            for (int j =0;j<i; j++)
            {
               int b=(num_states-1)*(num_states-1);
                MI[i][j]-=(double)b*mdl_beta/norm_const;
            }

        /* Making the matrix symmetric */
        for (int i =0;i<dim; i++)
            for (int j =0;j<i; j++)
                MI[j][i]=MI[i][j];

        /* Initializing the list of connected nodes */
        node_used[dim-1]=1;
        for (int i =0;i<dim-1;i++)
            node_used[i]=0;

        /* Initializing the array of best edges */
        /* Current best edge is always from the only "attached" node */
        for (int i =0;i<dim-1;i++)
            best_edge[i]=dim-1;

        /* Trying to connect nodes 0,...,dim-2 to node dim-1 */
        num_edges=0;
        for (int e =0;e<dim-1;e++)
        {
            best_MI=NEG_INF;
           int i1=-1;
            for (int i =0;i<dim-1;i++)
                if(!node_used[i])
                    if(MI[i][best_edge[i]]>best_MI)
                    { /* Found a better edge */
                        i1=i;
                        best_MI=MI[i][best_edge[i]];
                    } /* Found a better edge */

            /* Adding the edge */
            node_used[i1]=1;
            if(best_MI>=0.0)
            {
                if(i1>best_edge[i1])
                {
                    edge[num_edges][0]=i1;
                    edge[num_edges][1]=best_edge[i1];
                    for (int i =0;i<num_states; i++)
                        for (int j =0;j<num_states; j++)
                            edge_prob[num_edges][i][j]=pair_prob[i1][best_edge[i1]][i][j];
                }
                else
                {
                    edge[num_edges][0]=best_edge[i1];
                    edge[num_edges][1]=i1;
                    for (int i =0;i<num_states; i++)
                        for (int j =0;j<num_states; j++)
                            edge_prob[num_edges][i][j]=pair_prob[best_edge[i1]][i1][i][j];
                }

                /* Storing the information about the edge */
                edge_MI[num_edges]=best_MI;
                /* Rank is not yet available */

                /* Updating the number of edges */
                num_edges++;
            }

            /* Adjusting the list of best edges to not-yet-connected nodes */
            for (int i =0;i<dim-1;i++)
                if(!node_used[i])
                    if(MI[i][i1]>MI[i][best_edge[i]])
                        best_edge[i]=i1;
        }

        /* Deallocating temporary individual probabilities */
        for (int i =0;i<dim; i++)
            temp_ind_prob[i] = null;
        temp_ind_prob = null;

        /* Deallocaing pairwise probabilities */
        for (int i =0;i<dim; i++)
        {
            for (int j =0;j<i; j++)
            {
                for(int i1=0;i1<num_states; i1++)
                    pair_prob[i][j][i1] = null;
                pair_prob[i][j] = null;
            }
            pair_prob[i] = null;
        }
        pair_prob = null;

        /* Deallocating mutual information */
        for (int i =0;i<dim; i++)
            MI[i] = null;
        MI = null;

        /* Deallocating the array of best edges */
        best_edge = null;

        if(subdist != null)
        { /* Mixture */
            /* Deleting the old envelopes and computing the new ones */
            for (int env =0;env<num_envelopes; env++)
                envelope[env] = null;

            num_envelopes=compute_envelopes_full();

            /* Updating mixture components */

            /* Passing weights of the data to the mixture components */
            for (int i =0;i<dim; i++)
                for (int j =0;j<num_states; j++)
                {
                    sum=0.0;
                    for (int n =0;n<data.num_seqs;n++)
                        for (int t =0;t<data.sequence[n].seq_length;t++)
                            sum+=mix_weight[i][j][n][t];

                    /* Updating mixture components */
                    subdist[i][j].UpdateEmissionParameters(data,mix_weight[i][j],sum);
                }
        } /* Mixture */

        if(subdist!= null)
        { /* Mixture */
            /* Deallocating weights used for mixture components */
            for (int i =0;i<dim; i++)
            {
                for (int j =0;j<num_states; j++)
                {
                    for (int n =0;n<data.num_seqs;n++)
                        mix_weight[i][j][n] = null;
                    mix_weight[i][j] = null;
                }
                mix_weight[i] = null;
            }
            mix_weight = null;
        } /* Mixture */

        return;
    }

    void UpdateEmissionConditionalChowLiu(Data data,double[][] prob_s,double norm_const){
  /* Indices:
     0,..,dim-1 -- previous entry observations
     dim,..,2*dim-1 -- current entry observations
  */

        /* Allocating the array of pairwise probabilities */
  /* pair_prob[i][j][i1][i2]:
     i is the node for the current observation
     j is the node for the previous observation
     i1 -- value for node i
     i2 -- value for node j */

        double[][][][]pair_prob;         // Projection on the pair of the variables
        double[][]MI;                  // Mutual information
        int[]best_edge;              // The list of current best edges
        double best_MI;               // Best value of mutual information in the current pass

        /* Temporary variable(s) */
        double sum;

        pair_prob=new double[dim][][][];
        for (int i =0;i<dim; i++)
        {
            pair_prob[i]=new double[2*dim][][];
            for (int j =i+1;j<2*dim;j++)
            {
                pair_prob[i][j]=new double[num_states][];
                for(int i1=0;i1<num_states; i1++)
                    pair_prob[i][j][i1]=new double[num_states];
            }
        }

        /* Allocating mutual information lower triangular matrix */
        MI=new double*[dim];
        for (int i =0;i<dim; i++)
            MI[i]=new double[2*dim];

        /* Array of best edges */
        best_edge=new int[dim];

        /* Initializing */

        /* Individual probability projections */
        for (int i =0;i<2*dim;i++)
            for(int i1=0;i1<num_states; i1++)
                ind_prob[i][i1]=0.0;

        /* Pairwise probability projections */
        for (int i =0;i<dim; i++)
            for (int j =i+1;j<2*dim;j++)
                for(int i1=0;i1<num_states; i1++)
                    for(int i2=0;i2<num_states; i2++)
                        pair_prob[i][j][i1][i2]=0.0;

        /* Mutual information */
        for (int i =0;i<dim; i++)
            for (int j =i+1;j<2*dim;j++)
                MI[i][j]=0.0;
        for (int i =0;i<dim; i++)
            MI[i][i]=NEG_INF;

        /* Looping through the data to collect the counts */
        for (int n =0;n<data.num_seqs;n++)
        {
            for (int t =0;t<data.sequence[n].seq_length;t++)
                for (int i =0;i<dim; i++)
                { /* !!! Checking for missing data !!! */
                    if(!is_missing(data.sequence[n].entry[t].ddata[dim_index[i]]))
                    {
                        ind_prob[i][data.sequence[n].entry[t].ddata[dim_index[i]]]+=prob_s[n][t];
                        for (int j =i+1;j<dim; j++)
                            /* !!! !!! */
                            if(!is_missing(data.sequence[n].entry[t].ddata[dim_index[j]]))
                                pair_prob[i][j][data.sequence[n].entry[t].ddata[dim_index[i]]]
                                        [data.sequence[n].entry[t].ddata[dim_index[j]]]+=prob_s[n][t];
                    }
                } /* !!! Checking for missing data !!! */

            for (int t =1;t<data.sequence[n].seq_length;t++)
                for (int i =0;i<dim; i++)
                { /* !!! Checking for missing data !!! */
                    if(!is_missing(data.sequence[n].entry[t-1].ddata[dim_index[i]]))
                    {
                        ind_prob[i+dim][data.sequence[n].entry[t-1].ddata[dim_index[i]]]+=prob_s[n][t];
                        for (int j =0;j<dim; j++)
                            /* !!! !!! */
                            if(!is_missing(data.sequence[n].entry[t].ddata[dim_index[j]]))
                                pair_prob[j][i+dim][data.sequence[n].entry[t].ddata[dim_index[j]]]
                                        [data.sequence[n].entry[t-1].ddata[dim_index[i]]]+=prob_s[n][t];
                    }
                } /* !!! Checking for missing data !!! */
        }

        /* Calculating probability distribution projections from the counts */
        /* The normalizing factor is the normalizing constant! */
        for (int i =0;i<2*dim;i++)
        {
            sum=0.0;
            for(int i1=0;i1<num_states; i1++)
                sum+=ind_prob[i][i1];

            for(int i1=0;i1<num_states; i1++)
                ind_prob[i][i1]/=sum;
        }

        for (int i =0;i<dim; i++)
            for (int j =i+1;j<2*dim;j++)
            {
                sum=0.0;
                for(int i1=0;i1<num_states; i1++)
                    for(int i2=0;i2<num_states; i2++)
                        sum+=pair_prob[i][j][i1][i2];

                for(int i1=0;i1<num_states; i1++)
                    for(int i2=0;i2<num_states; i2++)
                        pair_prob[i][j][i1][i2]/=sum;
            }

        /* Calculating mutual information */
        for (int i =0;i<dim; i++)
            for (int j =i+1;j<2*dim;j++)
            {
                for(int i1=0;i1<num_states; i1++)
                {
                    /* !!! Assuming same number of states for all variables !!! */
                    MI[i][j]-=xlogx(ind_prob[i][i1]);
                    MI[i][j]-=xlogx(ind_prob[j][i1]);
                    for(int i2=0;i2<num_states; i2++)
                        MI[i][j]+=xlogx(pair_prob[i][j][i1][i2]);
                }

                /* Applying the penalty -- MDL prior */
                /* Computing the number of free parameters associated with the edge */
                b=(num_states-1)*(num_states-1);
                MI[i][j]-=(double)b*mdl_beta/norm_const;

                if(j<dim )
                    MI[j][i]=MI[i][j];
            }

        /* Initializing the list of connected nodes */
        for (int i =0;i<dim; i++)
            node_used[i]=0;
        for (int i =dim;i<2*dim;i++)
            node_used[i]=1;

        /* Initializing the array of best edges */
        /* Initially, the best edges are computed from previous entry measurements to the current */
        for (int i =0;i<dim; i++)
        { /* To node i */
            best_edge[i]=-1;
            best_MI=NEG_INF;
            for (int j =dim;j<2*dim;j++)
                if(MI[i][j]>best_MI)
                {
                    best_MI=MI[i][j];
                    best_edge[i]=j;
                }
        } /* To node i */

        /* Trying to connecting nodes dim,...,2*dim-1 to nodes 0,...,dim-1 */
        num_edges=0;
        for (int e =0;e<dim; e++)
        {
            best_MI=NEG_INF;
            i1=-1;
            for (int i =0;i<dim; i++)
                if(!node_used[i])
                    if(MI[i][best_edge[i]]>best_MI)
                    { /* Found a better edge */
                        i1=i;
                        best_MI=MI[i][best_edge[i]];
                    } /* Found a better edge */

            /* Adding the edge */
            node_used[i1]=1;
            if(best_MI>=0.0)
            {
                if(i1<best_edge[i1])
                { /* Unconnected index is smaller than connected one */
                    edge[num_edges][0]=i1;
                    edge[num_edges][1]=best_edge[i1];
                    for (int i =0;i<num_states; i++)
                        for (int j =0;j<num_states; j++)
                            edge_prob[num_edges][i][j]=pair_prob[i1][best_edge[i1]][i][j];
                } /* best_edge[i1] is from the previous time point */
                else
                { /* Connected index is smaller than unconnected one */
                    edge[num_edges][0]=best_edge[i1];
                    edge[num_edges][1]=i1;
                    for (int i =0;i<num_states; i++)
                        for (int j =0;j<num_states; j++)
                            edge_prob[num_edges][i][j]=pair_prob[best_edge[i1]][i1][i][j];
                } /* Connected index is smaller than unconnected one */

                /* Storing the information about the edge */
                edge_MI[num_edges]=best_MI;
                /* Rank is not yet available */

                /* Updating the number of edges */
                num_edges++;
            }

            /* Adjusting the list of best edges to not-yet-connected nodes */
            for (int i =0;i<dim; i++)
                if(!node_used[i])
                    if(MI[i][i1]>MI[i][best_edge[i]])
                        best_edge[i]=i1;
        }

        /* Deallocaing pairwise probabilities */
        for (int i =0;i<dim; i++)
        {
            for (int j =i+1;j<2*dim;j++)
            {
                for(int i1=0;i1<num_states; i1++)
                    pair_prob[i][j][i1] = null;
                pair_prob[i][j] = null;
            }
            pair_prob[i] = null;
        }
        pair_prob = null;

        /* Deallocating mutual information */
        for (int i =0;i<dim; i++)
            MI[i] = null;
        MI = null;

        /* Deallocating the array of best edges */
        best_edge = null;

        return;
    }

    void UpdateEmissionFullBivariateMaxEnt(Data data,double[][]prob_s,double norm_const){
        /* !!! No missing values !!! */

        double[][]count;
        int num_combs;
        double[]uprob;
        int num_unchanged;

        double nu,f;

        /* Temporary variable(s) */
        double sum;

        /* Initializing the constraint counts */
        count=new double[dim][];
        for (int i =0;i<dim; i++)
        {
            count[i]=new double[i+1];
            for (int j =0;j<i+1;j++)
                count[i][j]=0.0;
        }

        /* Calculating counts */
        for (int n =0;n<data.num_seqs;n++)
            for (int t =0;t<data.sequence[n].seq_length;t++)
                for (int i =0;i<dim; i++)
                    if(data.sequence[n].entry[t].ddata[dim_index[i]]==1)
                        for (int j =0;j<i+1;j++)
                            if(data.sequence[n].entry[t].ddata[dim_index[j]]==1)
                                count[i][j]+=prob_s[n][t];

        /* Normalizing counts */
        for (int i =0;i<dim; i++)
            for (int j =0;j<i+1;j++)
            {
                count[i][j]/=norm_const;
                if(count[i][j]<CONSTRAINT_MIN_VALUE )
                    count[i][j]=CONSTRAINT_MIN_VALUE;
                else if(count[i][j]>CONSTRAINT_MAX_VALUE)
                    count[i][j]=CONSTRAINT_MAX_VALUE;
            }


        /* Initializing the parameters */
        for (int i =0;i<dim; i++)
            for (int j =0;j<i+1;j++)
                sigma[i][j]=0.0;
        det=(double)dim*CONST_LN2;

        /* Determining the total number of combinations */
        num_combs=power2[dim];

        /* Allocating the array of unnormalized probabilities */
        uprob=new double[num_combs];

        /* Initializing the array */
        for (int b =0;b<num_combs; b++)
            uprob[b]=1.0;

        num_unchanged=0;
        i=0;j=0;

        while(num_unchanged<dim*(dim+1)/2)
        {
            /* Calculating the probability mass for entries satisfying the constraint */
            sum=0.0;
            if(i==j)
            { /* Univariate constraint */
                for(int i1=0;i1<power2[dim-i-1];i1++)
                    for (int b =i1*power2[i+1]+power2[i];b<(i1+1)*power2[i+1];b++)
                        sum+=uprob[b];
                sum*=exp(-det);
            } /* Univariate constraint */
            else
            { /* Bivariate constraint */
                for(int i1=0;i1<power2[dim-i-1];i1++)
                    for(int i2=0;i2<power2[i-j-1];i2++)
                        for (int b =i1*power2[i+1]+power2[i]+i2*power2[j+1]+power2[j];
                             b<i1*power2[i+1]+power2[i]+(i2+1)*power2[j+1];b++)
                            sum+=uprob[b];
                sum*=exp(-det);
            } /* Bivariate constraint */

            /* Gradient */
            nu=count[i][j]-sum;

            if(abs(nu)>MAXENT_EPSILON)
            { /* Constraint not satisfied -- updating the parameter */
                if(sum>2.0*CONSTRAINT_MAX_VALUE||sum<0.5*CONSTRAINT_MIN_VALUE)
                    /* Too sensitive -- will skip for now */
                    num_unchanged++;
                else
                {
                    f=log(count[i][j])-log(1.0-count[i][j])+log(1.0-sum)-log(sum);
                    sigma[i][j]+=f;
                    f=exp(f);

                    /* Updating the probabilities */
                    if(i==j)
                    { /* Univariate constraint */
                        for(int i1=0;i1<power2[dim-i-1];i1++)
                            for (int b =i1*power2[i+1]+power2[i];b<(i1+1)*power2[i+1];b++)
                                uprob[b]*=f;
                    } /* Univariate constraint */
                    else
                    { /* Bivariate constraint */
                        for(int i1=0;i1<power2[dim-i-1];i1++)
                            for(int i2=0;i2<power2[i-j-1];i2++)
                                for (int b =i1*power2[i+1]+power2[i]+i2*power2[j+1]+power2[j];
                                     b<i1*power2[i+1]+power2[i]+(i2+1)*power2[j+1];b++)
                                    uprob[b]*=f;
                    } /* Bivariate constraint */

                    /* Updating the normalization constant */
                    det-=log(1.0-count[i][j])-log(1.0-sum);

                    num_unchanged=0;
                }
            } /* Constraint not satisfied -- updating the parameter */
            else
                /* Constraint satisfied */
                num_unchanged++;

            /* Determining the next constraint to update */
            if(j==i)
            {
                j=0;
                if(i==dim-1)
                    i=0;
                else
                    i++;
            }
            else
                j++;
        }

        /* Deallocating the array of unnormalized probabilities */
        uprob = null;

        /* Deallocating the constraint counts */
        for (int i =0;i<dim; i++)
            count[i] = null;
        count = null;

        return;
    }

    void UpdateEmissionPUCMaxEnt(Data data,double[][]prob_s,double norm_const){
        /* !!! Assuming no missing values !!! */

        double[][]score,alpha;
        int num_points;
        int[][]flat_data;    // Flattened data
        double[]w;           // Flat array of weights
        double[][]count;
        int[]old_num_features;
        int[][]old_feature_index;
        double baseline_ll;
        double new_ll;
        double[][]old_sigma;
        int from_index,to_index;
        double[]uprob;
        int[][]descendant,candidate;
        double[]factor_ll;
        double best_score;

        /* Temporary variables */
        double temp1;

        /* Creating a flat data set */
        num_points=data.num_points();
        w=new double[num_points];
        flat_data=new int*[num_points];
       int  i1=0;
        for (int n =0;n<data.num_seqs;n++)
            for (int t =0;t<data.sequence[n].seq_length;t++)
            {
                flat_data[i1]=new int[dim];
                for (int i =0;i<dim; i++)
                    flat_data[i1][i]=data.sequence[n].entry[t].ddata[dim_index[i]];
                w[i1]=prob_s[n][t];
                i1++;
            }


        /* Computing mass of univariate and bivariate occurences */
        uprob=new double[dim];
        count=new double[dim][];
        for (int i =0;i<dim; i++)
            count[i]=new double[dim];

        for (int i =0;i<dim; i++)
        {
            uprob[i]=0.0;
            for (int n =0;n<num_points; n++)
                if(flat_data[n][i])
                    uprob[i]+=w[n];

            /* Making sure the sum is not too small or too large */
            /* Guaranteeing the probability under the constraint to be in (1e-12, 1-1e-12) */
            if(uprob[i]>norm_const*CONSTRAINT_MAX_VALUE)
                uprob[i]=norm_const*CONSTRAINT_MAX_VALUE;
            else if(uprob[i]<norm_const*CONSTRAINT_MIN_VALUE)
                uprob[i]=norm_const*CONSTRAINT_MIN_VALUE;

            for (int j =0;j<i; j++)
            {
                count[i][j]=0.0;
                for (int n =0;n<num_points; n++)
                    if(flat_data[n][i]==1&&flat_data[n][j]==1)
                        count[i][j]+=w[n];

                /* Making sure the sum is not too small or too large */
                /* Guaranteeing the probability under the constraint to be in (1e-12, 1-1e-12) */
                if(count[i][j]>norm_const*CONSTRAINT_MAX_VALUE)
                    count[i][j]=norm_const*CONSTRAINT_MAX_VALUE;
                else if(count[i][j]<norm_const*CONSTRAINT_MIN_VALUE)
                    count[i][j]=norm_const*CONSTRAINT_MIN_VALUE;

                count[j][i]=count[i][j];
            }
            count[i][i]=uprob[i];
        }

        /* Computing parameters for the current structure */
        baseline_ll=0.0;
        /* Allocating and initializing the array for the old structure */
        old_num_features=new int[dim];
        old_feature_index=new int*[dim];
        old_sigma=new double*[dim];
        for (int i =0;i<dim; i++)
        {
            temp1=
                    learn_univariate_conditional_maxent(num_points,flat_data,w,norm_const,i,
                            count[i],num_features[i],sigma[i],feature_index[i]);

            if(!finite(temp1))
            {
                /* Trying to set all parameters to zero */
                for (int j =0;j<num_features[i];j++)
                    sigma[i][j]=0.0;

                temp1=
                        learn_univariate_conditional_maxent(num_points,flat_data,w,norm_const,i,
                                count[i],num_features[i],sigma[i],feature_index[i]);
                baseline_ll+=
                        learn_univariate_conditional_maxent(num_points,flat_data,w,norm_const,i,
                                count[i],num_features[i],sigma[i],feature_index[i]);

                if(!finite(temp1))
                    /* Cannot find a solution */
                    temp1=NEG_INF;
            }

            baseline_ll+=temp1;

            old_num_features[i]=num_features[i];
            old_feature_index[i]=new int[dim];
            old_sigma[i]=new double[dim];
            for (int j =0;j<num_features[i];j++)
            {
                old_feature_index[i][j]=feature_index[i][j];
                old_sigma[i][j]=sigma[i][j];
            }
        }

        /* Factor log-likelihood */
        factor_ll=new double[dim];

        /* Initializing to having only the univariate features */
        for (int i =0;i<dim; i++)
        {
            num_features[i]=1;
            feature_index[i][0]=i;
        }

        for (int i =0;i<dim; i++)
        {
            sigma[i][0]=log(uprob[i])-log(norm_const-uprob[i]);
            factor_ll[i]=uprob[i]*sigma[i][0]-norm_const*log(1.0+exp(sigma[i][0]));
        }

        /* Allocating and initializing the matrix indicating the descendants in the network */
        // descendant[i][j]=1 means that j is a descendant of i.  i is a descendant of i be default.
        descendant=new int*[dim];
        for (int i =0;i<dim; i++)
        {
            descendant[i]=new int[dim];
            for (int j =0;j<dim; j++)
                descendant[i][j]=0;
        }

        /* Allocating and initializing the matrix of candidate features for each factor */
        candidate=new int*[dim];
        for (int i =0;i<dim; i++)
        {
            candidate[i]=new int[dim];
            for (int j =0;j<dim; j++)
                candidate[i][j]=1;
            candidate[i][i]=0;
        }

        /* Initializing the array of scores */
        score=new double*[dim];
        for (int i =0;i<dim; i++)
            score[i]=new double[dim];

        /* Initializing the array of starting values for parameters as determined by the scoring procedure */
        alpha=new double*[dim];
        for (int i =0;i<dim; i++)
            alpha[i]=new double[dim];

        if (BLAH) {
            /* Computing the score for each candidate */
            for (int i =0;i<dim; i++)
                InformationGain(data,prob_s,norm_const,candidate[i],score[i],alpha[i],i);
        }
        /* Efficient computation available when only one conditioning variable */
        for (int i =0;i<dim; i++)
            for (int j =0;j<i; j++)
            {
                score[i][j]=xlogx(count[i][j])+xlogx(uprob[i]-count[i][j])+xlogx(uprob[j]-count[i][j])
                        +xlogx(norm_const+count[i][j]-uprob[i]-uprob[j])+xlogx(norm_const)
                        -xlogx(uprob[i])-xlogx(uprob[j])-xlogx(norm_const-uprob[i])-xlogx(norm_const-uprob[j]);
                score[i][j]/=norm_const;
                score[j][i]=score[i][j];
                /* alpha's are irrelevant for pairs */
                alpha[i][j]=0.0;
                alpha[j][i]=0.0;
            }

        temp1=1.0;
        best_score=0.0;
        /* Determining the best score */
        for (int i =0;i<dim; i++)
            for (int j =0;j<dim; j++)
                if(candidate[i][j])
                    if(score[i][j]>best_score+COMP_EPSILON)
                    {
                        best_score=score[i][j];
                        from_index=j;
                        to_index=i;
                    }
                    else
                    {
                        if(abs(score[i][j]-best_score)<=COMP_EPSILON)
                        { /* Two scores are identical -- deciding which edge to pick */
                            temp1+=1.0;
                            if (Constants.drand48()<1.0/temp1)
                            { /* Pick the newer edge */
                                from_index=j;
                                to_index=i;
                            } /* Pick the newer edge */
                        } /* Two scores are identical -- deciding which edge to pick */
                    }

        //  while( best_score>maxent_epsilon )
        while(best_score>mdl_beta/norm_const)
        { /* Adding an edge */

            /* Updating the indices in the factor */
            feature_index[to_index][num_features[to_index]]=from_index;

            /* Initializing the value corresponding to the added feature */
            sigma[to_index][num_features[to_index]]=alpha[to_index][from_index];

            num_features[to_index]++;

            /* Updating the parameters */
            if(num_features[to_index]==2)
            { /* Special case of two variables with bivariate features */
                sigma[to_index][0]=log(uprob[to_index]-count[to_index][from_index])
                        -log(norm_const+count[to_index][from_index]-uprob[to_index]-uprob[from_index]);
                sigma[to_index][1]=log(count[to_index][from_index])
                        +log(norm_const+count[to_index][from_index]-uprob[to_index]-uprob[from_index])
                        -log(uprob[to_index]-count[to_index][from_index])
                        -log(uprob[from_index]-count[to_index][from_index]);

                factor_ll[to_index]=uprob[to_index]*sigma[to_index][0]+
                        count[to_index][from_index]*sigma[to_index][1]-
                        uprob[from_index]*log(1.0+exp(sigma[to_index][0]+sigma[to_index][1]))-
                        (norm_const-uprob[from_index])*log(1.0+exp(sigma[to_index][0]));
            } /* Special case of two variables with bivariate features */
            else
                factor_ll[to_index]=
                        learn_univariate_conditional_maxent(num_points,flat_data,w,norm_const,to_index,count[to_index],
                                num_features[to_index],sigma[to_index],feature_index[to_index]);

            candidate[to_index][from_index]=0;

            /* Updating the list of descendants */
            descendant[from_index][to_index]=1;
            for (int i =0;i<dim; i++)
                if(descendant[to_index][i])
                    descendant[from_index][i]=1;

            /* Updating descendant table for all nodes having to_index as a descendant */
            for (int i =0;i<dim; i++)
                if(descendant[i][from_index])
                    for (int j =0;j<dim; j++)
                        if(descendant[i][j]==1||descendant[from_index][j]==1)
                            descendant[i][j]=1;

            /* Updating the list of candidates */
            for (int i =0;i<dim; i++)
                for (int j =0;j<dim; j++)
                    if(candidate[i][j])
                        if(descendant[i][j])
                            candidate[i][j]=0;

            /* Recomputing the scores for the factor to_index */
            InformationGain(num_points,flat_data,w,norm_const,count[to_index],candidate[to_index],
                    score[to_index],alpha[to_index],to_index);

            temp1=1.0;
            best_score=0.0;
            /* Determining the best score */
            for (int i =0;i<dim; i++)
                for (int j =0;j<dim; j++)
                    if(candidate[i][j])
                        if(score[i][j]>best_score+COMP_EPSILON)
                        {
                            best_score=score[i][j];
                            from_index=j;
                            to_index=i;
                        }
                        else
                        if(abs(score[i][j]-best_score)<=COMP_EPSILON)
                        { /* Two scores are identical -- deciding which edge to pick */
                            temp1+=1.0;
                            if (Constants.drand48()<1.0/temp1)
                            { /* Pick the newer edge */
                                from_index=j;
                                to_index=i;
                            } /* Pick the newer edge */
                        } /* Two scores are identical -- deciding which edge to pick */
        } /* Adding an edge */

        /* Computing the new log-likelihood */
        new_ll=0.0;
        for (int i =0;i<dim; i++)
            new_ll+=factor_ll[i];

        if(new_ll>baseline_ll)
        {
            /* Deleting arrays with previous structure */
            old_num_features = null;
            for (int i =0;i<dim; i++)
            {
                old_feature_index[i] = null;
                old_sigma[i] = null;
            }
            old_feature_index = null;
            old_sigma = null;
        }
        else
        {
            /* Deleting arrays with current structure */
            num_features = null;
            for (int i =0;i<dim; i++)
            {
                feature_index[i] = null;
                sigma[i] = null;
            }
            feature_index = null;
            sigma = null;

            /* Reusing the old structure */
            num_features=old_num_features;
            feature_index=old_feature_index;
            sigma=old_sigma;
        }

        /* Deallocating the arrays */
        for (int i =0;i<dim; i++)
            alpha[i] = null;
        alpha = null;

        for (int i =0;i<dim; i++)
            score[i] = null;
        score = null;

        for (int i =0;i<dim; i++)
            candidate[i] = null;
        candidate = null;

        for (int i =0;i<dim; i++)
            descendant[i] = null;
        descendant = null;

        uprob = null;
        for (int i =0;i<dim; i++)
            count[i] = null;
        count = null;

        factor_ll = null;

        /* Deallocating flat data set */
        w = null;
        for (int n =0;n<num_points; n++)
            flat_data[n] = null;
        flat_data = null;

        return;
    }

    void UpdateEmissionConditionalPUCMaxEnt(Data data,double[][]prob_s,double norm_const){
        /* !!! Assuming no missing values !!! */

        double[][]score,alpha;
        int num_points;
        int[][]flat_data;    // Flattened data
        double[]w;           // Flat array of weights
        double[][]count;
        int[]old_num_features;
        int[][] old_feature_index;
        double baseline_ll;
        double new_ll;
        double[][]old_sigma;
        int from_index,to_index;
        double[]uprob;
        int[][]descendant,candidate;
        double[]factor_ll;
        double best_score;

        /* Temporary variables */
        double temp1;

  /* Creating a flat data set 
     
  Only time points 2..T are used.
  
  components 0..dim-1 -- current time point
  components dim..2*dim-1 -- previous time point
  */

        /* Computing first entry probability of one */
        for (int i =0;i<dim; i++)
        {
            state_prob[i]=0.0;
            for (int n =0;n<data.num_seqs;n++)
                for (int t =0;t<data.sequence[n].seq_length;t++)
                    if(data.sequence[n].entry[t].ddata[dim_index[i]]==1)
                        state_prob[i]+=prob_s[n][t];

            state_prob[i]/=norm_const;
        }

        num_points=data.num_points()-data.num_seqs;
        w=new double[num_points];
        flat_data=new int[num_points][];
       int i1=0;
        for (int n =0;n<data.num_seqs;n++)
            for (int t =1;t<data.sequence[n].seq_length;t++)
            {
                flat_data[i1]=new int[2*dim];
                for (int i =0;i<dim; i++)
                {
                    flat_data[i1][i]=data.sequence[n].entry[t].ddata[dim_index[i]];
                    flat_data[i1][i+dim]=data.sequence[n].entry[t-1].ddata[dim_index[i]];
                }
                w[i1]=prob_s[n][t];
                i1++;
            }

        /* Computing mass of univariate and bivariate occurences */
        uprob=new double[2*dim];
        count=new double[dim][];
        for (int i =0;i<dim; i++)
            count[i]=new double[2*dim];

        for (int i =0;i<dim; i++)
        {
            uprob[i]=0.0;
            for (int n =0;n<num_points; n++)
                if(flat_data[n][i]==1)
                    uprob[i]+=w[n];

            uprob[i+dim]=0.0;
            for (int n =0;n<num_points; n++)
                if(flat_data[n][i+dim]==1)
                    uprob[i+dim]+=w[n];

            /* Making sure the sum is not too small or too large */
            /* Guaranteeing the probability under the constraint to be in (1e-12, 1-1e-12) */
            if(uprob[i]>norm_const*CONSTRAINT_MAX_VALUE)
                uprob[i]=norm_const*CONSTRAINT_MAX_VALUE;
            else if(uprob[i]<norm_const*CONSTRAINT_MIN_VALUE)
                uprob[i]=norm_const*CONSTRAINT_MIN_VALUE;
            if(uprob[i+dim]>norm_const*CONSTRAINT_MAX_VALUE)
                uprob[i+dim]=norm_const*CONSTRAINT_MAX_VALUE;
            else if(uprob[i+dim]<norm_const*CONSTRAINT_MIN_VALUE)
                uprob[i+dim]=norm_const*CONSTRAINT_MIN_VALUE;

            for (int j =i+1;j<2*dim;j++)
            {
                count[i][j]=0.0;
                for (int n =0;n<num_points; n++)
                    if(flat_data[n][i]==1&&flat_data[n][j]==1)
                        count[i][j]+=w[n];

                /* Making sure the sum is not too small or too large */
                /* Guaranteeing the probability under the constraint to be in (1e-12, 1-1e-12) */
                if(count[i][j]>norm_const*CONSTRAINT_MAX_VALUE)
                    count[i][j]=norm_const*CONSTRAINT_MAX_VALUE;
                else if(count[i][j]<norm_const*CONSTRAINT_MIN_VALUE)
                    count[i][j]=norm_const*CONSTRAINT_MIN_VALUE;

                if(j<dim )
                    count[j][i]=count[i][j];
            }
            count[i][i]=uprob[i];
        }

        /* Computing parameters for the current structure */
        baseline_ll=0.0;
        /* Allocating and initializing the array for the old structure */
        old_num_features=new int[dim];
        old_feature_index=new int*[dim];
        old_sigma=new double*[dim];
        for (int i =0;i<dim; i++)
        {
            temp1=learn_univariate_conditional_maxent(num_points,flat_data,w,norm_const,i,
                    count[i],num_features[i],sigma[i],feature_index[i]);

            if(!finite(temp1))
            {
                /* Trying to set all parameters to zero */
                for (int j =0;j<num_features[i];j++)
                    sigma[i][j]=0.0;

                temp1=
                        learn_univariate_conditional_maxent(num_points,flat_data,w,norm_const,i,
                                count[i],num_features[i],sigma[i],feature_index[i]);
                baseline_ll+=
                        learn_univariate_conditional_maxent(num_points,flat_data,w,norm_const,i,
                                count[i],num_features[i],sigma[i],feature_index[i]);

                if(!finite(temp1))
                    /* Cannot find a solution */
                    temp1=NEG_INF;
            }

            baseline_ll+=temp1;

            old_num_features[i]=num_features[i];
            //      old_feature_index[i]=new int[num_features[i]];
            //      old_sigma[i]=new double[num_features[i]];
            old_feature_index[i]=new int[2*dim];
            old_sigma[i]=new double[2*dim];
            for (int j =0;j<num_features[i];j++)
            {
                old_feature_index[i][j]=feature_index[i][j];
                old_sigma[i][j]=sigma[i][j];
            }
        }

        /* Factor log-likelihood */
        factor_ll=new double[dim];

        /* Initializing to having only the univariate features */
        for (int i =0;i<dim; i++)
        {
            num_features[i]=1;
            feature_index[i][0]=i;
        }

        for (int i =0;i<dim; i++)
        {
            sigma[i][0]=log(uprob[i])-log(norm_const-uprob[i]);
            factor_ll[i]=uprob[i]*sigma[i][0]-norm_const*log(1.0+exp(sigma[i][0]));
        }

        /* Allocating and initializing the matrix indicating the descendants in the network */
        /* descendant[i][j]=1 means that j is a descendant of i.  i is a descendant of i be default. */
        descendant=new int*[dim];
        for (int i =0;i<dim; i++)
        {
            descendant[i]=new int[dim];
            for (int j =0;j<dim; j++)
                descendant[i][j]=0;
        }

        /* Allocating and initializing the matrix of candidate features for each factor */
        candidate=new int*[dim];
        for (int i =0;i<dim; i++)
        {
            candidate[i]=new int[2*dim];
            for (int j =0;j<2*dim;j++)
                candidate[i][j]=1;
            candidate[i][i]=0;
        }

        /* Initializing the array of scores */
        score=new double*[dim];
        for (int i =0;i<dim; i++)
            score[i]=new double[2*dim];

        /* Initializing the array of starting values for parameters as determined by the scoring procedure */
        alpha=new double*[dim];
        for (int i =0;i<dim; i++)
            alpha[i]=new double[2*dim];

        if (BLAH) {
            /* Computing the score for each candidate */
            for (int i =0;i<dim; i++)
                InformationGain(data,prob_s,norm_const,candidate[i],score[i],alpha[i],i);
        }
        /* Efficient computation available when only one conditioning variable */
        for (int i =0;i<dim; i++)
            for (int j =i+1;j<2*dim;j++)
            {
                score[i][j]=xlogx(count[i][j])+xlogx(uprob[i]-count[i][j])+xlogx(uprob[j]-count[i][j])
                        +xlogx(norm_const+count[i][j]-uprob[i]-uprob[j])+xlogx(norm_const)
                        -xlogx(uprob[i])-xlogx(uprob[j])-xlogx(norm_const-uprob[i])-xlogx(norm_const-uprob[j]);
                score[i][j]/=norm_const;
                if(j<dim )
                    score[j][i]=score[i][j];
                /* alpha's are irrelevant for pairs */
                alpha[i][j]=0.0;
                if(j<dim )
                    alpha[j][i]=0.0;
            }

        temp1=1.0;
        best_score=0.0;
        /* Determining the best score */
        for (int i =0;i<dim; i++)
            for (int j =0;j<2*dim;j++)
                if(candidate[i][j])
                    if(score[i][j]>best_score+COMP_EPSILON)
                    {
                        best_score=score[i][j];
                        from_index=j;
                        to_index=i;
                    }
                    else
                    {
                        if(abs(score[i][j]-best_score)<=COMP_EPSILON)
                        { /* Two scores are identical -- deciding which edge to pick */
                            temp1+=1.0;
                            if (Constants.drand48()<1.0/temp1)
                            { /* Pick the newer edge */
                                from_index=j;
                                to_index=i;
                            } /* Pick the newer edge */
                        } /* Two scores are identical -- deciding which edge to pick */
                    }

        //  while( best_score>MIN_ME_LL_CHANGE )
        while(best_score>mdl_beta/norm_const)
        { /* Adding an edge */

            /* Updating the indices in the factor */
            feature_index[to_index][num_features[to_index]]=from_index;

            /* Initializing the value corresponding to the added feature */
            sigma[to_index][num_features[to_index]]=alpha[to_index][from_index];

            num_features[to_index]++;

            /* Updating the parameters */
            if(num_features[to_index]==2)
            { /* Special case of two variables with bivariate features */
                sigma[to_index][0]=log(uprob[to_index]-count[to_index][from_index])
                        -log(norm_const+count[to_index][from_index]-uprob[to_index]-uprob[from_index]);
                sigma[to_index][1]=log(count[to_index][from_index])
                        +log(norm_const+count[to_index][from_index]-uprob[to_index]-uprob[from_index])
                        -log(uprob[to_index]-count[to_index][from_index])
                        -log(uprob[from_index]-count[to_index][from_index]);

                factor_ll[to_index]=uprob[to_index]*sigma[to_index][0]+
                        count[to_index][from_index]*sigma[to_index][1]-
                        uprob[from_index]*log(1.0+exp(sigma[to_index][0]+sigma[to_index][1]))-
                        (norm_const-uprob[from_index])*log(1.0+exp(sigma[to_index][0]));
            } /* Special case of two variables with bivariate features */
            else
                factor_ll[to_index]=
                        learn_univariate_conditional_maxent(num_points,flat_data,w,norm_const,to_index,count[to_index],
                                num_features[to_index],sigma[to_index],feature_index[to_index]);

            candidate[to_index][from_index]=0;

            if(from_index<dim )
            {
                /* Updating the list of descendants */
                descendant[from_index][to_index]=1;
                for (int i =0;i<dim; i++)
                    if(descendant[to_index][i])
                        descendant[from_index][i]=1;

                /* Updating descendant table for all nodes having to_index as a descendant */
                for (int i =0;i<dim; i++)
                    if(descendant[i][from_index])
                        for (int j =0;j<dim; j++)
                            if(descendant[i][j]==1||descendant[from_index][j]==1)
                                descendant[i][j]=1;

                /* Updating the list of candidates */
                for (int i =0;i<dim; i++)
                    for (int j =0;j<dim; j++)
                        if(candidate[i][j])
                            if(descendant[i][j])
                                candidate[i][j]=0;
            }

            /* Recomputing the scores for the factor to_index */
            InformationGain(num_points,flat_data,w,norm_const,count[to_index],candidate[to_index],
                    score[to_index],alpha[to_index],to_index);

            temp1=1.0;
            best_score=0.0;
            /* Determining the best score */
            for (int i =0;i<dim; i++)
                for (int j =0;j<2*dim;j++)
                    if(candidate[i][j])
                        if(score[i][j]>best_score+COMP_EPSILON)
                        {
                            best_score=score[i][j];
                            from_index=j;
                            to_index=i;
                        }
                        else
                        if(abs(score[i][j]-best_score)<=COMP_EPSILON)
                        { /* Two scores are identical -- deciding which edge to pick */
                            temp1+=1.0;
                            if (Constants.drand48()<1.0/temp1)
                            { /* Pick the newer edge */
                                from_index=j;
                                to_index=i;
                            } /* Pick the newer edge */
                        } /* Two scores are identical -- deciding which edge to pick */
        } /* Adding an edge */

        /* Computing the new log-likelihood */
        new_ll=0.0;
        for (int i =0;i<dim; i++)
            new_ll+=factor_ll[i];

        if(new_ll>baseline_ll)
        {
            /* Deleting arrays with previous structure */
            old_num_features = null;
            for (int i =0;i<dim; i++)
            {
                old_feature_index[i] = null;
                old_sigma[i] = null;
            }
            old_feature_index = null;
            old_sigma = null;
        }
        else
        {
            /* Deleting arrays with current structure */
            num_features = null;
            for (int i =0;i<dim; i++)
            {
                feature_index[i] = null;
                sigma[i] = null;
            }
            feature_index = null;
            sigma = null;

            /* Reusing the old structure */
            num_features=old_num_features;
            feature_index=old_feature_index;
            sigma=old_sigma;
        }

        /* Deallocating the arrays */
        for (int i =0;i<dim; i++)
            alpha[i] = null;
        alpha = null;

        for (int i =0;i<dim; i++)
            score[i] = null;
        score = null;

        for (int i =0;i<dim; i++)
            candidate[i] = null;
        candidate = null;

        for (int i =0;i<dim; i++)
            descendant[i] = null;
        descendant = null;

        uprob = null;
        for (int i =0;i<dim; i++)
            count[i] = null;
        count = null;

        factor_ll = null;

        /* Deallocating flat data set */
        w = null;
        for (int n =0;n<num_points; n++)
            flat_data[n] = null;
        flat_data = null;

        return;
    }

    void UpdateEmissionDeltaExponential(Data data,double[][]prob_s,double norm_const){

        /* Temporary variable(s) */
        double sum;
        double[]temp,temp_prob_mix,temp_exp_value;
        double[]value_contrib;
        double max_value;

        /* Flags */
        //  int leave_unchanged=0;

        /* Updating parameters for delta-exponential distribution */
        temp_prob_mix=new double[num_states];
        temp_exp_value=new double[num_states-1];
        temp=new double[num_states-1];
        value_contrib=new double[num_states-1];

        /* Calculating mixing parameters and the exponential parameter */
        for (int i =0;i<num_states; i++)
            temp_prob_mix[i]=0.0;
        for (int i =0;i<num_states-1;i++)
            temp_exp_value[i]=0.0;

        for (int n =0;n<data.num_seqs;n++)
            for (int t =0;t<data.sequence[n].seq_length;t++)
                if(!is_missing(data.sequence[n].entry[t].rdata[dim_index[0]]))
                {
                    if(abs(data.sequence[n].entry[t].rdata[dim_index[0]])<=COMP_THRESHOLD)
                        temp_prob_mix[0]+=prob_s[n][t];
                    else
                    {
                        max_value=NEG_INF;
                        for (int i =0;i<num_states-1;i++)
                        {
                            value_contrib[i]=log(mix_prob[i+1])+log(exp_param[i])-exp_param[i]*data.sequence[n].entry[t].rdata[dim_index[0]];
                            if(value_contrib[i]>max_value)
                                max_value=value_contrib[i];
                        }

                        sum=0.0;
                        for (int i =0;i<num_states-1;i++)
                        {
                            temp[i]=exp(value_contrib[i]-max_value);
                            sum+=temp[i];
                        }

                        for (int i =0;i<num_states-1;i++)
                        {
                            temp[i]*=(prob_s[n][t]/sum);
                            temp_prob_mix[i+1]+=temp[i];
                            temp_exp_value[i]+=temp[i]*data.sequence[n].entry[t].rdata[dim_index[0]];
                        }
                    }
                }

        /* Normalizing mixing probabilities */
        sum=0.0;
        for (int i =0;i<num_states; i++)
            sum+=temp_prob_mix[i];

        for (int i =0;i<num_states; i++)
            mix_prob[i]=temp_prob_mix[i]/sum;

        /* Parameters for the exponential components */
        for (int i =0;i<num_states-1;i++)
            exp_param[i]=temp_prob_mix[i+1]/temp_exp_value[i];

        value_contrib = null;
        temp_prob_mix = null;
        temp_exp_value = null;
        temp = null;

        return;
    }

    void UpdateEmissionDeltaGamma(Data data,double[][]prob_s,double norm_const){

        double f,fprime;

        /* Temporary variable(s) */
        double sum;
        double[]temp,temp_prob_mix,temp_exp_value,temp_exp_log;
        double[]value_contrib;
        double max_value;

        /* Flags */
        //  int leave_unchanged=0;

        /* Updating parameters for delta-gamma distribution */
        temp_prob_mix=new double[num_states];
        temp_exp_value=new double[num_states-1];
        temp_exp_log=new double[num_states-1];
        temp=new double[num_states-1];
        value_contrib=new double[num_states-1];

        /* Calculating mixing parameters and the exponential parameter */
        for (int i =0;i<num_states; i++)
            temp_prob_mix[i]=0.0;
        for (int i =0;i<num_states-1;i++)
        {
            temp_exp_value[i]=0.0;
            temp_exp_log[i]=0.0;
        }

        for (int n =0;n<data.num_seqs;n++)
            for (int t =0;t<data.sequence[n].seq_length;t++)
                if(!is_missing(data.sequence[n].entry[t].rdata[dim_index[0]]))
                {
                    if(abs(data.sequence[n].entry[t].rdata[dim_index[0]])<=COMP_THRESHOLD)
                        temp_prob_mix[0]+=prob_s[n][t];
                    else
                    {
                        max_value=NEG_INF;
                        for (int i =0;i<num_states-1;i++)
                        {
                            value_contrib[i]=log(mix_prob[i+1])
                                    +(gamma_param1[i]-1.0)*log(data.sequence[n].entry[t].rdata[dim_index[0]])
                                    -gamma_param2[i]*data.sequence[n].entry[t].rdata[dim_index[0]]
                                    +gamma_param1[i]*log(gamma_param2[i])-gammaln(gamma_param1[i]);
                            //+log(gamma_dist(data.sequence[n].entry[t].rdata[dim_index[0]],gamma_param1[i],gamma_param2[i]));
                            if(value_contrib[i]>max_value)
                                max_value=value_contrib[i];
                        }

                        sum=0.0;
                        for (int i =0;i<num_states-1;i++)
                        {
                            temp[i]=exp(value_contrib[i]-max_value);
                            sum+=temp[i];
                        }

                        for (int i =0;i<num_states-1;i++)
                        {
                            temp[i]*=(prob_s[n][t]/sum);
                            temp_prob_mix[i+1]+=temp[i];
                            temp_exp_value[i]+=temp[i]*data.sequence[n].entry[t].rdata[dim_index[0]];
                            temp_exp_log[i]+=temp[i]*log(data.sequence[n].entry[t].rdata[dim_index[0]]);
                        }
                    }
                }

        /* Parameters for the gamma components */
        for (int i =0;i<num_states-1;i++)
        {
            temp[i]=gamma_param1[i];

            /* Starting from the previous value */
            f=digamma(gamma_param1[i])-log(gamma_param1[i])-temp_exp_log[i]/temp_prob_mix[i+1]
                    -log(temp_prob_mix[i+1])+log(temp_exp_value[i]);
            fprime=trigamma(gamma_param1[i])-1.0/gamma_param1[i];

            while(abs(f)>NR_EPSILON&&abs(fprime)>NR_EPSILON)
            {
                gamma_param1[i]-=f/fprime;
                f=digamma(gamma_param1[i])-log(gamma_param1[i])-temp_exp_log[i]/temp_prob_mix[i+1]
                        -log(temp_prob_mix[i+1])+log(temp_exp_value[i]);
                fprime=trigamma(gamma_param1[i])-1.0/gamma_param1[i];
            }

            if(abs(fprime)<NR_EPSILON ||gamma_param1[i]<=0.0)
                gamma_param1[i]=temp[i];
            else
                gamma_param2[i]=gamma_param1[i]*temp_prob_mix[i+1]/temp_exp_value[i];
        }

        /* Normalizing mixing probabilities */
        sum=0.0;
        for (int i =0;i<num_states; i++)
            sum+=temp_prob_mix[i];

        for (int i =0;i<num_states; i++)
            mix_prob[i]=temp_prob_mix[i]/sum;

        value_contrib = null;
        temp_prob_mix = null;
        temp_exp_value = null;
        temp_exp_log = null;
        temp = null;

        return;
    }

    void UpdateEmissionExponential(Data data,double[][]prob_s){

        /* Temporary variable(s) */
        double temp;
        double sum=0.0;

        temp=0.0;
        for (int n =0;n<data.num_seqs;n++)
            for (int t =0;t<data.sequence[n].seq_length;t++)
                if(!is_missing(data.sequence[n].entry[t].rdata[dim_index[0]]))
                {
                    temp+=prob_s[n][t]*data.sequence[n].entry[t].rdata[dim_index[0]];
                    sum+=prob_s[n][t];
                }

        exp_param1=sum/temp;
    }

    void UpdateEmissionGamma(Data data,double[][]prob_s){

        double f,fprime,ll,ll_old;

        /* Temporary variable(s) */
        double temp,total_mass,total_comb,total_log;

        /* Updating parameters for gamma distribution */

        /* Precomputing the constants */
        total_mass=0.0;
        total_comb=0.0;
        total_log=0.0;

        for (int n =0;n<data.num_seqs;n++)
            for (int t =0;t<data.sequence[n].seq_length;t++)
                if(!is_missing(data.sequence[n].entry[t].rdata[dim_index[0]])&&prob_s[n][t]>0)
                {
                    total_mass+=prob_s[n][t];
                    total_comb+=prob_s[n][t]*data.sequence[n].entry[t].rdata[dim_index[0]];
                    total_log+=prob_s[n][t]*log(data.sequence[n].entry[t].rdata[dim_index[0]]);
                }

       int  iteration_ll=0;
        ll=NEG_INF;
        ll_old=NEG_INF;
        while(iteration_ll==0||ll>ll_old+CONJ_GRAD_EPSILON)
        {
            iteration_ll++;
            ll_old=ll;

            /* Parameters for the gamma components */
            temp=gamma1;

           int iteration=0;
            f=0.0;
            fprime=0.0;
            while(iteration==0||(iteration<MAX_NR_ITERATIONS &&abs(f)>NR_EPSILON&&abs(fprime)>NR_EPSILON))
            {
                iteration++;
                f=total_log/total_mass+log(gamma2)-digamma(gamma1);
                fprime=-trigamma(gamma1);

                gamma1-=f/fprime;
            }

            if(iteration==MAX_NR_ITERATIONS)
                gamma1=temp;

            gamma2=gamma1*total_mass/total_comb;

            /* Computing new log-likelihoood */
            ll=(gamma1-1.0)*total_log/total_mass-gamma2*total_comb/total_mass+gamma1*log(gamma2)-gammaln(gamma1);
        }
    }

    void UpdateEmissionLognormal(Data data,double[][]prob_s){
        double sum_log=0.0;
        double sum_log2=0.0;
        double sum_weight=0.0;

        for (int n =0;n<data.num_seqs;n++)
            for (int t =0;t<data.sequence[n].seq_length;t++)
                if(!is_missing(data.sequence[n].entry[t].rdata[dim_index[0]])&&prob_s[n][t]>0)
                {
                    sum_weight+=prob_s[n][t];
                    sum_log+=prob_s[n][t]*log(data.sequence[n].entry[t].rdata[dim_index[0]]);
                    sum_log2+=prob_s[n][t]*log(data.sequence[n].entry[t].rdata[dim_index[0]])
                            *log(data.sequence[n].entry[t].rdata[dim_index[0]]);
                }

        log_normal1=sum_log/sum_weight;
        log_normal2=(sum_log2-2.0*log_normal1*sum_log+log_normal1*log_normal1*sum_weight)/sum_weight;

    }

    void UpdateEmissionGaussian(Data data,double[][]prob_s,double norm_const){

        /* Zeroing out the mean */
        for (int i =0;i<dim; i++)
            mu[i]=0.0;

        /* Updating the mean */
        for (int i =0;i<dim; i++)
        { /* i-th component of the mean */
            for (int n =0;n<data.num_seqs;n++)
                for (int t =0;t<data.sequence[n].seq_length;t++)
                    mu[i]+=(prob_s[n][t]*data.sequence[n].entry[t].rdata[dim_index[i]]);

            mu[i]/=norm_const;
        } /* i-th component of the mean */

        /* Zeroing out the covariance matrix */
        for (int i =0;i<dim; i++)
            for (int j =0;j<dim; j++)
            {
                /* Storing sigma's values for the case of singular matrix */
                inv_sigma[i][j]=sigma[i][j];
                sigma[i][j]=0.0;
            }

        /* Updating the covariance matrix */
        for (int i =0;i<dim; i++)
            for (int j =0;j<dim; j++)
            {
                for (int n =0;n<data.num_seqs;n++)
                    for (int t =0;t<data.sequence[n].seq_length;t++)
                        sigma[i][j]+=(prob_s[n][t]*
                                (mu[i]-data.sequence[n].entry[t].rdata[dim_index[i]])*
                                (mu[j]-data.sequence[n].entry[t].rdata[dim_index[j]]));
                sigma[i][j]/=norm_const;
            }

        /* Calculating the inverse of the covariance matrix and its determinant */
        det=find_det(sigma,dim);
        if(SINGULAR_FAIL)
        { /* New matrix is singular -- replacing with the old one */
            /* !!! Need better solution !!! */
            for (int i =0;i<dim; i++)
                for (int j =0;j<dim; j++)
                    sigma[i][j]=inv_sigma[i][j];
            det=find_det(sigma,dim);

            if(!is_done)
            { /* Stopping EM */
                is_done=1;
                num_failed++;
            } /* Stopping EM */

        } /* New matrix is singular -- replacing with the old one */

        find_inv(sigma,dim,inv_sigma);

    }

    void UpdateEmissionARGaussian(Data data,double[][]prob_s,double norm_const){

        double[]prev_datum_ave;       // Weighted previous datum
        double[][]W_num,W_denom;    // Matrices used to calculate W


        /* Temporary variable(s) */
        double sum,temp1,temp2;

        /* First state updates */

        /* First state mean */
        for (int i =0;i<dim; i++)
        { /* i-th component of the first state mean */
            first_mu[i]=0.0;
            for (int n =0;n<data.num_seqs;n++)
                for (int t =0;t<data.sequence[n].seq_length;t++)
                    first_mu[i]+=(prob_s[n][t]*data.sequence[n].entry[t].rdata[dim_index[i]]);
            first_mu[i]/=norm_const;
        } /* i-th component of the first state mean */

        /* First state covariance matrix */
        for (int i =0;i<dim; i++)
            for (int j =0;j<dim; j++)
            { /* (i,j)-th entry of the covariance matrix */
                /* Storing old values for the case the new matrix is singular */
                inv_first_sigma[i][j]=first_sigma[i][j];

                /* Updating */
                first_sigma[i][j]=0.0;
                for (int n =0;n<data.num_seqs;n++)
                    for (int t =0;t<data.sequence[n].seq_length;t++)
                        first_sigma[i][j]+=(prob_s[n][t]*
                                (first_mu[i]-data.sequence[n].entry[t].rdata[dim_index[i]])*
                                (first_mu[i]-data.sequence[n].entry[t].rdata[dim_index[j]]));
                first_sigma[i][j]/=norm_const;
            } /* (i,j)-th entry of the covariance matrix */

        /* Calculating the inverse of the first state covariance matrix and its determinant */
        first_det=find_det(first_sigma,dim);
        if(SINGULAR_FAIL)
        { /* New matrix is singular -- replacing with the old one */
            /* !!! Need better solution !!! */
            for (int i =0;i<dim; i++)
                for (int j =0;j<dim; j++)
                    first_sigma[i][j]=inv_first_sigma[i][j];
            first_det=find_det(first_sigma,dim);

            if(!is_done)
            { /* Stopping EM */
                is_done=1;
                num_failed++;
            } /* Stopping EM */

        } /* New matrix is singular -- replacing with the old one */

        find_inv(first_sigma,dim,inv_first_sigma);

        if(SINGULAR_FAIL)
            return;

        /* Auto-regressive parameter updates */

        /* Normalization constant */
        sum=0.0;
        for (int n =0;n<data.num_seqs;n++)
            for (int t =1;t<data.sequence[n].seq_length;t++)
                sum+=prob_s[n][t];

        /* Linear transformation matrix */

        /* Calculating constant related to the linear transformation matrix update */
        prev_datum_ave=new double[dim];

        W_num=new double*[dim];
        for (int i =0;i<dim; i++)
            W_num[i]=new double[dim];

        W_denom=new double*[dim];
        for (int i =0;i<dim; i++)
            W_denom[i]=new double[dim];

        for (int i =0;i<dim; i++)
        {
            prev_datum_ave[i]=0.0;
            for (int n =0;n<data.num_seqs;n++)
                for (int t =1;t<data.sequence[n].seq_length;t++)
                    prev_datum_ave[i]+=prob_s[n][t]*data.sequence[n].entry[t-1].rdata[dim_index[i]];
            prev_datum_ave[i]/=sum;
        }

        /* Updating linear transformation matrix */
        for (int i =0;i<dim; i++)
            for (int j =0;j<dim; j++)
            { /* (i,j)-th entry */
                W_num[i][j]=0.0;
                for (int n =0;n<data.num_seqs;n++)
                    for (int t =1;t<data.sequence[n].seq_length;t++)
                        W_num[i][j]+=(prob_s[n][t]*data.sequence[n].entry[t-1].rdata[dim_index[i]]*
                                (data.sequence[n].entry[t-1].rdata[dim_index[j]]-prev_datum_ave[j]));
            } /* (i,j)-th entry */

        find_inv(W_num,dim,W_denom);

        if(SINGULAR_FAIL)
        { /* New matrix is singular */
            if(!is_done)
            { /* Stopping EM */
                is_done=1;
                num_failed++;
            } /* Stopping EM */
        } /* New matrix is singular */
        else
        {
            for (int i =0;i<dim; i++)
                for (int j =0;j<dim; j++)
                { /* (i,j)-th entry */
                    W_num[i][j]=0.0;
                    for (int n =0;n<data.num_seqs;n++)
                        for (int t =1;t<data.sequence[n].seq_length;t++)
                            W_num[i][j]+=(prob_s[n][t]*data.sequence[n].entry[t].rdata[dim_index[i]]*
                                    (data.sequence[n].entry[t-1].rdata[dim_index[j]]-prev_datum_ave[j]));
                } /* (i,j)-th entry */

            /* Updated W is the product of W_num and W_denom */
            for (int i =0;i<dim; i++)
                for (int j =0;j<dim; j++)
                {
                    W[i][j]=0.0;
                    for(int i1=0;i1<dim; i1++)
                        W[i][j]+=W_num[i][i1]*W_denom[i1][j];
                }
        }


        prev_datum_ave = null;

        for (int i =0;i<dim; i++)
            W_num[i] = null;
        W_num = null;

        for (int i =0;i<dim; i++)
            W_denom[i] = null;
        W_denom = null;

        if(SINGULAR_FAIL)
            return;

        /* Updating translation and covariance */

        /* Normalization constant is the same as used with linear transformation update */

        /* Not updating translation */
        if (BLAH) {
            /* Translation */
            for (int i =0;i<dim; i++)
            { /* i-th component */
                mu[i]=0.0;
                for (int n =0;n<data.num_seqs;n++)
                    for (int t =1;t<data.sequence[n].seq_length;t++)
                    {
                        temp1=0.0;
                        for (int j =0;j<dim; j++)
                            temp1+=W[i][j]*data.sequence[n].entry[t-1].rdata[dim_index[j]];
                        mu[i]+=(prob_s[n][t]*(data.sequence[n].entry[t].rdata[dim_index[i]]-temp1));
                    }

                mu[i]/=sum;
            } /* i-th component */
        }

        /* Noise covariance */
        for (int i =0;i<dim; i++)
            for (int j =0;j<dim; j++)
            { /* (i,j)-th entry */
                /* Storing previous values */
                inv_sigma[i][j]=sigma[i][j];

                /* Updating */
                sigma[i][j]=0.0;
                for (int n =0;n<data.num_seqs;n++)
                    for (int t =1;t<data.sequence[n].seq_length;t++)
                    {
                        temp1=mu[i]-data.sequence[n].entry[t].rdata[dim_index[i]];
                        for(int i1=0;i1<dim; i1++)
                            temp1+=W[i][i1]*data.sequence[n].entry[t-1].rdata[dim_index[i1]];

                        temp2=mu[j]-data.sequence[n].entry[t].rdata[dim_index[j]];
                        for(int i2=0;i2<dim; i2++)
                            temp2+=W[j][i2]*data.sequence[n].entry[t-1].rdata[dim_index[i2]];

                        sigma[i][j]+=(prob_s[n][t]*temp1*temp2);
                    }

                sigma[i][j]/=sum;
            } /* (i,j)-th entry */

        /* Calculating the inverse of the covariance matrix and its determinant */
        det=find_det(sigma,dim);
        if(SINGULAR_FAIL)
        { /* New matrix is singular -- replacing with the old one */
            /* !!! Need better solution !!! */
            for (int i =0;i<dim; i++)
                for (int j =0;j<dim; j++)
                    sigma[i][j]=inv_sigma[i][j];
            det=find_det(sigma,dim);

            if(!is_done)
            { /* Stopping EM */
                is_done=1;
                num_failed++;
            } /* Stopping EM */

        } /* New matrix is singular -- replacing with the old one */

        find_inv(sigma,dim,inv_sigma);

        return;
    }

    void UpdateEmissionGaussianChowLiu(Data data,double[][]prob_s,double norm_const){

        double[][]MI;                  // Mutual information
        int[]best_edge;              // The list of current best edges

        /* Temporary variable(s) */
        double temp1,best_MI;

        /* Zeroing out the mean */
        for (int i =0;i<dim; i++)
            mu[i]=0.0;

        /* Updating the mean */
        for (int i =0;i<dim; i++)
        { /* i-th component of the mean */
            for (int n =0;n<data.num_seqs;n++)
                for (int t =0;t<data.sequence[n].seq_length;t++)
                    mu[i]+=(prob_s[n][t]*data.sequence[n].entry[t].rdata[dim_index[i]]);

            mu[i]/=norm_const;
        } /* i-th component of the mean */

        /* First, computing full covariance matrix */

        /* Zeroing out the covariance matrix */
        for (int i =0;i<dim; i++)
            for (int j =0;j<dim; j++)
                sigma[i][j]=0.0;

        /* Updating the covariance matrix */
        for (int i =0;i<dim; i++)
            for (int j =0;j<dim; j++)
            {
                for (int n =0;n<data.num_seqs;n++)
                    for (int t =0;t<data.sequence[n].seq_length;t++)
                        sigma[i][j]+=(prob_s[n][t]*
                                (mu[i]-data.sequence[n].entry[t].rdata[dim_index[i]])*
                                (mu[j]-data.sequence[n].entry[t].rdata[dim_index[j]]));
                sigma[i][j]/=norm_const;
            }


        /* Now, applying Chow-Liu tree algorithm to the covariance matrix */

        /* Transforming covariance matrix into a correlation matrix */
        for (int i =0;i<dim; i++)
            for (int j =i+1;j<dim; j++)
            {
                sigma[i][j]/=sqrt(sigma[i][i]*sigma[j][j]);
                sigma[j][i]=sigma[i][j];
            }

        /* Building the inverse covariance matrix */
        /* Zeroing out first */
        for (int i =0;i<dim; i++)
            for (int j =0;j<dim; j++)
                inv_sigma[i][j]=0.0;

        /* Starting with the identity matrix */
        for (int i =0;i<dim; i++)
            inv_sigma[i][i]=1.0;

        /* Allocating mutual information lower triangular matrix */
        MI=new double*[dim];
        for (int i =0;i<dim; i++)
            MI[i]=new double[dim];

        /* Array of best edges */
        best_edge=new int[dim-1];

        /* Initializing */

        /* Mutual information */
        for (int i =0;i<dim; i++)
            for (int j =0;j<i; j++)
            {
                MI[i][j]=-0.5*log(1.0-sigma[i][j]*sigma[i][j]);
                MI[j][i]=MI[i][j];
            }

        /* Initializing the list of connected nodes */
        node_used[dim-1]=1;
        for (int i =0;i<dim-1;i++)
            node_used[i]=0;

        /* Initializing the array of best edges */
        /* Current best edge is always from the only "attached" node */
        for (int i =0;i<dim-1;i++)
            best_edge[i]=dim-1;

        /* Inserting dim-1 edges */
        for (int num_edges =0;num_edges<dim-1;num_edges++)
        {
            best_MI=NEG_INF;
            i1=-1;
            for (int i =0;i<dim-1;i++)
                if(!node_used[i])
                    if(MI[i][best_edge[i]]>best_MI)
                    { /* Found a better edge */
                        i1=i;
                        best_MI=MI[i][best_edge[i]];
                    } /* Found a better edge */

            /* Adding the edge */
            node_used[i1]=1;
            if(i1>best_edge[i1])
            {
                edge[num_edges][0]=i1;
                edge[num_edges][1]=best_edge[i1];
            }
            else
            {
                edge[num_edges][0]=best_edge[i1];
                edge[num_edges][1]=i1;
            }

            /* Storing the information about the edge */
            edge_MI[num_edges]=best_MI;

            /* Updating the inverse covariance matrix */
            temp1=sigma[edge[num_edges][0]][edge[num_edges][1]];
            inv_sigma[edge[num_edges][0]][edge[num_edges][0]]+=(temp1*temp1)/(1.0-temp1*temp1);
            inv_sigma[edge[num_edges][1]][edge[num_edges][1]]+=(temp1*temp1)/(1.0-temp1*temp1);
            inv_sigma[edge[num_edges][0]][edge[num_edges][1]]=-temp1/(1.0-temp1*temp1);
            inv_sigma[edge[num_edges][1]][edge[num_edges][0]]=inv_sigma[edge[num_edges][0]][edge[num_edges][1]];

            /* Adjusting the list of best edges to not-yet-connected nodes */
            for (int i =0;i<dim-1;i++)
                if(!node_used[i])
                    if(MI[i][i1]>MI[i][best_edge[i]])
                        best_edge[i]=i1;
        }

        /* Deallocating mutual information */
        for (int i =0;i<dim; i++)
            MI[i] = null;
        MI = null;

        /* Deallocating the array of best edges */
        best_edge = null;

        /* Adjusting the inverse covariance matrix -- dividing by appropriate deviation values */
        for (int i =0;i<dim; i++)
            inv_sigma[i][i]/=sigma[i][i];
        for (int i =0;i<num_edges; i++)
        {
            inv_sigma[edge[i][0]][edge[i][1]]/=sqrt(sigma[edge[i][0]][edge[i][0]]*sigma[edge[i][1]][edge[i][1]]);
            inv_sigma[edge[i][1]][edge[i][0]]=inv_sigma[edge[i][0]][edge[i][1]];
        }

        /* Calculating the new covariance matrix */
        /* !!! May be replaced by a more efficient procedure !!! */
        find_inv(inv_sigma,dim,sigma);
        /* Calculating the inverse of the covariance matrix and its determinant */
        det=find_det(sigma,dim);
        if(SINGULAR_FAIL)
        { /* New matrix is singular */
            if(!is_done)
            { /* Stopping EM */
                is_done=1;
                num_failed++;
            } /* Stopping EM */
        } /* New matrix is singular */

        return;
    }

    void UpdateLogistic(Data data,double[][]prob_s){
        double[][][]function_value;                // Value of each function for each datum (not counting the variable)
        double[][][]lambda_sum_f;                   // Linear sums in the exponents (contribution from each state value)
        double[]sum_function_values;             // Sum of function values over all vectors
        double[][][]p;                             // Probability P(x|y, current parameters)
        double wll=0.0;                          // Current weighted log-likelihood
        double prev_wll=NEG_INF;                 // Previous value of weighted log-likelihood 
        double norm_const=0.0;                   // Normalization constant for data weights
        int iteration=0;                        // Iteration of the conjugate gradient algorithm
        double[]gradient,old_gradient;         // Gradient of the weighted log-likelihood
        double[]xi;                              // Direction of the ascent
        double[][][]xi_sum_f;                      // Linear sums in the exponents (contribution from the direction)
        double gamma,gamma_pr,gamma_fr;        // Coefficient used in determining the direction of the ascent
        double nu;                               // The solution of the linear search problem
        int iteration_nr;                       // Newton-Raphson iteration index
        double d1,d2;                            // First and second derivative of the log-likelihood

        /* Temporary variable(s) */
        double gg,og,oo;
        double temp;
        DataPoint datum;                        // Current data vector
        double max;

        /* Allocating the array of function values for each datum */
        function_value=new double[dim][][];
        for (int i =0;i<dim; i++)
        {
            function_value[i]=new double[data.num_seqs][];
            for (int n =0;n<data.num_seqs;n++)
                function_value[i][n]=new double[data.sequence[n].seq_length];
        }

        /* Allocating the array of linear sums for each datum */
        lambda_sum_f=new double[num_states][][];
        for (int i =0;i<num_states; i++)
        {
            lambda_sum_f[i]=new double[data.num_seqs][];
            for (int n =0;n<data.num_seqs;n++)
                lambda_sum_f[i][n]=new double[data.sequence[n].seq_length];
        }

        /* Allocating the array for sum of features */
        sum_function_values=new double[dim];

        /* Allocating the array of probabilities for data under current model */
        p=new double[num_states][][];
        for (int i =0;i<num_states; i++)
        {
            p[i]=new double[data.num_seqs][];
            for (int n =0;n<data.num_seqs;n++)
                p[i][n]=new double[data.sequence[n].seq_length];
        }

        /* Allocating the vector with current direction for the ascent */
        xi=new double[dim];

        xi_sum_f=new double[num_states][][];
        for (int i =0;i<num_states; i++)
        {
            xi_sum_f[i]=new double[data.num_seqs][];
            for (int n =0;n<data.num_seqs;n++)
                xi_sum_f[i][n]=new double[data.sequence[n].seq_length];
        }

        for (int i =0;i<num_states; i++)
            for (int n =0;n<data.num_seqs;n++)
                for (int t =0;t<data.sequence[n].seq_length;t++)
                    lambda_sum_f[i][n][t]=0.0;

        for (int i =0;i<dim; i++)
            sum_function_values[i]=0.0;

        /* Precomputing values */
        for (int n =0;n<data.num_seqs;n++)
            for (int t =0;t<data.sequence[n].seq_length;t++)
            {
                datum=data.sequence[n].entry[t];

                /* Computing the function value */
                for (int i =0;i<dim; i++)
                {
                    function_value[i][n][t]=1.0;
                    for (int j =1;j<num_features[i];j++)
                        if(feature_index[i][j]<datum.ddim)
                        { /* Categorical feature */
                            if(feature_value[i][j]!=datum.ddata[feature_index[i][j]])
                            {
                                function_value[i][n][t]=0.0;
                                /* No need to look at other features for this function */
                                j=num_features[i];
                            }
                        } /* Categorical feature */
                        else
                            /* Real-valued feature */
                            function_value[i][n][t]*=datum.rdata[feature_index[i][j]-datum.ddim];
                }

                for (int i =0;i<dim; i++)
                    /* Considering function i */
                    /* Computing the linear sums in the exponents */
                    lambda_sum_f[feature_value[i][0]][n][t]+=lambda[i]*function_value[i][n][t];

                /* Computing weighted log-likelihood */
                /* For scaling reasons, finding the max of the exponents */
                max=lambda_sum_f[0][n][t];
                for (int i =1;i<num_states; i++)
                    if(lambda_sum_f[i][n][t]>max)
                        max=lambda_sum_f[i][n][t];

                /* Computing the (scaled) partition function */
                temp=0.0;
                for (int i =0;i<num_states; i++)
                    temp+=exp(lambda_sum_f[i][n][t]-max);

                wll+=prob_s[n][t]*(lambda_sum_f[datum.ddata[0]][n][t]-max-log(temp));

                for (int i =0;i<num_states; i++)
                    p[i][n][t]=exp(lambda_sum_f[i][n][t]-max)/temp;

                /* Computing sum of function values */
                for (int i =0;i<dim; i++)
                    if(datum.ddata[0]==feature_value[i][0])
                        sum_function_values[i]+=prob_s[n][t]*function_value[i][n][t];

                /* Updating the normalization constant for weights */
                norm_const+=prob_s[n][t];
            }

        /* Normalizing log-likelihood by the sum of the weights */
        wll/=norm_const;

        old_gradient=null;gradient=null;   // Not to trigger compiler warnings
        while((wll-prev_wll)/norm_const>cg_epsilon)
        { /* Main loop */
            prev_wll=wll;

            /* Computing the gradient */
            if(iteration>0)
            {
                if(iteration>1)
                    old_gradient = null;
                old_gradient=gradient;
            }

            gradient=new double[dim];
            for (int i =0;i<dim; i++)
                gradient[i]=sum_function_values[i];

            for (int i =0;i<dim; i++)
            {
                for (int n =0;n<data.num_seqs;n++)
                    for (int t =0;t<data.sequence[n].seq_length;t++)
                    {
                        datum=data.sequence[n].entry[t];
                        gradient[i]-=prob_s[n][t]*p[feature_value[i][0]][n][t]*function_value[i][n][t];
                    }
                gradient[i]/=norm_const;
            }

            /* Computing the new direction */
            if(iteration==0)
                for (int i =0;i<dim; i++)
                    xi[i]=-gradient[i];
            else
            {
                gg=0.0;
                og=0.0;
                oo=0.0;
                for (int i =0;i<dim; i++)
                {
                    gg+=gradient[i]*gradient[i];
                    og+=gradient[i]*old_gradient[i];
                    oo+=old_gradient[i]*old_gradient[i];
                }

                gamma_pr=(gg-og)/oo;  // Polak-Ribiere
                gamma_fr=gg/oo;       // Fletcher-Reeves

                if(gamma_pr<-gamma_fr)
                    gamma=-gamma_fr;
                else if(gamma_pr>gamma_fr)
                    gamma=gamma_fr;
                else
                    gamma=gamma_pr;

                for (int i =0;i<dim; i++)
                    xi[i]=gradient[i]-gamma*old_gradient[i];
            }

            /* Line search optimization algorithm */

            /* Pre-computing commonly used values */
            /* Exponent contribution from the new direction */
            for (int n =0;n<data.num_seqs;n++)
                for (int t =0;t<data.sequence[n].seq_length;t++)
                {
                    datum=data.sequence[n].entry[t];
                    for (int i =0;i<num_states; i++)
                        xi_sum_f[i][n][t]=0.0;

                    for (int i =0;i<dim; i++)
                        /* Considering function i */
                        /* Computing the linear sums in the exponents */
                        xi_sum_f[feature_value[i][0]][n][t]+=xi[i]*function_value[i][n][t];
                }

            nu=0.0;
            iteration_nr=0;

            /* Newton-Raphson */
            d1=norm_const;   // Not to trigger compiler warnings
            while((iteration_nr==0||abs(d1)/norm_const>NR_EPSILON)&&iteration_nr<MAX_NR_ITERATIONS )
            {
                iteration_nr++;

                /* First derivative */
                d1=0.0;
                for (int i =0;i<dim; i++)
                    d1+=sum_function_values[i]*xi[i];

                for (int n =0;n<data.num_seqs;n++)
                    for (int t =0;t<data.sequence[n].seq_length;t++)
                        for (int i =0;i<num_states; i++)
                            d1-=prob_s[n][t]*p[i][n][t]*xi_sum_f[i][n][t];

                d2=0.0;
                for (int n =0;n<data.num_seqs;n++)
                    for (int t =0;t<data.sequence[n].seq_length;t++)
                        for (int i =0;i<num_states; i++)
                            d2-=prob_s[n][t]*xi_sum_f[i][n][t]*xi_sum_f[i][n][t]*p[i][n][t]*(1.0-p[i][n][t]);

                nu-=d1/d2;

                /* Updating the probabilities */
                for (int n =0;n<data.num_seqs;n++)
                    for (int t =0;t<data.sequence[n].seq_length;t++)
                    {
                        datum=data.sequence[n].entry[t];

                        for (int i =0;i<dim; i++)
                            /* Considering function i */
                            /* Computing the linear sums in the exponents */
                            lambda_sum_f[feature_value[i][0]][n][t]+=nu*xi[i]*function_value[i][n][t];

                        /* Computing weighted log-likelihood */
                        /* For scaling reasons, finding the max of the exponents */
                        max=lambda_sum_f[0][n][t];
                        for (int i =1;i<num_states; i++)
                            if(lambda_sum_f[i][n][t]>max)
                                max=lambda_sum_f[i][n][t];

                        /* Computing the (scaled) partition function */
                        temp=0.0;
                        for (int i =0;i<num_states; i++)
                            temp+=exp(lambda_sum_f[i][n][t]-max);

                        wll+=prob_s[n][t]*(lambda_sum_f[datum.ddata[0]][n][t]-max-log(temp));

                        for (int i =0;i<num_states; i++)
                            p[i][n][t]=exp(lambda_sum_f[i][n][t]-max)/temp;
                    }
            }

            /* Updating the parameters */
            for (int i =0;i<dim; i++)
                lambda[i]+=nu*xi[i];

            /* Updating sums of features */
      /* Already updated 
	 for (int n=0;n<num_points; n++ )
	 delta_sum_f[n]+=nu*xi_sum_f[n];
      */

            /* Already updated */
            //      wll=0.0;
            /* Computing weighted log-likelihood */
      /*
	for (int n=0;n<num_points; n++ )
	if( data[n][index] )
	wll+=w[n]*(delta_sum_f[n]-log(1.0+exp(delta_sum_f[n])));
	else
	wll-=w[n]*log(1.0+exp(delta_sum_f[n]));
      */

            iteration++;
        } /* Main loop */

        /* Deallocating */
        for (int i =0;i<num_states; i++)
        {
            for (int n =0;n<data.num_seqs;n++)
                xi_sum_f[i][n] = null;
            xi_sum_f[i] = null;
        }
        xi_sum_f = null;

        xi = null;

        if(iteration>0)
        {
            gradient = null;
            if(iteration>1)
                old_gradient = null;
        }

        sum_function_values = null;

        for (int i =0;i<num_states; i++)
        {
            for (int n =0;n<data.num_seqs;n++)
                p[i][n] = null;
            p[i] = null;
        }

        for (int i =0;i<num_states; i++)
        {
            for (int n =0;n<data.num_seqs;n++)
                lambda_sum_f[i][n] = null;
            lambda_sum_f[i] = null;
        }
        lambda_sum_f = null;

        for (int i =0;i<dim; i++)
        {
            for (int n =0;n<data.num_seqs;n++)
                function_value[i][n] = null;
            function_value[i] = null;
        }
        function_value = null;

        return;
    }

}