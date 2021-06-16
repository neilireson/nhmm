package uk.ac.shef.wit.nhmm;

import java.io.File;
import java.io.IOException;
import java.io.PrintStream;

import static java.lang.Double.isFinite;
import static java.lang.Math.*;
import static uk.ac.shef.wit.nhmm.Constants.*;
import static uk.ac.shef.wit.nhmm.Constants.DistributionType.*;
import static uk.ac.shef.wit.nhmm.Data.is_missing;
import static uk.ac.shef.wit.nhmm.Data.missing_value;
import static uk.ac.shef.wit.nhmm.Simulation.generateBernoulli;

public class HMM {
    final Parameters params;
    final int num_variables;             // Number of latent variables
    final int num_states;                // Number of states for each variable
    Distribution transition; // Transition distribution
    Distribution[][] emission; // Emission distribution
    double ll;                      // Log-likelihood
    double lp;                      // Log-posterior
    double lp_normalized;           // Log-posterior per dimension of data
    double bic;                     // BIC score

    /* EM related */
    double[][][][] log_pb;             // Log-probability values P(R_nt|S_nt=i)
    double[][] norm_const;           // Normalization factors


    HMM(Parameters params) {
        /* Basic instantiation -- transition and emission added separately */

        this.params = params;
        if (params.transition != null) {
            num_variables = params.transition.dim;
            num_states = params.transition.num_states;
        } else {
            num_variables = params.num_variables;
            num_states = params.num_states;
        }

        emission = new Distribution[num_variables][];
        for (int i = 0; i < num_variables; i++)
            emission[i] = new Distribution[num_states];

        /* Initializing EM arrays */

        /* Allocating array of log-probabilities of data for each value of each latent variable */
        log_pb = new double[num_variables][][][];
        for (int i = 0; i < num_variables; i++)
            log_pb[i] = new double[num_states][][];

        /* Allocating array of aggregates of weights for each value of each latent variable */
        norm_const = new double[num_variables][];
        for (int i = 0; i < num_variables; i++)
            norm_const[i] = new double[num_states];

        ll = NEG_INF;
        lp = NEG_INF;
        lp_normalized = NEG_INF;
        bic = NEG_INF;
    }

    void close() {

        /* Deallocating emission structure */
        for (int i = 0; i < num_variables; i++) {
            for (int j = 0; j < num_states; j++)
                emission[i][j] = null;
            emission[i] = null;
        }
        emission = null;

        /* Deallocating transition structure */
        transition = null;

        /* Deallocating arrays of log-probabilities and weights */
        for (int i = 0; i < num_variables; i++)
            log_pb[i] = null;
        log_pb = null;

        for (int i = 0; i < num_variables; i++)
            norm_const[i] = null;
        norm_const = null;
    }

    HMM copy() {
        /* Copy constructor */
        HMM model;

        model = new HMM(params);

        /* Copying emissions */
        for (int i = 0; i < num_variables; i++)
            for (int j = 0; j < num_states; j++)
                model.emission[i][j] = emission[i][j].copy();

        /* Copying input distributions */
        model.transition = transition.copy();

        model.ll = ll;
        model.lp = lp;
        model.lp_normalized = lp_normalized;
        model.bic = bic;

        return (model);
    }

    void Initialize(Data data, Data input_data) {

        /* Initializing parameters for the HMM */
        transition.Initialize(input_data);

        /* Initializing emission distribution */
        for (int i = 0; i < num_variables; i++)
            for (int j = 0; j < num_states; j++)
                emission[i][j].Initialize(data);

    }

    void InitializeByEM(Data data, Data input_data) {
  /* Initializes the model by estimating the parameters from the data according to
     a simpler sub-model */

        Distribution trans_copy;  // Copy of the transition distribution
        int num_failed = 0;    // Number of failed runs so far -- should not change in the end
        int num_prev_failed;      // Temporary value of num_failed;

        /* Saving the original distribution structure */
        trans_copy = transition;

        /* Simplifying the model by removing the dependence on the input variable for latent state chain */
        transition = trans_copy.Project();

        if (transition != null) { /* Models with complex input layer */
            num_prev_failed = num_failed;
            do {
                /* Initializing the compressed model */
                Initialize(data, input_data);

                EM(data, input_data, EM_EPSILON);
                if (num_failed != num_prev_failed)
                    /* Run failed -- setting the counter of failed EM runs back to the previous value */
                    num_failed = num_prev_failed;
                else
                    /* Run is ok -- making the counters differ */
                    num_prev_failed--;
            }
            while (num_failed == num_prev_failed);

            /* Expanding the model to the original type */
            transition.Expand(trans_copy);
            if (BLAH) {
                transition.WriteToFile2(System.out);
                trans_copy.WriteToFile2(System.out);
            }
            /* Updating the pointers */
            transition = null;
            transition = trans_copy;

        } /* Models with complex input layer */ else
            transition = trans_copy;
    }

    double log_likelihood(Data data, Data input_vars) {

        ll = 0.0;
        for (int n = 0; n < data.num_seqs; n++)
            if (input_vars != null)
                ll += log_likelihood_sequence(data.sequence[n], input_vars.sequence[n], n, 0);
            else
                ll += log_likelihood_sequence(data.sequence[n], null, n, 0);

        //  ll_normalized=ll/data.total_entries();

        return (ll);
    }

    double log_posterior(Data data, Data input) {
        /* Computing log-posterior of the parameter set (with unnormalized prior) */

        double log_prior = 0.0;

        /* First, computing the log-likelihood */
        ll = log_likelihood(data, input);

        /* Computing log-prior from emission distributions */
        for (int v = 0; v < num_variables; v++)
            for (int i = 0; i < num_states; i++)
                /* Computing log-prior contribution for state i of variable v */
                log_prior += emission[v][i].log_prior();

        /* Computing log-prior from transition distribution */
        log_prior += transition.log_prior();

        lp = ll + log_prior;
        lp_normalized = lp / data.total_entries();

        return (lp);
    }

    double hole_score(Data data, Data input_data, int analysis_type) {
        double score = 0.0;

        /* Sequences are assumed iid */
        for (int i = 0; i < data.num_seqs; i++)
            if (input_data != null)
                score += hole_score_sequence(data.sequence[i], input_data.sequence[i], analysis_type, i);
            else
                score += hole_score_sequence(data.sequence[i], null, analysis_type, i);

        return (score);
    }

    double hole_score_sequence(Sequence data, Sequence input_data, int analysis_type, int index) {
        double score = 0.0;            // Per bit log-likelihood score of the data
        double ll_unaltered;         // Log-likelihood of the unaltered sequence

        /* Temporary variable(s) */
        int covered_value;
        int entry_dim;
        double current_ll;

        /* Calculating the log-likelihood of the sequence */
        ll_unaltered = log_likelihood_sequence(data, input_data, index, 0);

        if (HOLE_VERBOSE) {
            System.out.format(" Log-likelihood for sequence %d = %0.6le\n", index + 1, ll_unaltered);
        }

        /* Calculating the scores */
        /* !!! Determining the dimension of the discrete-valued entries !!! */
        entry_dim = data.entry[0].ddim;

        /* Starting changing values in the end of the sequence and moving towards the front */
        for (int t = 0; t < data.seq_length; t++) { /* t-th sequence */
            /* !!! Hack -- should be replaced by something modular !!! */
            for (int m = 0; m < entry_dim; m++) {
                /* !!! Assuming two possible values (0 and 1) !!! */

                /* Making part of the entry different from before */
                covered_value = data.entry[t].ddata[m];
                if (covered_value == 0)
                    data.entry[t].ddata[m] = 1;
                if (covered_value == 1)
                    data.entry[t].ddata[m] = 0;

                if (!is_missing(covered_value)) {
                    /* Calculating the score for the altered sequence */
                    current_ll = log_likelihood_sequence(data, input_data, index, t);
                    if (HOLE_VERBOSE) {
                        System.out.format("%0.2le ", current_ll);
                    }
                    /* Updating the score */
                    switch (analysis_type) {
                        case POKING_TYPE_LOGP:
                            /* Log-probability of the correct prediction */
                            score += -log(exp(current_ll - ll_unaltered) + 1.0);
                            break;
                        case POKING_TYPE_PRED:
                            /* Number of correct predictions */
                            if (ll_unaltered > current_ll)
                                score += 1.0;
                            break;
                        default:
                            ;
                    }
                } else {
                    if (HOLE_VERBOSE) {
                        System.out.format("-1 ");
                    }
                }

                /* Putting the original value back */
                data.entry[t].ddata[m] = covered_value;
            }
            if (HOLE_VERBOSE) {
                System.out.format("\n");
            }
            /* Resetting intermediate values needed to compute the log-likelihood */
            ll_unaltered = log_likelihood_sequence(data, input_data, index, t);
        } /* t-th sequence */

        return (score);
    }

    void MissingIndividualMarginals(Data data, Data input_data, PrintStream output, int mode_type) {
        /* Computing and saving to a file marginal probabilities for all missing entries */

        double prob;               // Probability of one for missing data

        double baseline_ll;        // log-likelihood of a sequence in question
        double new_ll;

        double[] sequence_prob = null;
        int num_missing_entries = 0;
        Sequence temp_sequence = null;

        /* !!! Assuming all variables have the same number of values -- binary !!! */
        for (int n = 0; n < data.num_seqs; n++) { /* For each sequence */
            if (input_data != null)
                baseline_ll = log_likelihood_sequence(data.sequence[n], input_data.sequence[n], n, 0);
            else
                baseline_ll = log_likelihood_sequence(data.sequence[n], null, n, 0);

            /* The entries in the sequences are processed in reverse order */
            if (mode_type == 0) { /* Marginal probability mode */
                /* Storing the probabilities of missing entries as they arrive and then printing them in opposite order */
                num_missing_entries = 0;
                sequence_prob = new double[data.sequence[n].seq_length * data.sequence[n].entry[0].ddim];
            } /* Marginal probability mode */ else
                /* Marginal prediction mode */
                temp_sequence = new Sequence(data.sequence[n].seq_length);

            for (int t = data.sequence[n].seq_length - 1; t >= 0; t--) { /* Entry t */
                if (mode_type == 1 && temp_sequence != null)
                    /* Marginal prediction mode */
                    temp_sequence.entry[t] = new DataPoint(data.sequence[n].entry[t].ddim, data.sequence[n].entry[t].rdim);

                /* Looping backwards so that when the probabilities are printed, they are in the right order */
                for (int i = data.sequence[n].entry[t].ddim - 1; i >= 0; i--)
                    if (is_missing(data.sequence[n].entry[t].ddata[i])) { /* Computing probability of missing entry being 1 */
                        data.sequence[n].entry[t].ddata[i] = 1;
                        if (input_data != null)
                            new_ll = log_likelihood_sequence(data.sequence[n], input_data.sequence[n], n, t);
                        else
                            new_ll = log_likelihood_sequence(data.sequence[n], null, n, t);

                        prob = exp(new_ll - baseline_ll);
                        if (mode_type == 0) { /* Marginal probability mode */
                            sequence_prob[num_missing_entries] = prob;
                            num_missing_entries++;
                        } /* Marginal probability mode */ else
                            /* Marginal prediction mode */
                            if (prob > 0.5)
                                temp_sequence.entry[t].ddata[i] = 1;
                            else
                                temp_sequence.entry[t].ddata[i] = 0;

                        /* Returning the value to missing */
                        data.sequence[n].entry[t].ddata[i] = missing_value((int) 0);
                    } /* Computing probability of missing entry being 1 */ else
                        /* Entry is not missing */
                        if (mode_type == 1)
                            /* Marginal prediction mode */
                            temp_sequence.entry[t].ddata[i] = data.sequence[n].entry[t].ddata[i];

                /* Recomputing alphas */
                if (input_data != null)
                    log_likelihood_sequence(data.sequence[n], input_data.sequence[n], n, t);
                else
                    log_likelihood_sequence(data.sequence[n], null, n, t);
            } /* Entry t */

            /* Outputting the values for the marginals */
            if (mode_type == 0) { /* Marginal probability mode */
                /* Printing the probabilities */
                for (int i = num_missing_entries - 1; i >= 0; i--)
                    System.out.format("%.12f\n", sequence_prob[i]);

                sequence_prob = null;
            } /* Marginal probability mode */ else { /* Marginal prediction mode */
                /* Outputting filled-in examples */
                temp_sequence.WriteToFile(output);

                /* Deallocating temporary sequence */
                temp_sequence = null;
            } /* Marginal prediction mode */
        } /* For each sequence */

        return;
    }


    void StateProbabilities(Data data, Data input_data, PrintStream output) {
        /* Computing and writing to a file P(S_nt|R_n,X_n) */

        EStep(data, input_data);
        CalculateSummaries(data);

        /* Computing the probabilities */
        /* !!! !!! */
        for (int n = 0; n < data.num_seqs; n++)
            for (int t = 0; t < data.sequence[n].seq_length; t++) {
                for (int i = 0; i < num_states; i++)
                    System.out.format("%.12f\t", transition.uni_prob[0][i][n][t]);
                System.out.format("\n");
            }

    }

    void WriteToFile(PrintStream output) {
        /* Records model parameters into a file as 12-digit doubles */

        /* File clarification */
        System.out.format("%c This file contains the parameters of the HMM\n", COMMENT_SYMBOL);

        /* Transition distribution */
        System.out.format("%c Transition probability:\n", COMMENT_SYMBOL);
        if (!params.short_dist_display)
            transition.WriteToFile(output);
        else
            transition.WriteToFile2(output);
        System.out.format("\n");

        /* Emission distributions */
        System.out.format("%c Parameters for the emission distributions for different possible hidden states:\n",
                COMMENT_SYMBOL);
        for (int i = 0; i < num_variables; i++) {
            System.out.format("%c Variable #%d:\n", COMMENT_SYMBOL, i + 1);
            for (int j = 0; j < num_states; j++) {
                System.out.format("%c State #%d:\n", COMMENT_SYMBOL, j + 1);
                if (!params.short_dist_display)
                    emission[i][j].WriteToFile(output);
                else
                    emission[i][j].WriteToFile2(output);
            }
            System.out.format("\n");
        }
        System.out.format("\n");

        /* Displaying the log-likelihood for the data set */
        System.out.format("%c Log-likelihood of the data set: %.12f\n", COMMENT_SYMBOL, ll);

        /* Displaying the log-posterior */
        System.out.format("%c Log-posterior of the data set: %.12f\n", COMMENT_SYMBOL, lp);

        /* Displaying the per-dim log-likelihood for the data set */
        System.out.format("%c Per-dimension log-posterior of the data set: %.12f\n",
                COMMENT_SYMBOL, lp_normalized);

        /* Displaying number of free parameters */
        System.out.format("%c Number of free (real-valued) parameters: %d\n",
                COMMENT_SYMBOL, num_params());

        /* Displaying BIC score for the model and set */
        System.out.format("%c Bayes Information Criterion score: %.12f\n", COMMENT_SYMBOL, bic);
    }

    void WriteToFileBare(PrintStream output) {
        /* Records model parameters into a file as 12-digit doubles */

        /* Transition distribution */
        transition.WriteToFileBare(output);

        /* Emission distributions */
        for (int i = 0; i < num_variables; i++)
            for (int j = 0; j < num_states; j++)
                emission[i][j].WriteToFileBare(output);
    }

    int num_params() {
        /* Calculating the number of free parameters for the model */
        int num = 0;

        /* Accounting for output layer */
        for (int i = 0; i < num_variables; i++)
            for (int j = 0; j < num_states; j++)
                num += emission[i][j].num_params();

        /* Accounting for transition layer */
        num += transition.num_params();

        return (num);
    }

    double BIC(int num_points) {
        /* Calculating Bayes Inference Criterion score */

        return (log((double) (num_points)) * (double) (num_params()) - 2 * ll);
    }

    void PostProcess() {
        /* Post training rearrangement of parameters */

        /* So far, may need to rearrange only the emission probabilities */
        for (int i = 0; i < num_variables; i++)
            for (int j = 0; j < num_states; j++)
                emission[i][j].PostProcess();
    }

    double[] KL(HMM q) {
  /* Computes the entropy of the current model and the Kullback-Leibler divergence between the current model and a model q
     for a very narrow class of models */
        double[] kl = new double[2];
        double[] kl_ll;

        Distribution[] temp_dist;      // Distributions in question

        kl[0] = 0.0;
        kl[1] = 0.0;

        /* First verifying that we can compute KL in closed form */
        if (transition.type == bernoulli || q.transition.type == bernoulli)
            if (num_variables == 1 && q.num_variables == 1)
                if (num_states == 1 && q.num_states == 1) {
                    temp_dist = new Distribution[2];
                    temp_dist[0] = emission[0][0];
                    temp_dist[1] = q.emission[0][0];

                    kl_ll = emission[0][0].TrueLogLikelihood(temp_dist, 2);
                    kl[0] = -kl_ll[0];
                    kl[1] = kl_ll[0] - kl_ll[1];

                    kl_ll = null;

                    temp_dist = null;
                }

        return (kl);
    }

    Data FillIn(Data data) {
        Data new_data = null;

        if (num_states == 1 && num_variables == 1) {
            new_data = new Data(data.num_seqs, "filled-in", false);
            for (int n = 0; n < data.num_seqs; n++) {
                new_data.sequence[n] = new Sequence(data.sequence[n].seq_length);
                for (int t = 0; t < data.sequence[n].seq_length; t++)
                    new_data.sequence[n].entry[t] = emission[0][0].FillIn(data.sequence[n].entry[t]);
            }
        }

        return (new_data);
    }

    StateData viterbi(Data data, Data input, int num_best_seqs) {
  /* Performing Viterbi algorithm to find most likely sequence(s) of states given the data
     and the model. */

        /* Variables */
        StateData states;            // Output structure

        double[] log_first;            // Logs of the probabilities of the first state
        double[][] log_trans;           // Logs of the transition matrix entries
        double[][] log_pb;              // Logs of the P(O_t|S_i)

        StateSequence[] best_seq;     // Structure holding best sequences for a given final state
        StateSequence[] new_best_seq; // Structure holding best sequences during the update

        /* Temporary variable(s) */
        int[] long_array_temp;
        StateSequence[] temp_seq;

        /* Allocating the structure for the state sequences */
        states = new StateData(data.num_seqs);

        /* Allocating the structures for the log-values */
        log_first = new double[num_states];
        log_trans = new double [num_states][];
        for (int i = 0; i < num_states; i++)
            log_trans[i] = new double[num_states];

        if (transition.type == DIST_CONDBERNOULLI || transition.type == bernoulli)
            /* Calculating values of the logs of the first state or mixing parameters */
            for (int i = 0; i < num_states; i++)
                log_first[i] = transition.log_state_prob[i];

        if (transition.type == DIST_CONDBERNOULLI)
            /* Calculating the values of the logs of the transition matrix entries */
            for (int i = 0; i < num_states; i++)
                for (int j = 0; j < num_states; j++)
                    log_trans[i][j] = transition.log_cond_state_prob[i][j];

        if (transition.type == bernoulli)
            /* Setting log transition values for the mixture model */
            for (int i = 0; i < num_states; i++)
                for (int j = 0; j < num_states; j++)
                    log_trans[i][j] = log_first[j];

        /* Allocating log_pb */
        log_pb = new double[num_states][];

        /* Allocating best_seq */
        best_seq = new StateSequence[num_states];

        /* Allocating new_best_seq */
        new_best_seq = new StateSequence[num_states];

        /* Starting the loop on each of the sequences */
        for (int seq_index = 0; seq_index < states.num_seqs; seq_index++) { /* Generating best missing value sequences for each of the input sequences */

            if (VITERBI_VERBOSE) {
                System.out.format("Analyzing best state(s) for sequence #%d\n", seq_index + 1);
            }

            /* Allocating states[i] */
            states.sequence[seq_index] = new StateSequence(data.sequence[seq_index].seq_length, num_best_seqs);

            /* Calculating log of P(O_t|S_t) */
            for (int i = 0; i < num_states; i++) {
                log_pb[i] = new double[data.sequence[seq_index].seq_length];
                for (int t = 0; t < data.sequence[seq_index].seq_length; t++)
                    if (t == 0)
                        log_pb[i][t] = emission[0][i].log_prob(data.sequence[seq_index].entry[t], null);
                    else
                        log_pb[i][t] = emission[0][i].log_prob(data.sequence[seq_index].entry[t],
                                data.sequence[seq_index].entry[t - 1]);
            }

            for (int i = 0; i < num_states; i++)
                best_seq[i] = new StateSequence(1, num_best_seqs);

            /* Initializing best state sequences and corresponding log-likelihoods */
            /* Considering sequences of length 1 */
            if (transition.type == DIST_TRANSLOGISTIC || transition.type == DIST_LOGISTIC) {
                transition.probLogistic(input.sequence[seq_index].entry[0], -1, log_first);
                for (int i = 0; i < num_states; i++)
                    log_first[i] = log(log_first[i]);
            }

            for (int i = 0; i < num_states; i++) {
                long_array_temp = new int[1];
                long_array_temp[0] = i;
                best_seq[i].AddSequence(long_array_temp, log_first[i] + log_pb[i][0]);
            }

            for (int t = 1; t < data.sequence[seq_index].seq_length; t++) { /* Considering subsequences of length t+1 */

                /* Allocating new structures for new_best_seq for each final state */
                for (int i = 0; i < num_states; i++)
                    new_best_seq[i] = new StateSequence(t + 1, num_best_seqs);

                for (int i = 0; i < num_states; i++)
                    if (transition.type == DIST_TRANSLOGISTIC || transition.type == DIST_LOGISTIC) {
                        transition.probLogistic(input.sequence[seq_index].entry[t], i, log_trans[i]);
                        for (int j = 0; j < num_states; j++)
                            log_trans[i][j] = log(log_trans[i][j]);
                    }

                for (int j = 0; j < num_states; j++) { /* State of S_t+1 is j */
                    for (int i = 0; i < num_states; i++) { /* State of S_t is i */
                        for (int k = 0; k < best_seq[i].num_best; k++) { /* For each of the best sequences ending in state j */

                            /* Generating new sequences */
                            long_array_temp = new int[t + 1];

                            /* Copying the prefix */
                            for (int p = 0; p < t; p++)
                                long_array_temp[p] = best_seq[i].state[k][p];

                            /* Adding the latest state (j) */
                            long_array_temp[t] = j;

                            /* Trying to add the sequence to the list of the best */
                            new_best_seq[j].AddSequence(long_array_temp,
                                    best_seq[i].ll[k] + log_trans[i][j] + log_pb[j][t]);
                        } /* For each of the best sequences ending in state j */
                    } /* State of S_t is i */
                } /* State of S_t+1 is j */

                /* Moving new_best_seq into best_seq */
                temp_seq = best_seq;
                best_seq = new_best_seq;
                new_best_seq = temp_seq;

                /* Deallocating new structures for new_best_seq for each final state */
                for (int i = 0; i < num_states; i++)
                    new_best_seq[i] = null;
            } /* Considering subsequences of length t+1 */

            /* Finding the best sequences and likelihoods */
            for (int j = 0; j < num_states; j++)
                /* State of S_T is j */

                for (int k = 0; k < best_seq[j].num_best; k++) { /* k-th best sequence of ending in state j of length T */

                    /* Copying the sequence into a temp variable */
                    long_array_temp = new int[data.sequence[seq_index].seq_length];
                    for (int p = 0; p < data.sequence[seq_index].seq_length; p++)
                        long_array_temp[p] = best_seq[j].state[k][p];

                    /* Trying to add the sequence to the list of the best */
                    states.sequence[seq_index].AddSequence(long_array_temp, best_seq[j].ll[k]);
                } /* k-th best sequence of ending in state j of length T */

            /* Deallocating best_seq */
            for (int i = 0; i < num_states; i++)
                best_seq[i] = null;

            /* Deallocating log_pb */
            for (int i = 0; i < num_states; i++)
                log_pb[i] = null;

            if (VITERBI_VERBOSE) {
                System.out.format("Optimal state(s) for sequence #%d calculated\n", seq_index + 1);
            }

        } /* Generating best missing value sequences for each of the input sequences */

        /* Deallocating variables */
        new_best_seq = null;

        best_seq = null;

        log_pb = null;

        for (int i = 0; i < num_states; i++)
            log_trans[i] = null;
        log_trans = null;

        log_first = null;

        return (states);
    }

    int[][] simulate_states(Data input, int num_seqs, int num_entries) {
        /* Simulating sequences of hidden states according to the model theta */

        /* Variables */
        int[][] states;                // Output structure

        double[] prob;                 // Probability distribution for S_nt given previous S_n,t-1 and Xs

        /* Allocating the structure for the state sequences */
        if (input != null)
            states = new int[input.num_seqs][];
        else
            states = new int[num_seqs][];

        /* Allocating probability distribution for P(S_nt|S_n,t-1,Xs) */
        prob = new double[num_states];

        for (int n = 0; n < num_seqs; n++) { /* Generating state sequence n */

            /* Allocating states[i] */
            if (input != null)
                num_entries = input.sequence[n].seq_length;

            states[n] = new int[num_entries];

            for (int t = 0; t < num_entries; t++) { /* Generating state S_nt */

                /* Resetting probability distribution for the state */
                switch (transition.type) {
                    case DIST_CONDBERNOULLI:
                    case DIST_CONDBERNOULLIG:
                        if (t == 0)
                            prob = transition.state_prob;
                        else
                            prob = transition.cond_state_prob[states[n][t - 1]];
                        break;
                    case DIST_TRANSLOGISTIC:
                        if (t == 0)
                            transition.probLogistic(input.sequence[n].entry[t], -1, prob);
                        else
                            transition.probLogistic(input.sequence[n].entry[t], states[n][t - 1], prob);
                        break;
                    case bernoulli:
                        prob = transition.state_prob;
                        break;
                    case DIST_LOGISTIC:
                        transition.probLogistic(input.sequence[n].entry[t], -1, prob);
                        break;
                    default:
                        ;
                }

                /* Simulating state S_nt */
                states[n][t] = generateBernoulli(prob, num_states);
            } /* Generating state S_nt */

        } /* Generating state sequence n */

        /* Deallocating structures */
        if (input != null)
            /* Deallocating distribution P(S_nt|S_n,t-1,Xs) */
            prob = null;

        return (states);
    }

    int[] simulate_state_sequence_gibbs(Sequence data, Sequence input, int index, boolean do_init) {
  /* Simulating a sequence of hidden states according to the model and the
     COMPLETE input and output sequence */

        /* Variables */
        int[] state;                 // Output structure
        double[] prob;                // Probability distribution for S_t given everyting instanstiated

        /* Temporary variable(s) */
        double sum;

        /* Allocating the structure for the state sequences */
        state = new int[data.seq_length];

        /* Allocating the array of probabilities */
        prob = new double[num_states];

        if (do_init) {

            /* Calculating emission probabilities */
            CalculateProbEmissionsSequence(data, index, 0);

            switch (transition.type) {
                case DIST_LOGISTIC:
                    transition.CalculateTransitionValuesSequence(input, index, 0);
                    break;
                case DIST_TRANSLOGISTIC:
                    transition.CalculateTransitionValuesSequence(input, index, 0);
                    transition.CalculateForwardUpdatesSequence(data, index, 0, log_pb);
                    break;
                case DIST_CONDBERNOULLI:
                case DIST_CONDBERNOULLIG:
                    transition.CalculateForwardUpdatesSequence(data, index, 0, log_pb);
                    break;
                default:
                    ;
            }
        }

        for (int t = data.seq_length - 1; t >= 0; t--) { /* Simulating hidden state S_t */
            sum = 0.0;
            switch (transition.type) {
                case bernoulli:
                    /* Homogeneous mixture model */
                    for (int i = 0; i < num_states; i++) {
                        prob[i] = exp(log_pb[0][i][index][t]) * transition.state_prob[i];
                        sum += prob[i];
                    }
                    break;
                case DIST_LOGISTIC:
                    /* Nonhomogeneous mixture model */
                    for (int i = 0; i < num_states; i++) {
                        prob[i] = exp(log_pb[0][i][index][t] + transition.log_p_tr[i][0][index][t]);
                        sum += prob[i];
                    }
                    break;
                case DIST_CONDBERNOULLI:
                case DIST_CONDBERNOULLIG:
                    /* Hidden Markov model */
                    if (t == data.seq_length - 1)
                        /* Last hidden variable */
                        for (int i = 0; i < num_states; i++) {
                            prob[i] = exp(transition.log_fwd_update[i][index][t]);
                            sum += prob[i];
                        }
                    else
                        /* Not the last hidden variable */
                        for (int i = 0; i < num_states; i++) {
                            prob[i] = exp(transition.log_fwd_update[i][index][t] + transition.log_cond_state_prob[i][state[t + 1]]);
                            sum += prob[i];
                        }
                    break;
                case DIST_TRANSLOGISTIC:
                    /* Nonhomogeneous hidden Markov model */
                    if (t == data.seq_length - 1)
                        /* Last hidden variable */
                        for (int i = 0; i < num_states; i++) {
                            prob[i] = exp(transition.log_fwd_update[i][index][t]);
                            sum += prob[i];
                        }
                    else
                        /* Not the last hidden variable */
                        for (int i = 0; i < num_states; i++) {
                            prob[i] = exp(transition.log_fwd_update[i][index][t] + transition.log_p_tr[state[t + 1]][i][index][t + 1]);
                            sum += prob[i];
                        }
                    break;
                default:
                    ;
            }

            /* Normalizing probabilities */
            for (int i = 0; i < num_states; i++)
                prob[i] /= sum;

            /* Simulating state t */
            state[t] = generateBernoulli(prob, num_states);
        } /* Simulating hidden state S_t */

        prob = null;

        return (state);
    }

    int[][][] simulate_latent_sequence_po(Data data, Data input, int num_sims) {
        int[][][] S;              // Sequence of latent states
        double[] prob;           // Current probability vector

        /* Temporary variable(s) */
        double sum;

        prob = new double[num_states];

        /* Allocating array of latent state values */
        S = new int[num_sims][][];
        for (int s = 0; s < num_sims; s++) {
            S[s] = new int[data.num_seqs][];
            for (int n = 0; n < data.num_seqs; n++)
                S[s][n] = new int[data.sequence[n].seq_length];
        }

        /* Allocating necessary structures */
        AllocateAlpha(data, input);

        for (int n = 0; n < data.num_seqs; n++) {
            /* Computing forward updates */
            CalculateProbEmissionsSequence(data.sequence[n], n, 0);
            transition.CalculateTransitionValuesSequence(input.sequence[n], n, 0);
            transition.CalculateForwardUpdatesSequence(data.sequence[n], n, 0, log_pb);
        }

        for (int s = 0; s < num_sims; s++) {
            for (int n = 0; n < data.num_seqs; n++) {
                for (int t = data.sequence[n].seq_length - 1; t >= 0; t--) { /* Simulating hidden state S_t */

                    sum = 0.0;
                    switch (transition.type) {
                        case bernoulli:
                            /* Mixture */
                            for (int i = 0; i < num_states; i++) {
                                prob[i] = exp(log_pb[0][i][n][t]) * transition.state_prob[i];
                                sum += prob[i];
                            }
                            break;
                        case DIST_LOGISTIC:
                            /* Non-homogeneous mixture */
                            for (int i = 0; i < num_states; i++) {
                                prob[i] = exp(log_pb[0][i][n][t] + transition.log_p_tr[i][0][n][t]);
                                sum += prob[i];
                            }
                            break;
                        case DIST_CONDBERNOULLI:
                        case DIST_CONDBERNOULLIG:
                            /* HMM */
                            if (t == data.sequence[n].seq_length - 1)
                                /* Last hidden variable */
                                for (int i = 0; i < num_states; i++) {
                                    prob[i] = exp(transition.log_fwd_update[i][n][t]);
                                    sum += prob[i];
                                }
                            else
                                /* Not the last hidden variable */
                                for (int i = 0; i < num_states; i++) {
                                    prob[i] = exp(transition.log_fwd_update[i][n][t]) * transition.cond_state_prob[i][S[s][n][t + 1]];
                                    sum += prob[i];
                                }
                            break;
                        case DIST_TRANSLOGISTIC:
                            /* NHMM */
                            if (t == data.sequence[n].seq_length - 1)
                                /* Last hidden variable */
                                for (int i = 0; i < num_states; i++) {
                                    prob[i] = exp(transition.log_fwd_update[i][n][t]);
                                    sum += prob[i];
                                }
                            else
                                /* Not the last hidden variable */
                                for (int i = 0; i < num_states; i++) {
                                    prob[i] = exp(transition.log_fwd_update[i][n][t] + transition.log_p_tr[S[s][n][t + 1]][i][n][t + 1]);
                                    sum += prob[i];
                                }
                            break;
                        default:
                            ;
                    }

                    /* Normalizing probabilities */
                    for (int i = 0; i < num_states; i++)
                        prob[i] /= sum;

                    /* Simulating latent state value */
                    S[s][n][t] = generateBernoulli(prob, num_states);
                } /* Simulating hidden state S_t */
            }
        }

        /* Deallocating necessary structures */
        DeallocateAlpha(data, input);

        prob = null;

        return (S);
    }

    void InitializeMissingEntriesSequence(Sequence data) {
        /* !!! Assinging initial values uniformly at random */
        /* !!! Assuming binary values */

        for (int m = 0; m < data.num_missing_discrete_entries; m++) {
            int t = data.missing_discrete[m][0];
            int i = data.missing_discrete[m][1];

            if (Constants.drand48() < 0.5)
                data.entry[t].ddata[i] = 0;
            else
                data.entry[t].ddata[i] = 1;
        }

        return;
    }

    void ReadParameters(File input_file) throws IOException {

        /* Reading in the transition distribution */
        transition.ReadParameters2(input_file);
        if (INPUT_VERBOSE) {
            System.out.format("Transition distribution parameters read\n");
        }

        /* Reading in the emissions */
        for (int i = 0; i < num_variables; i++)
            for (int j = 0; j < num_states; j++)
                if (!params.short_dist_display)
                    emission[i][j].ReadParameters(input_file);
                else
                    emission[i][j].ReadParameters2(input_file);
        if (INPUT_VERBOSE) {
            System.out.format("Emission distribution parameters read\n");
        }

    }


    private int num_failed = 0;
    void EM(Data data, Data input_data, double precision) {

        /* Variables */
        double old_lp_normalized; // Previous log-posterior

        /* Previous emission and transition variables */
        Distribution[][] prev_emission;
        Distribution prev_transition;

        int num_emissions_replaced;

        int iteration;           // Current iteration number

        prev_emission = new Distribution [num_variables][];
        for (int i = 0; i < num_variables; i++)
            prev_emission[i] = new Distribution [num_states];

        boolean is_done = false;

        /* Since likelihood can be estimated after performing an E-step, doing just that. */

        /* Allocating necessary structures */
        AllocateEMStructures(data, input_data);

        /* E-step */
        /* Need to perform forward-backward algorithm */
        EStep(data, input_data);

        iteration = 0;
        if (params.em_verbose)
            System.out.format("%d:\tll=%.12f\tlp=%.12f\tper-dimension-lp=%.12f\n",
                    iteration, ll, lp, lp_normalized);

        //  WriteToFile( stdout );
        //  WriteToFileBare( stdout );


        if (!isFinite(lp_normalized)) {
            if (params.em_verbose) {
                System.err.format("Log-posterior is infinite\n");
                num_failed++;
                is_done = true;
                //	WriteToFile( stderr );
            }
        }

        while (!is_done) { /* Main loop */
            iteration++;

            /* M-step */

            /* Storing the previous value of the log-likelihood */
            old_lp_normalized = lp_normalized;

            /* Calculating necessary summaries */
            CalculateSummaries(data);

            /* Updating emission probabilities */
            for (int i = 0; i < num_variables; i++)
                for (int j = 0; j < num_states; j++) {
                    prev_emission[i][j] = emission[i][j].copy();
                    emission[i][j].UpdateEmissionParameters(data, transition.uni_prob[i][j], norm_const[i][j]);
                }

            /* Updating transition layer parameters */
            prev_transition = transition.copy();
            transition.UpdateTransitionParameters(data, input_data, iteration);

            /* E-step */
            /* Need to perform forward-backward algorithm */
            EStep(data, input_data);

            if (params.em_verbose)
                System.out.format("%d:\tll=%.12f\tlp=%.12f\tper-dimension-lp=%.12f\n",
                        iteration, ll, lp, lp_normalized);

            if (!isFinite(lp_normalized)) { /* Log-posterior is infinite -- run failed */
                num_failed++;
                is_done = true;
            } /* Log-posterior is infinite -- run failed */ else if (lp_normalized - old_lp_normalized < precision) { /* Checking whether the end condition is reached */
                if (lp_normalized >= old_lp_normalized)
                    is_done = true;
                else { /* Figuring out what went wrong with the model */
                    if (params.em_verbose)
                        System.out.format("Log-posterior went down\n");

                    /* Analyzing why log-likelihood went down */
                    /* Most likely, one of the emissions changed to a worse set of parameters */
                    num_emissions_replaced = 0;
                    for (int i = 0; i < num_variables; i++)
                        for (int j = 0; j < num_states; j++) {
                            if (prev_emission[i][j].weighted_ll(data, transition.uni_prob[i][j]) >
                                    emission[i][j].weighted_ll(data, transition.uni_prob[i][j])) { /* Emission i got worse */
                                num_emissions_replaced++;
                                emission[i][j] = null;
                                emission[i][j] = prev_emission[i][j].copy();
                            } /* Emission i got worse */
                        }

                    if (num_emissions_replaced == 0) {
                        DeallocateEMStructures(data, input_data);
                        transition = null;
                        transition = prev_transition.copy();
                        AllocateEMStructures(data, input_data);
                    }

                    /* Redoing the E-step */
                    EStep(data, input_data);

                    if (params.em_verbose)
                        System.out.format("Corrected ll=%.12f\tlp=%.12f\tper-dimension lp=%.12f\n",
                                ll, lp, lp_normalized);

                    if (!isFinite(lp_normalized)) { /* Log-posterior is infinite -- run failed */
                        num_failed++;
                        is_done = true;
                    } /* Log-posterior is infinite -- run failed */ else if (lp_normalized - old_lp_normalized < precision) { /* Checking whether the end condition is reached */
                        if (lp_normalized >= old_lp_normalized)
                            is_done = true;
                        else {
                            if (params.em_verbose)
                                System.out.format("Corrected log-posterior still went down.  Reverting to previous emission parameter values.\n");
                            for (int i = 0; i < num_variables; i++)
                                for (int j = 0; j < num_states; j++) {
                                    emission[i][j] = null;
                                    emission[i][j] = prev_emission[i][j].copy();
                                }

                            DeallocateEMStructures(data, input_data);
                            transition = null;
                            transition = prev_transition.copy();
                            AllocateEMStructures(data, input_data);

                            /* Redoing the E-step */
                            EStep(data, input_data);
                            is_done = true;
                        }
                    } /* Checking whether the end condition is reached */
                } /* Figuring out what went wrong with the model */
            } /* Checking whether the end condition is reached */

            for (int i = 0; i < num_variables; i++)
                for (int j = 0; j < num_states; j++)
                    prev_emission[i][j] = null;

            //      System.out.format( "\n\n\n\n\n" );
            //      WriteToFileBare( stdout );
        } /* Main loop */

        /* Deallocating arrays */
        DeallocateEMStructures(data, input_data);

        prev_transition = null;

        for (int i = 0; i < num_variables; i++)
            prev_emission[i] = null;
        prev_emission = null;

    }

    void AllocateEMStructures(Data data, Data input) {
        /* Allocating structures needed for forward-backward algorithm */

        AllocateAlpha(data, input);
        transition.AllocateEMStructures(data, input);
    }

    void AllocateAlpha(Data data, Data input) {
        /* Allocating structures needed to calculate scaled alphas for forward-backward algorithm */

        /* Allocation block */

        /* Allocating array of P(R_nt|S_nt=ij) (or P(R_nt,X_nt|S_nt=ij) ) */
        for (int i = 0; i < num_variables; i++)
            for (int j = 0; j < num_states; j++) { /* For each state */
                log_pb[i][j] = new double[data.num_seqs][];
                for (int n = 0; n < data.num_seqs; n++)
                    log_pb[i][j][n] = new double[data.sequence[n].seq_length];
            } /* For each state */

        transition.AllocateForwardPassStructures(data, input);

    }

    void EStep(Data data, Data input) {

        lp = log_posterior(data, input);
        lp_normalized = lp / data.total_entries();

        for (int n = 0; n < data.num_seqs; n++)
            transition.CalculateBackwardUpdatesSequence(data.sequence[n], n, log_pb);

    }

    double log_likelihood_sequence(Sequence data, Sequence input, int n, int start_index) {
        double ll = 0.0;

        /*** Calculating needed values (forward) ***/

        /* Calculating emission probabilities */
        CalculateProbEmissionsSequence(data, n, start_index);

        /* Calculating P(S_nt|S_n,t-1, X_nt) */
        transition.CalculateTransitionValuesSequence(input, n, start_index);

        /* Need to perform forward pass -- calculating scaled alpha values */
        transition.CalculateForwardUpdatesSequence(data, n, start_index, log_pb);

        /* Calculating the log-likelihood using scaled alphas */
        for (int t = 0; t < data.seq_length; t++)
            ll -= transition.log_upd_scale[n][t];

        return (ll);
    }

    void CalculateProbEmissionsSequence(Sequence data, int n, int start_index) {

        /* Computing the log-probabilities of emissions log P(R_nt|S_nt=ij) */
        for (int t = start_index; t < data.seq_length; t++)
            if (t == 0)
                for (int i = 0; i < num_variables; i++)
                    for (int j = 0; j < num_states; j++)
                        log_pb[i][j][n][t] = emission[i][j].log_prob(data.entry[t], null);
            else
                for (int i = 0; i < num_variables; i++)
                    for (int j = 0; j < num_states; j++)
                        log_pb[i][j][n][t] = emission[i][j].log_prob(data.entry[t], data.entry[t - 1]);

    }


    void CalculateSummaries(Data data) {

        transition.CalculateSummaries(data, log_pb);

        /* Calculating aggregate weights */
        for (int i = 0; i < num_variables; i++)
            for (int j = 0; j < num_states; j++) {
                norm_const[i][j] = 0.0;
                for (int n = 0; n < data.num_seqs; n++)
                    for (int t = 0; t < data.sequence[n].seq_length; t++)
                        norm_const[i][j] += transition.uni_prob[i][j][n][t];
            }

    }


    void DeallocateEMStructures(Data data, Data input) {
        /* Deallocating structures needed for forward-backward algorithm */

        DeallocateAlpha(data, input);
        transition.DeallocateEMStructures(data, input);

    }

    void DeallocateAlpha(Data data, Data input_vars) {
        /* Deallocating arrays needed to calculate scaled alphas in forward-backward algorithm */

        /* log_pb */
        for (int i = 0; i < num_variables; i++)
            for (int j = 0; j < num_states; j++) { /* For each state */
                for (int n = 0; n < data.num_seqs; n++)
                    log_pb[i][j][n] = null;
                log_pb[i][j] = null;
            } /* For each state */

        transition.DeallocateForwardPassStructures(data, input_vars);
    }

}
