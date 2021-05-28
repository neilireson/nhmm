package uk.ac.shef.wit.nhmm;

import java.io.File;
import java.io.IOException;
import java.io.PrintStream;

import static java.lang.Math.sqrt;
import static uk.ac.shef.wit.nhmm.Constants.*;
import static uk.ac.shef.wit.nhmm.Data.is_missing;

public class NHMM {

    Parameters params;               // Parameters for the file
    HMM theta = null;                  // Parameters of the HMM
    Data output_data = null;           // Data for the output variables
    Data input_data = null;            // Data for the input variables
    Data extra_data = null;            // Extra data
    HMM[] passed_theta = null;          // Parameters from the model file

    File data_file = null;             // Data file
    File input_file = null;            // Input file
    File extra_file = null;            // Extra data file
    File model_file = null;            // File with values of parameters for the model
    File state_output_file = null;     // File to write states to

    PrintStream output_stream = null;           // Output file

    int num_failed;                  // Number of failed runs

    double cl_threshold;              // Chow-Liu mutual information threshold !!! !!!


    public int robust_first_state;

    double cg_epsilon;
    double maxent_epsilon;

    public static void main(String[] args) throws IOException {
        new NHMM(args);
    }

    public NHMM(String[] args) throws IOException {
        /* Variables */
        File param_file;                  // Parameter file

        Data current_output_data;         // Current output data set
        Data current_input_data;          // Current input data set

        StateData best_states;            // Most likely sequences of states according to the model and the data

        int num_xval_sets = 0;                // Number of cross-validation sets
        int num_seqs;                     // Number of sequences to be simulated

        double[] kl;                        // Entropy and the KL-divergence

        long start_time, end_time;

        if (MEASURE_TIME) {
            start_time = System.currentTimeMillis();
        }

        /* Processing inputs */
        if (args.length == 1) { /* Displaying help */
            /* Displaying the version number */
            System.out.format("Compiled from source with time stamp November 5, 2006.\n");

            /* Command line */
            System.out.format("Usage: mvnhmm <parameter file> [<seed>]\n");
            test();
        } /* Displaying help */ else { /* Processing parameters file */
            param_file = new File(args[1]);
            if (!param_file.isFile()) { /* File not found */
                System.err.format("Unable to open parameter file %s!  Aborting.\n", args[1]);
                System.exit(-1);
            } /* File not found */

            if (INPUT_VERBOSE) {
                System.out.format("Parameter file %s successfully opened.\n", args[1]);
            }

            /* Reading the parameters */
            params = new Parameters();
            params.Read(param_file);

            if (args.length <= 2)
                seed = 0;
            else
                seed = Integer.parseInt(args[2]);

        } /* Processing parameters file */

        /* Making sure parameters are ok */
        ProcessParameters();

        /* Determining the number of cross-validation sets */
        switch (params.xval_type) {
            case XVAL_NONE:
                num_xval_sets = 1;
                break;
            case XVAL_LEAVENOUT:
                switch (params.action) {
                    case ACTION_CODE_LEARN:
                    case ACTION_CODE_VITERBI:
                    case ACTION_CODE_LL:
                    case ACTION_CODE_LLTRAIN:
                    case ACTION_CODE_FILLING:
                    case ACTION_CODE_PREDICT:
                        num_xval_sets = output_data.num_seqs / params.number_out;
                        break;
                    case ACTION_CODE_SIM:
                        num_xval_sets = params.num_models;
                        break;
                    case ACTION_CODE_ANALYZE:
                        num_xval_sets = output_data.num_seqs / (params.num_simulations * params.number_out);
                        break;
                    case ACTION_CODE_DEBUG:
                        num_xval_sets = output_data.num_seqs / params.number_out;
                        break;
                    default:
                        ;
                        num_xval_sets = 1;
                }
                break;
            case ACTION_CODE_INIT:
            case ACTION_CODE_KL:
                break;
            default:
                num_xval_sets = 1;
        }

        /* Allocating data structures */
        current_input_data = null;
        current_output_data = null;
        switch (params.xval_type) {
            case XVAL_NONE:
                switch (params.action) {
                    case ACTION_CODE_LEARN:
                    case ACTION_CODE_VITERBI:
                    case ACTION_CODE_LL:
                    case ACTION_CODE_LLTRAIN:
                    case ACTION_CODE_FILLING:
                    case ACTION_CODE_PREDICT:
                        current_output_data = new Data(output_data.num_seqs, "data\0", false);
                        if (input_data!=null)
                            current_input_data = new Data(input_data.num_seqs, "input\0", false);
                        else
                            current_input_data = null;
                        break;
                    case ACTION_CODE_SIM:
                        if (input_data!= null)
                            current_input_data = new Data(input_data.num_seqs, "input\0", false);
                        else
                            current_input_data = null;
                        num_seqs = params.num_data_seqs;
                        break;
                    case ACTION_CODE_ANALYZE:
                        current_output_data = new Data(output_data.num_seqs, "data\0", false);
                        if (input_data!= null)
                            current_input_data = new Data(input_data.num_seqs, "input\0", false);
                        else
                            current_input_data = null;
                        break;
                    case ACTION_CODE_DEBUG:
                        current_output_data = new Data(output_data.num_seqs, "data\0", false);
                        if (input_data!= null)
                            current_input_data = new Data(input_data.num_seqs, "input\0", false);
                        else
                            current_input_data = null;
                        break;
                    case ACTION_CODE_INIT:
                    case ACTION_CODE_KL:
                        break;
                    default:
                        ;
                }
                break;
            case XVAL_LEAVENOUT:
                switch (params.action) {
                    case ACTION_CODE_LEARN:
                    case ACTION_CODE_LLTRAIN:
                        current_output_data = new Data(output_data.num_seqs - params.number_out, "data\0", false);
                        if (input_data!=null)
                            current_input_data = new Data(input_data.num_seqs - params.number_out, "input\0", false);
                        else
                            current_input_data = null;
                        break;
                    case ACTION_CODE_VITERBI:
                    case ACTION_CODE_LL:
                    case ACTION_CODE_FILLING:
                    case ACTION_CODE_PREDICT:
                        current_output_data = new Data(params.number_out, "data\0", false);
                        if (input_data!=null)
                            current_input_data = new Data(params.number_out, "input\0", false);
                        else
                            current_input_data = null;
                        break;
                    case ACTION_CODE_SIM:
                        if (input_data!=null)
                            current_input_data = new Data(params.number_out, "input\0", false);
                        else
                            current_input_data = null;
                        num_seqs = params.number_out;
                        break;
                    case ACTION_CODE_ANALYZE:
                        current_output_data = new Data(params.num_simulations * params.number_out, "data\0", false);
                        if (input_data!=null)
                            current_input_data = new Data(params.num_simulations * params.number_out, "input\0", false);
                        else current_input_data = null;
                        break;
                    case ACTION_CODE_DEBUG:
                        current_output_data = new Data(params.number_out, "data\0", false);
                        if (input_data!=null)
                            current_input_data = new Data(params.number_out, "input\0", false);
                        else
                            current_input_data = null;
                        break;
                    case ACTION_CODE_INIT:
                    case ACTION_CODE_KL:
                        break;
                    default:
                        ;
                }
                break;
            default:
                ;
        }


        /* Looping over cross-validation sets */
        for (int xval_index = 0; xval_index < num_xval_sets; xval_index++) {
            /* Initializing sets */
            switch (params.xval_type) {
                case XVAL_NONE:
                    switch (params.action) {
                        case ACTION_CODE_LEARN:
                            for (int n = 0; n < output_data.num_seqs; n++) {
                                current_output_data.sequence[n] = output_data.sequence[n];
                                if (input_data!=null)
                                    current_input_data.sequence[n] = input_data.sequence[n];
                            }
                            break;
                        case ACTION_CODE_VITERBI:
                        case ACTION_CODE_LL:
                        case ACTION_CODE_LLTRAIN:
                        case ACTION_CODE_FILLING:
                        case ACTION_CODE_PREDICT:
                            for (int n = 0; n < output_data.num_seqs; n++) {
                                current_output_data.sequence[n] = output_data.sequence[n];
                                if (input_data!=null)
                                    current_input_data.sequence[n] = input_data.sequence[n];
                            }
                            break;
                        case ACTION_CODE_SIM:
                            if (input_data!=null)
                                for (int n = 0; n < input_data.num_seqs; n++)
                                    current_input_data.sequence[n] = input_data.sequence[n];
                            break;
                        case ACTION_CODE_ANALYZE:
                            for (int n = 0; n < output_data.num_seqs; n++) {
                                current_output_data.sequence[n] = output_data.sequence[n];
                                if (input_data!=null)
                                    current_input_data.sequence[n] = input_data.sequence[n];
                            }
                            break;
                        case ACTION_CODE_DEBUG:
                            for (int n = 0; n < output_data.num_seqs; n++) {
                                current_output_data.sequence[n] = output_data.sequence[n];
                                if (input_data!=null)
                                    current_input_data.sequence[n] = input_data.sequence[n];
                            }
                            break;
                        case ACTION_CODE_INIT:
                        case ACTION_CODE_KL:
                            break;
                        default:
                            ;
                    }
                    break;
                case XVAL_LEAVENOUT:
                    switch (params.action) {
                        case ACTION_CODE_LEARN:
                            output_stream.format("%c Best model for set without years %d-%d\n",
                                    COMMENT_SYMBOL, xval_index * params.number_out + 1, (xval_index + 1) * params.number_out);

                            if (VERBOSE_BRIEF) {
                                System.out.format("Set #%d\n", xval_index + 1);
                            }
                        case ACTION_CODE_LLTRAIN:
                            int current_n = 0;

                            for (int n = 0; n < output_data.num_seqs; n++)
                                if (n < xval_index * params.number_out || n >= (xval_index + 1) * params.number_out) {
                                    current_output_data.sequence[current_n] = output_data.sequence[n];
                                    if (input_data!=null)
                                        current_input_data.sequence[current_n] = input_data.sequence[n];

                                    current_n++;
                                }
                            break;
                        case ACTION_CODE_VITERBI:
                        case ACTION_CODE_LL:
                        case ACTION_CODE_FILLING:
                        case ACTION_CODE_PREDICT:
                            for (int n = 0; n < params.number_out; n++) {
                                current_output_data.sequence[n] = output_data.sequence[xval_index * params.number_out + n];
                                if (input_data!=null)
                                    current_input_data.sequence[n] = input_data.sequence[xval_index * params.number_out + n];
                            }
                            break;
                        case ACTION_CODE_SIM:
                            for (int n = 0; n < params.number_out; n++)
                                if (input_data!=null)
                                    current_input_data.sequence[n] = input_data.sequence[xval_index * params.number_out + n];
                            break;
                        case ACTION_CODE_ANALYZE:
                            for (int n = 0; n < params.num_simulations * params.number_out; n++) {
                                current_output_data.sequence[n] =
                                        output_data.sequence[xval_index * params.num_simulations * params.number_out + n];
                                if (input_data!=null)
                                    current_input_data.sequence[n] =
                                            input_data.sequence[xval_index * params.num_simulations * params.number_out + n];
                            }
                            break;
                        case ACTION_CODE_DEBUG:
                            for (int n = 0; n < params.number_out; n++) {
                                current_output_data.sequence[n] = output_data.sequence[xval_index * params.number_out + n];
                                if (input_data!=null)
                                    current_input_data.sequence[n] = input_data.sequence[xval_index * params.number_out + n];
                            }
                            break;
                        case ACTION_CODE_INIT:
                        case ACTION_CODE_KL:
                            break;
                        default:
                            ;
                    }
                    break;
                default:
                    ;
            }

            if (XVAL_VERBOSE) {
                System.out.format( "Running cross-validation run %d.\n", xval_index + 1);
                if (current_output_data!=null)
                    System.out.format( "Output data consists of %d sequences of length %d each.\n",
                            current_output_data.num_seqs, current_output_data.sequence[0].seq_length);
                else
                    System.out.format( "No output data created\n");

                if (current_input_data!=null)
                    System.out.format( "Input data consists of %d sequences of length %d each.\n",
                            current_input_data.num_seqs, current_input_data.sequence[0].seq_length);
                else
                    System.out.format( "No input data created\n");

            }

            /* Running model */
            switch (params.action) {
                case ACTION_CODE_LEARN:
                    RunLearnParameters(current_output_data,
                            current_input_data,
                            theta,
                            passed_theta,
                            params,
                            output_stream);
                    break;
                case ACTION_CODE_VITERBI:
                    /* Calculating the most likely sequences of states */
                    best_states = passed_theta[xval_index].viterbi(current_output_data, current_input_data, 1);
                    best_states.WriteToFile(output_stream);

                    /* Deallocating the sequences of states */
                    best_states = null;
                    break;
                case ACTION_CODE_LL:
                case ACTION_CODE_LLTRAIN:
                    RunLogLikelihoodData(current_output_data, current_input_data, passed_theta[xval_index], output_file);
                    break;
                case ACTION_CODE_SIM:
                    for (int i = 0; i < params.num_simulations; i++)
                        RunSimulateData(current_input_data,
                                passed_theta[xval_index],
                                params,
                                state_output_file,
                                output_file,
                                num_seqs);
                    break;
                case ACTION_CODE_ANALYZE:
                    if (params.analysis_type != ANALYSIS_COMP)
                        RunEvaluate(current_output_data, params.analysis_type, output_stream);
                    else
                        output_stream.format("%d\n", CompareDataSets(current_output_data, current_input_data));
                    break;
                case ACTION_CODE_FILLING:
                    RunHoleData(current_output_data, current_input_data, passed_theta[xval_index], output_file, params.poking_type);
                    break;
                case ACTION_CODE_PREDICT:
                    RunPredictionData(current_output_data, current_input_data, passed_theta[xval_index], params.lookahead, output_file);
                    break;
                case ACTION_CODE_INIT:
                    RunGenerateParameters(theta, params.num_models, output_file);
                    break;
                case ACTION_CODE_KL:
                    kl = passed_theta[0].KL(passed_theta[1]);
                    output_stream.format("%.12f\t%.12f\n", kl[0], kl[1]);
                    kl = null;
                    break;
                case ACTION_CODE_DEBUG:
                    TestDebug(current_output_data, current_input_data, passed_theta[xval_index], output_file);
                    break;
                default:
                    System.err.format("Action is unidentified or unspecified.  Aborting.\n");
                    return (-1);
            }
        }

        /* Deallocating data structures */
        switch (params.action) {
            case ACTION_CODE_LEARN:
            case ACTION_CODE_VITERBI:
            case ACTION_CODE_LL:
            case ACTION_CODE_LLTRAIN:
            case ACTION_CODE_FILLING:
            case ACTION_CODE_PREDICT:
                current_output_data = null;
                if (current_input_data!=null)
                    current_input_data = null;
                break;
            case ACTION_CODE_SIM:
                if (current_input_data!=null)
                    current_input_data = null;
                break;
            case ACTION_CODE_ANALYZE:
                current_output_data = null;
                if (current_input_data !=null)
                    current_input_data = null;
                break;
            case ACTION_CODE_DEBUG:
                current_output_data = null;
                if (current_input_data!=null)
                    current_input_data = null;
                break;
            /* ACTION_CODE_INIT */
            default:
                ;
        }

        /* Deallocating passed models */
        if (passed_theta!=null) {
            for (int i = 0; i < params.num_models; i++)
                passed_theta[i] = null;
            passed_theta = null;
        }

        /* Deallocating the model */
        if (theta!=null)
            theta = null;

        /* Deallocating the structures with data */
        if (output_data!=null)
            output_data = null;

        if (input_data!=null)
            input_data = null;

        if (extra_data!=null)
            extra_data = null;

        /* Deallocating the structure with parameters */
        params = null;

        if (MEASURE_TIME) {
            end_time = System.currentTimeMillis();
            System.out.format("Elapsed CPU time: %f\n", (double) (end_time - start_time) / (double) 1000);
        }

        return (0);
    }

    void RunEvaluate(Data output_data, int option, PrintStream out) {
        double[]mean;
        double[][][]cov;
        double[][]pers;
        int[][]dist;
        double[][]MI;
        double[][]lo;

        int dim;

        /* !!! !!! */
        dim = output_data.sequence[0].entry[0].ddim;

        switch (option) {
            case ANALYSIS_MEAN:
                /* Calculating mean value for each component */
                dim = output_data.sequence[0].entry[0].ddim + output_data.sequence[0].entry[0].rdim;
                mean = output_data.mean();
                for (int i = 0; i < dim; i++)
                    out.format("%.8f\t", mean[i]);
                out.format("\n");

                mean = null;

                break;
            case ANALYSIS_CORR:
                /* Calculating correlation of the components */

                /* First, calculating the covariance matrices */
                cov = output_data.covariance();

                /* corr(i,j)=cov(i,j)/sqrt(cov(i,i)*cov(j,j)) */
                if (cov[0] != null) {
                    dim = output_data.sequence[0].entry[0].ddim;
                    for (int i = 0; i < dim; i++) {
                        for (int j = 0; j < dim; j++)
                            out.format("%.8f\t", cov[0][i][j] / (sqrt(cov[0][i][i] * cov[0][j][j])));
                        out.format("\n");
                    }

                    for (int i = 0; i < dim; i++)
                        cov[0][i] = null;
                    cov[0] = null;
                }
                if (cov[1] != null) {
                    dim = output_data.sequence[0].entry[0].rdim;
                    for (int i = 0; i < dim; i++) {
                        for (int j = 0; j < dim; j++)
                            out.format("%.8f\t", cov[1][i][j] / (sqrt(cov[1][i][i] * cov[1][j][j])));
                        out.format("\n");
                    }

                    for (int i = 0; i < dim; i++)
                        cov[1][i] = null;
                    cov[1] = null;
                }

                cov = null;

                break;
            case ANALYSIS_PERS:
                /* Calculating probabilities of persisting */
                pers = output_data.persistence();

                for (int i = 0; i < dim; i++) {
                    for (int j = 0; j < 2 /* !!! !!! */; j++)
                        out.format("%.8f\t", pers[i][j]);
                    out.format("\n");
                }

                for (int i = 0; i < dim; i++)
                    pers[i] = null;
                pers = null;

                break;
            case ANALYSIS_DRY:
                /* Calculating the distribution of the dry spell length */
                dist = output_data.spell(0);

                for (int i = 0; i < dim; i++) {
                    for (int j = 0; j < output_data.sequence[0].seq_length /* !!! !!! */;
                         j++ )
                        out.format("%d\t", dist[i][j]);
                    out.format("\n");
                }

                for (int i = 0; i < dim; i++)
                    dist[i] = null;
                dist = null;

                break;
            case ANALYSIS_WET:
                /* Calculating the distribution of the wet spell length */
                dist = output_data.spell(1);

                for (int i = 0; i < dim; i++) {
                    for (int j = 0; j < output_data.sequence[0].seq_length /* !!! !!! */;
                         j++ )
                        out.format("%d\t", dist[i][j]);
                    out.format("\n");
                }

                for (int i = 0; i < dim; i++)
                    dist[i] = null;
                dist = null;

                break;
            case ANALYSIS_MI:
                /* Calculating mutual information of the components */

                /* Calculating mutual information */
                MI = output_data.mutual_information();

                for (int i = 0; i < dim; i++) {
                    for (int j = 0; j < dim; j++)
                        out.format("%.8f\t", MI[i][j]);
                    out.format("\n");
                }

                for (int i = 0; i < dim; i++)
                    MI[i] = null;
                MI = null;

                break;
            case ANALYSIS_LO:
                /* Calculating log-odds ratio for the pairs of components */
                lo = output_data.log_odds_ratio();

                for (int i = 0; i < dim; i++) {
                    for (int j = 0; j < dim; j++)
                        out.format("%.8f\t", lo[i][j]);
                    out.format("\n");
                }

                for (int i = 0; i < dim; i++)
                    lo[i] = null;
                lo = null;

                break;
            default:
                ;
        }
    }

    void RunLearnParameters(Data output,
                            Data input,
                            HMM theta,
                            HMM passed_theta[],
                            Parameters params,
                            PrintStream out) {

        /* Enveloping routines for em algorithm */

        HMM final_theta;  // Final parameters of the model
        HMM temp_theta;   // Current model

        int num_failed; // Number of failed runs
        int prev_num_failed;   // Temporary variable
        int noninit_index;     // Index of first non-initialized run


        /* Temporary variables */
        int sum;

        /* Initializing */
        final_theta = null;
        num_failed = 0;

        /* !!! Determining whether to run regular EM or EM-MCMC !!! */
        sum = 0;
        for (int i = 0; i < output. num_seqs; i++)
            sum += output. sequence[i].num_missing_discrete_entries;

        /* Checking whether initial conditions have been passed */
        if (passed_theta != null) {
            if (params.num_models < params.num_restarts)
                noninit_index = params.num_models;
            else
                noninit_index = params.num_restarts;

            for (int i = 0; i < noninit_index; i++) {
                if (params.em_verbose)
                    System.out.format("Starting restart #%d\n", i + 1);

                prev_num_failed = num_failed;

                /* Allocating a new set of parameters */
                temp_theta = passed_theta[i].copy();

                /* Learning the parameters */
                temp_theta.EM(output, input, params.em_precision);

                /* Updating the best model */
                if (num_failed == prev_num_failed) { /* Run finished ok */
                    if (final_theta==null)
                        final_theta = temp_theta;
                    else if (temp_theta.ll < final_theta.ll) {
                        final_theta = temp_theta;
                    } else
                        temp_theta = null;
                } /* Run finished ok */
            }
        } else
            noninit_index = 0;

        for (int i = noninit_index; i < params.num_restarts; i++) {
            /* Allocating a new set of parameters */
            temp_theta = theta.copy();

            if (params.em_verbose)
                System.out.format("Starting restart #%d\n", i + 1);

            prev_num_failed = num_failed;

            /* Initializing the parameters */
            switch (params.initialization_type) {
                case INIT_RANDOM:
                    temp_theta.Initialize(output, input);
                    break;
                case INIT_EM:
                    temp_theta.InitializeByEM(output, input);
                    break;
                default:
                    ;
            }

            /* Learning the parameters */
            temp_theta.EM(output, input, params.em_precision);

            if (num_failed == prev_num_failed)
                /* Run finished ok */
                if (final_theta == null)
                    final_theta = temp_theta;
                else if (temp_theta.ll > final_theta.ll) { /* Updating the best model */
                    final_theta = temp_theta;
                } /* Updating the best model */ else
                    temp_theta = null;
        }

        if (num_failed < params.num_restarts) { /* At least one run is successful */
            if (params.bare_display)
                final_theta.WriteToFileBare(out);
            else {
                /* Calculating BIC score */
                final_theta.bic = final_theta.BIC(output.num_points());

                final_theta.PostProcess();

                final_theta.WriteToFile(out);

                /* Displaying the statistics about the run */
                System.out.format("%c Failure rate: %d out of %d.\n",
                        COMMENT_SYMBOL, num_failed, params.num_restarts);
            }
        } /* At least one run is successful */ else { /* All runs failed */
            if (params.bare_display)
                System.out.format("%c All runs are unsuccessful\n",
                        COMMENT_SYMBOL);
        } /* All runs failed */

        /* Deallocating the model */
        if (final_theta!=null)
            final_theta = null;

    }

    void ProcessParameters(void) throws IOException {
  /* Making sure that passed parameters are proper, that all necessary
     files are passed and contain needed information */

        /* Index variables */
        int i, j;

        /* Creating templates for data points in data and input files */
        params.output_datum = new DataPoint(params.num_ddata_components,
                params.num_rdata_components);
        params.input_datum = new DataPoint(params.num_dinput_components,
                params.num_rinput_components);
        params.extra_datum = new DataPoint(params.num_dextra_components,
                params.num_rextra_components);

        /* !!! Hack !!! */
        cg_epsilon = params.cg_epsilon;
        maxent_epsilon = params.maxent_epsilon;

        /* Reading in the output file if passed */
        if (params.output_data_filename) {
            data_file = fopen(params.output_data_filename, "r");
            if (!data_file) {
                System.err.format("Unable to open data file %s for reading. Aborting\n",
                        params.output_data_filename);
                System.exit(-1);
            } else {
                if (INPUT_VERBOSE) {
                    System.out.format("Data file %s successfully opened for reading.\n",
                            params.output_data_filename);
                }
                output_data = new Data(params.num_data_seqs, "data\0", 1);
                if (params.length_data_seq)
                    output_data.ReadDataDistinct(data_file, 0, params.num_data_seqs,
                            params.length_data_seq, params.output_datum);
                else
                    output_data.ReadData(data_file, 0, params.num_data_seqs,
                            params.length_data_seqs, params.output_datum);

                /* Determining the number of missing entries */
                for (int i = 0; i < output_data.num_seqs; i++)
                    output_data.sequence[i].CatalogueMissingEntries();
            }
        }

        /* Reading in the input file if passed */
        if (params.input_filename != null) {
            input_file = new File(params.input_filename);
            if (!input_file.isFile()) {
                System.err.format("Unable to open input file %s for reading.\n",
                        params.input_filename);
            } else {
                if (INPUT_VERBOSE) {
                    System.out.format("Input variable file %s successfully opened for reading.\n",
                            params.input_filename);
                }
                input_data = new Data(params.num_input_seqs, "input\0", 1);
                if (params.length_input_seq != null)
                    input_data.ReadDataDistinct(input_file, 0, params.num_input_seqs,
                            params.length_input_seq, params.input_datum);
                else
                    input_data.ReadData(input_file, 0, params.num_input_seqs,
                            params.length_input_seqs, params.input_datum);
            }
        }

        /* Reading in the extra data file if passed */
        if (params.extra_data_filename) {
            extra_file = fopen(params.input_filename, "r");
            if (!extra_file) {
                System.err.format("Unable to open extra data file %s for reading.\n",
                        params.extra_data_filename);
            } else {
                if (INPUT_VERBOSE) {
                    System.out.format("Extra data file %s successfully opened for reading.\n",
                            params.extra_data_filename);
                }
                extra_data = new Data(params.num_extra_seqs, "extra\0", 1);
                extra_data.ReadData(extra_file, 0, params.num_extra_seqs,
                        params.length_extra_seqs, params.extra_datum);

                /* Closing the input file */
                fclose(extra_file);
            }
        }

        if (params.input_dim > 0 && !input_data) {
            System.err.format("Need input filename to work with input distribution.  Setting number of input components to 0.\n");
            params.input_dim = 0;
        }

        /* Output file */
        if (params.output_filename) { /* Output file listed */
            output_file = fopen(params.output_filename, "w");
            if (!output_file) {
                System.err.format("Unable to open file %s for writing.  Outputting to screen.\n",
                        params.output_filename);
                output_file = stdout;
            }
        } /* Output file listed */ else { /* No output file provided -- outputting results to the screen */
            output_file = stdout;
        } /* No output file provided -- outputting results to the screen */

        if (params.action != ACTION_CODE_ANALYZE) { /* HMM model is defined */
            /* Initializing the model structure */
            if (!params.emission) {
                System.err.format("Need to have proper emission distribution defined!  Aborting.\n");
                System.exit(-1);
            }

            /* Model instantiation */
            theta = new HMM(params);

            if (params.transition != null)
                /* Transition distribution passed -- updating the number of variables and states */
                theta.transition = params.transition.copy();
            else
                /* Creating transition probability distribution */
                switch (params.model_type) {
                    case MODEL_TYPE_HMM:
                        if (params.robust_first_state != 0)
                            theta.transition = new Distribution(DIST_CONDBERNOULLIG, params.num_states, 1);
                        else
                            theta.transition = new Distribution(DIST_CONDBERNOULLI, params.num_states, 1);
                        break;
                    case MODEL_TYPE_MIX:
                        theta.transition = new Distribution(DIST_BERNOULLI, params.num_states, 1);
                        break;
                    case MODEL_TYPE_NHMM:
                        theta.transition = new Distribution(DIST_TRANSLOGISTIC, params.num_states, params.input_dim);
                        break;
                    case MODEL_TYPE_NMIX:
                        theta.transition = new Distribution(DIST_LOGISTIC, params.num_states, params.input_dim);
                        break;
                    default:
                        ;
                }

            /* Emission probability distributions */
            for (int i = 0; i < theta.num_variables; i++)
                for (int j = 0; j < theta.num_states; j++)
                    theta.emission[i][j] = params.emission[i].copy();

            /* Reading in the model */
            if (params.model_filename && params.action != ACTION_CODE_INIT) {
                model_file = fopen(params.model_filename, "r");

                /* Correcting the number of models in the file */
                if (model_file) { /* Reading in the models */
                    switch (params.xval_type) {
                        case XVAL_NONE:
                            switch (params.action) {
                                case ACTION_CODE_LEARN:
                                    /* Number of models is as specified */
                                    break;
                                case ACTION_CODE_VITERBI:
                                case ACTION_CODE_LL:
                                case ACTION_CODE_LLTRAIN:
                                case ACTION_CODE_SIM:
                                case ACTION_CODE_FILLING:
                                case ACTION_CODE_PREDICT:
                                    /* Only one model in the file */
                                    params.num_models = 1;
                                    break;
                                case ACTION_CODE_KL:
                                    /* KL divergence -- expecting exactly two models */
                                    params.num_models = 2;
                                    break;
                                case ACTION_CODE_DEBUG:
                                    /* Only one model in the file */
                                    params.num_models = 1;
                                    break;
                                default:
                                    ;
                            }
                            break;
                        case XVAL_LEAVENOUT:
                            /* Determining whether the number of examples to be left out is proper */
                            if (input_data!=null) {
                                if (params.number_out > (int) ceil((double) params.num_input_seqs / 2.0)) {
                                    System.err.format("Number of examples to leave out is too large.  Setting it to 1.\n");
                                    params.number_out = 1;
                                }
                            } else if (params.number_out > (int) ceil((double) params.num_data_seqs / 2.0)) {
                                System.err.format("Number of examples to leave out is too large.  Setting it to 1.\n");
                                params.number_out = 1;
                            }

                            /* Number of models is specified by the number of cross-validated sets */
                            switch (params.action) {
                                case ACTION_CODE_VITERBI:
                                case ACTION_CODE_LL:
                                case ACTION_CODE_LLTRAIN:
                                case ACTION_CODE_SIM:
                                case ACTION_CODE_FILLING:
                                case ACTION_CODE_PREDICT:
                                    if (input_data!=null)
                                        params.num_models = params.num_input_seqs / params.number_out;
                                    else
                                        params.num_models = params.num_data_seqs / params.number_out;
                                    break;
                                case ACTION_CODE_DEBUG:
                                    if (input_data!=null)
                                        params.num_models = params.num_input_seqs / params.number_out;
                                    else
                                        params.num_models = params.num_data_seqs / params.number_out;
                                    break;
                                default:
                                    ;
                            }
                            break;
                        default:
                            ;
                    }

                    /* Reading in the model */
                    passed_theta = new HMM *[params.num_models];
                    for (int i = 0; i < params.num_models; i++) {
                        passed_theta[i] = theta.copy();
                        passed_theta[i].ReadParameters(model_file);
                    }

                    /* Closing the file */
                    fclose(model_file);
                } /* Reading in the models */ else {
                    System.err.format("Unable to open model file %s for reading.  Ignoring.\n",
                            params.model_filename);
                }
            }
        } /* HMM model is defined */

        /* Determining whether all information is passed for specific action */
        switch (params.action) {
            case ACTION_CODE_LEARN:
                if (!output_data) { /* No data provided */
                    System.err.format("No data sequences provided. Aborting.\n");
                    System.exit(-1);
                } /* No data provided */

                if (input_data!=null)
                    if (params.num_data_seqs != params.num_input_seqs ||
                            params.length_data_seqs != params.length_input_seqs) {
                        /* !!! To be changed later !!! */
                        System.err.format("The dimensions of input and output data are not the same. Aborting.\n");
                        System.exit(-1);
                    }

                break;
            case ACTION_CODE_VITERBI:
                if (!output_data) { /* No data provided */
                    System.err.format("No data sequences provided. Aborting.\n");
                    System.exit(-1);
                } /* No data provided */

                if (input_data!=null) {
                    if (params.num_data_seqs != params.num_input_seqs ||
                            params.length_data_seqs != params.length_input_seqs) {
                        /* !!! To be changed later !!! */
                        System.err.format("The dimensions of input and output data are not the same. Aborting.\n");
                        System.exit(-1);
                    }
                }

                if (!passed_theta) {
                    System.err.format("Action 'viterbi' requires models to be passed to it. Aborting.\n");
                    System.exit(-1);
                }
                break;
            case ACTION_CODE_LL:
            case ACTION_CODE_LLTRAIN:
                if (!output_data) { /* No data provided */
                    System.err.format("No data sequences provided. Aborting.\n");
                    System.exit(-1);
                } /* No data provided */

                if (input_data!=null) {
                    if (params.num_data_seqs != params.num_input_seqs ||
                            params.length_data_seqs != params.length_input_seqs) {
                        /* !!! To be changed later !!! */
                        System.err.format("The dimensions of input and output data are not the same. Aborting.\n");
                        System.exit(-1);
                    }
                }

                if (!passed_theta) {
                    System.err.format("Action 'll' requires models to be passed to it. Aborting.\n");
                    System.exit(-1);
                }
                break;
            case ACTION_CODE_SIM:
                if (!passed_theta) {
                    System.err.format("Action 'simulation' requires models to be passed to it. Aborting.\n");
                    System.exit(-1);
                }

                if (params.input_dim > 0) {
                    if (params.num_data_seqs != params.num_input_seqs) {
                        System.err.format("Changing the number of data sequences per simulated sets to %d.\n",
                                params.num_input_seqs);
                        params.num_data_seqs = params.num_input_seqs;
                    }

                    if (params.length_data_seqs != params.length_input_seqs) {
                        System.err.format("Changing the length of data sequences of simulated sets to %d.\n",
                                params.length_input_seqs);
                        params.length_data_seqs = params.length_input_seqs;
                    }
                }

                /* Checking whether to open a file for the hidden state sequences */
                if (params.state_filename) {
                    state_output_file = fopen(params.state_filename, "w");
                    if (!state_output_file) {
                        System.err.format("Unable to open file %s for writing. Not displaying the states\n",
                                params.state_filename);
                    }
                }

                /* No output data is needed */
                if (output_data) {
                    output_data = null;
                    output_data = null;
                }
                break;
            case ACTION_CODE_FILLING:
                if (!output_data) { /* No data provided */
                    System.err.format("No data sequences provided. Aborting.\n");
                    System.exit(-1);
                } /* No data provided */

                if (input_data!=null) {
                    if (params.num_data_seqs != params.num_input_seqs ||
                            params.length_data_seqs != params.length_input_seqs) {
                        /* !!! To be changed later !!! */
                        System.err.format("The dimensions of input and output data are not the same. Aborting.\n");
                        System.exit(-1);
                    }
                }

                if (!passed_theta) {
                    System.err.format("Action 'poking' requires models to be passed to it. Aborting.\n");
                    System.exit(-1);
                }
                break;
            case ACTION_CODE_PREDICT:
                if (!output_data) { /* No data provided */
                    System.err.format("No data sequences provided. Aborting.\n");
                    System.exit(-1);
                } /* No data provided */

                if (input_data!=null) {
                    if (params.num_data_seqs != params.num_input_seqs ||
                            params.length_data_seqs != params.length_input_seqs) {
                        /* !!! To be changed later !!! */
                        System.err.format("The dimensions of input and output data are not the same. Aborting.\n");
                        System.exit(-1);
                    }
                }

                if (!passed_theta) {
                    System.err.format("Action 'prediction' requires models to be passed to it. Aborting.\n");
                    System.exit(-1);
                }
                break;
            case ACTION_CODE_DEBUG:
                if (!output_data) { /* No data provided */
                    System.err.format("No data sequences provided. Aborting.\n");
                    System.exit(-1);
                } /* No data provided */

                if (input_data!=null) {
                    if (params.num_data_seqs != params.num_input_seqs ||
                            params.length_data_seqs != params.length_input_seqs) {
                        /* !!! To be changed later !!! */
                        System.err.format("The dimensions of input and output data are not the same. Aborting.\n");
                        System.exit(-1);
                    }
                }

                if (!passed_theta) {
                    System.err.format("Action 'debug' requires models to be passed to it. Aborting.\n");
                    System.exit(-1);
                }
                break;
            case ACTION_CODE_INIT:
            case ACTION_CODE_KL:
                break;
            default:
                ;
        }

        if (FINAL_CHECK) {
            public int em_verbose;

            System.out.format("Number of states: %d\n", params.num_states);

            System.out.format("Type of model: ");
            switch (params.model_type) {
                case MODEL_TYPE_HMM:
                    System.out.format("HMM\n");
                    break;
                case MODEL_TYPE_NHMM:
                    System.out.format("NHMM\n");
                    break;
                case MODEL_TYPE_MIX:
                    System.out.format("mixture\n");
                    break;
                case MODEL_TYPE_NMIX:
                    System.out.format("non-homogeneous mixture\n");
                    break;
                default:
                    System.out.format("unknown\n");
            }

            System.out.format("Action type: ");
            switch (params.action) {
                case ACTION_CODE_LEARN:
                    System.out.format("parameter estimation\n");
                    break;
                case ACTION_CODE_VITERBI:
                    System.out.format("best sequence calculation\n");
                    break;
                case ACTION_CODE_LL:
                case ACTION_CODE_LLTRAIN:
                    System.out.format("log-likelihood calculation\n");
                    break;
                case ACTION_CODE_SIM:
                    System.out.format("simulation\n");
                    break;
                case ACTION_CODE_ANALYZE:
                    System.out.format("data analysis\n");
                    break;
                case ACTION_CODE_FILLING:
                    System.out.format("data analysis with hole filling\n");
                    break;
                case ACTION_CODE_PREDICT:
                    System.out.format("prediction\n");
                    break;
                case ACTION_CODE_DEBUG:
                    System.out.format("debugging\n");
                    break;
                default:
                    System.out.format("unknown\n");
            }

            if (params.emission)
                System.out.format("Emission distribution defined\n");
            if (params.input_dist)
                System.out.format("Input distribution defined\n");

            if (output_data)
                System.out.format("Output data filename: %s\n", params.output_data_filename);
            else
                System.out.format("No output data provided\n");

            System.out.format("Number of sequences: %d\n", params.num_data_seqs);
            System.out.format("Length of each sequence: %d\n", params.length_data_seqs);
            System.out.format("Number of discrete-valued components: %d\n", params.num_ddata_components);
            System.out.format("Number of real-valued components: %d\n", params.num_rdata_components);

            if (input_data!=null) {
                System.out.format("Input data filename: %s\n", params.input_filename);
                System.out.format("Number of sequences: %d\n", params.num_input_seqs);
                System.out.format("Length of each sequence: %d\n", params.length_input_seqs);
                System.out.format("Number of discrete-valued components: %d\n", params.num_dinput_components);
                System.out.format("Number of real-valued components: %d\n", params.num_rinput_components);
            } else
                System.out.format("No input data provided\n");

            System.out.format("Cross-validation type: ");
            switch (params.xval_type) {
                case XVAL_NONE:
                    System.out.format("none\n");
                    break;
                case XVAL_LEAVENOUT:
                    System.out.format("leave-n-out\n");
                    System.out.format("Number of examples to leave out: %d\n", params.number_out);
                    break;
                default:
                    System.out.format("unknown\n");
            }

            if (passed_theta) {
                System.out.format("Model filename: %s\n", params.model_filename);
                System.out.format("Number of models: %d\n", params.num_models);
            } else
                System.out.format("No models passed\n");

            if (output_file != stdout)
                System.out.format("Output file: %s\n", params.output_filename);
            else
                System.out.format("Outputting to the screen\n");

            if (state_output_file)
                System.out.format("Output file for the simulated states: %s\n", params.state_filename);

            System.out.format("Number of random restarts: %d\n", params.num_restarts);
            System.out.format("EM threshold constant: %.12f\n", params.em_precision);
            if (em_verbose)
                System.out.format("Outputting intermediate EM messages to the screen\n");
            else
                System.out.format("Not outputting intermediate EM messages to the screen\n");
            System.out.format("Learning initialization type: ");
            switch (params.initialization_type) {
                case INIT_RANDOM:
                    System.out.format("random\n");
                    break;
                case INIT_EM:
                    System.out.format("EM without inputs\n");
                    break;
                default:
                    System.out.format("unknown\n");
            }

            if (params.robust_first_state)
                System.out.format("Calculating first state probabilities from all observations\n");
            else
                System.out.format("Calculating first state probabilities from first observations only\n");

            System.out.format("Number of simulations: %d\n", params.num_simulations);
            if (params.state_filename)
                System.out.format("Printing simulated hidden state sequences in file %s\n", params.state_filename);
            else
                System.out.format("No simulated state files\n");

            System.out.format("Analysis type: ");
            switch (params.analysis_type) {
                case ANALYSIS_MEAN:
                    System.out.format("mean\n");
                    break;
                case ANALYSIS_CORR:
                    System.out.format("correlation\n");
                    break;
                case ANALYSIS_PERS:
                    System.out.format("persistence\n");
                    break;
                case ANALYSIS_DRY:
                    System.out.format("dry spell length\n");
                    break;
                case ANALYSIS_WET:
                    System.out.format("wet spell length\n");
                    break;
                case ANALYSIS_MI:
                    System.out.format("mutual information\n");
                    break;
                case ANALYSIS_LO:
                    System.out.format("log-odds ratio\n");
                    break;
                default:
                    System.out.format("unknown\n");
            }

            System.out.format("Missing value filling analysis type: ");
            switch (params.poking_type) {
                case POKING_TYPE_LOGP:
                    System.out.format("log-probability\n");
                    break;
                case POKING_TYPE_PRED:
                    System.out.format("number of correct predictions\n");
                    break;
                case MISSING_IND_PROB:
                    System.out.format("marginal probabilities for missing values\n");
                    break;
                case HIDDEN_STATE_PROB:
                    System.out.format("probabilities for each value of hidden states\n");
                    break;
                default:
                    System.out.format("unknown\n");
            }

            System.out.format("Lookahead: %d\n", params.lookahead);
        }
        return;
    }

    static void test() {
        //  System.out.format( "%d %d\n", isinf(POS_INF), isinf(NEG_INF) );
        // System.out.format( "%f %f %f\n", exp(NEG_INF), log(exp(NEG_INF)), NEG_INF-NEG_INF );
        //  System.out.format( "%f\n", exp(log(0.0)) );
        //  System.out.format( "%f %f\n", NEG_INF, POS_INF );
        //  System.out.format( "%f\t%f\n", gammaln( 1.0 ), gammaln( 2.0 ) );
        //  srand48( 0 );
        //  System.out.format( "Missing value = %f\n", missing_value( (double)0.0 ) );
        //  System.out.format( "Infinity = %f\n", POS_INF );
        //  System.out.format( "Ratio of inf/inf = %f\n", POS_INF/POS_INF );
        //  System.out.format( "%f\n", 1/sqrt(2*PI*1e-200) );
    }


    void TestDebug(Data data, Data input_data, HMM theta, File output) {

        return;
    }

    int CompareDataSets(Data data1, Data data2) {
  /* Computes the number of not equal categorical entries (excluding NaNs)
     in two data sets */
        int not_equal = 0;           // Number of non-equal entries

        for (int n = 0; n < data1.num_seqs; n++)
            for (int t = 0; t < data1.sequence[n].seq_length; t++)
                for (int i = 0; i < data1.sequence[n].entry[t].ddim; i++)
                    if (!is_missing(data1.sequence[n].entry[t].ddata[i]) &&
                            !is_missing(data2.sequence[n].entry[t].ddata[i]) &&
                            data1.sequence[n].entry[t].ddata[i] !=
                                    data2.sequence[n].entry[t].ddata[i])
                        not_equal++;

        return (not_equal);
    }
}