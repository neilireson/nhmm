package uk.ac.shef.wit.nhmm;

import java.io.File;
import java.io.IOException;

import static uk.ac.shef.wit.nhmm.Constants.*;
import static uk.ac.shef.wit.nhmm.Distribution.ReadDistribution;

public class Parameters {
    /* Contains possible parameters to the program which can be obtained from
       the parameter file.
    */
    /* Read parameters */
    int num_variables;            // Number of latent variables
    int num_states;               // Number of states for each latent variable
    int model_type;               // Type of the model
    int action;                   // Action for the program to take
    Distribution transition;// Transition distribution
    Distribution[] emission; // Emission distributions
    String output_data_filename;    // Name of the file with the data for the outputs
    int num_ddata_components;     // Number of finite-valued components of the output data
    int num_rdata_components;     // Number of real-valued components of the output data
    int num_data_seqs;            // Number of data sequences
    int length_data_seqs;         // Number of entries per data sequence !!! CAN BE CHANGED LATER !!!
    int[] length_data_seq;         // Number of entries for each data sequence
    int input_dim;                // Number of input components
    String input_filename;          // Name of the file with the data for the inputs
    int num_dinput_components;    // Number of finite-valued components of the input data
    int num_rinput_components;    // Number of real-valued components of the input data
    int num_input_seqs;           // Number of input data sequences
    int length_input_seqs;        // Number of entries per input sequence !!! CAN BE CHANGED LATER !!!
    int[] length_input_seq;        // Number of entries for each input sequence
    int xval_type;                // Type of crossvalidation
    int number_out;               // Number of examples to leave out
    String model_filename;          // Name of the file with the model
    int num_models;               // Number of models in the model file
    String output_filename;         // Name of the output file
    String extra_data_filename;     // Name of the extra file (e.g., OLR data)
    int num_dextra_components;    // Number of finite-valued components of the input data
    int num_rextra_components;    // Number of real-valued components of the input data
    int num_extra_seqs;           // Number of input data sequences
    int length_extra_seqs;        // Number of entries per input sequence !!! CAN BE CHANGED LATER !!!
    int num_restarts;             // Number of random restarts of EM
    double em_precision;           // Precision of log-likelihood in EM
    int initialization_type;      // Simple (random) or complex (EM from a simpler model) initialization
    int robust_first_state;       // Probabilities for first state in the sequence are averages
    int num_simulations;          // Number of simulations per
    String state_filename;          // Name of the file to store the states for simulation
    int analysis_type;            // Type of data analysis
    int poking_type;              // Type of hole-poking analysis
    double cg_epsilon;             // Minimum conjugate gradient change
    double maxent_epsilon;         // Minimum maxent log-likelihood change
    int lookahead;                // Look-ahead

    /* Created parameters */
    DataPoint output_datum; // Template for an output data point
    DataPoint input_datum;  // Template for an input data point
    DataPoint extra_datum;  // Template for an extra data point

    boolean em_verbose;
    boolean bare_display;
    boolean short_dist_display;
    boolean input_dist;

    Parameters() {

        /* Filenames */
        output_data_filename = null;
        input_filename = null;
        model_filename = null;
        output_filename = null;
        extra_data_filename = null;

        state_filename = null;

        /* General variables */
        num_variables = 1;
        num_states = 2;
        model_type = MODEL_TYPE_HMM;
        action = ACTION_CODE_UNKNOWN;
        emission = new Distribution[MAXNUMVARIABLES];
        transition = null;
        for (int i = 0; i < MAXNUMVARIABLES; i++)
            emission[i] = null;
        num_ddata_components = 0;
        num_rdata_components = 0;
        num_data_seqs = 1;
        length_data_seqs = 1;
        input_dim = 0;
        num_dinput_components = 0;
        num_rinput_components = 0;
        num_input_seqs = 1;
        length_input_seqs = 1;
        num_dextra_components = 0;
        num_rextra_components = 0;
        num_extra_seqs = 1;
        length_extra_seqs = 1;
        xval_type = XVAL_NONE;
        number_out = 1;
        num_models = 1;
        lookahead = 1;
        length_data_seq = null;
        length_input_seq = null;

        /* Specific variables and options */
        /* Learning */
        num_restarts = 1;
        em_precision = EM_EPSILON;
        robust_first_state = 0;
        initialization_type = INIT_RANDOM;
        num_simulations = 1;
        analysis_type = ANALYSIS_MEAN;
        poking_type = POKING_TYPE_LOGP;

        cg_epsilon = CONJ_GRAD_EPSILON;
        maxent_epsilon = MIN_ME_LL_CHANGE;

        /* Other */
        output_datum = null;
        input_datum = null;
        extra_datum = null;
    }

    void close() {

        for (int i = 0; i < num_variables; i++)
            if (emission[i] != null)
                emission[i] = null;
        emission = null;
        if (transition != null)
            transition = null;

        /* Filenames */
        if (output_data_filename != null)
            output_data_filename = null;
        if (input_filename != null)
            input_filename = null;
        if (model_filename != null)
            model_filename = null;
        if (output_filename != null)
            output_filename = null;
        if (extra_data_filename != null)
            extra_data_filename = null;
        if (state_filename != null)
            state_filename = null;

        if (length_data_seq != null)
            length_data_seq = null;
        if (length_input_seq != null)
            length_input_seq = null;

        if (output_datum != null)
            output_datum = null;
        if (input_datum != null)
            input_datum = null;
        if (extra_datum != null)
            extra_datum = null;
    }

    void Read(File input) throws IOException {
        /* This procedure reads in the parameters from the file */

        String param_name;
        String temp_char;

        ReadFile readFile = new ReadFile(input);
        do {
            param_name = readFile.read_word();
            if (!readFile.EOF_TRUE) {
                if (param_name.compareTo("num_variables") == 0) { /* Reading in the number of states */
                    num_variables = readFile.read_long();
                    if (num_variables < 1) {
                        System.err.format("Error on line %d: number of hidden variables must be at least 1. Setting it to 1.\n",
                                readFile.line_number);
                        num_variables = 1;
                    } else if (num_variables > MAXNUMVARIABLES) {
                        System.err.format("Error on line %d: number of hidden variables has a ceiling of %d. Setting it to the ceiling.\n",
                                readFile.line_number, (int) MAXNUMVARIABLES);
                        num_variables = MAXNUMVARIABLES;
                    }
                    if (INPUT_VERBOSE) {
                        System.out.format("Number of latent variables: %d\n", num_variables);
                    }
                } /* Reading in the number of states */
                if (param_name.compareTo("num_states") == 0) { /* Reading in the number of states */
                    num_states = readFile.read_long();
                    if (num_states < 1) {
                        System.err.format("Error on line %d: number of hidden states must be at least 1. Aborting.\n",
                                readFile.line_number);
                        System.exit(-1);
                    }
                    if (INPUT_VERBOSE) {
                        System.out.format("Number of states: %d\n", num_states);
                    }
                } /* Reading in the number of states */ else if (param_name.compareTo("model_type") == 0) { /* Reading in the type of the model */
                    temp_char = readFile.read_word();
                    if (temp_char.compareTo("hmm") == 0)
                        model_type = MODEL_TYPE_HMM;
                    else if (temp_char.compareTo("nhmm") == 0)
                        model_type = MODEL_TYPE_NHMM;
                    else if (temp_char.compareTo("mixture") == 0)
                        model_type = MODEL_TYPE_MIX;
                    else if (temp_char.compareTo("nmixture") == 0)
                        model_type = MODEL_TYPE_NMIX;
                    else
                        model_type = MODEL_TYPE_HMM;

                    temp_char = null;
                    if (INPUT_VERBOSE) {
                        System.out.format("Type of the model: ");
                        switch (model_type) {
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
                    }
                } /* Reading in the type of the model */ else if (param_name.compareTo("action") == 0) { /* Action */
                    temp_char = readFile.read_word();
                    if (temp_char.compareTo("learn") == 0)
                        /* Estimate parameters */
                        action = ACTION_CODE_LEARN;
                    else if (temp_char.compareTo("viterbi") == 0)
                        /* Find most likely sequence of states */
                        action = ACTION_CODE_VITERBI;
                    else if (temp_char.compareTo("ll") == 0)
                        /* Calculate log-likelihood of the model given the data */
                        action = ACTION_CODE_LL;
                    else if (temp_char.compareTo("ll-train") == 0)
                        /* Calculating log-likelihood of the training set given the model */
                        action = ACTION_CODE_LLTRAIN;
                    else if (temp_char.compareTo("simulation") == 0)
                        /* Simulate the data from the model (and, possibly, inputs) */
                        action = ACTION_CODE_SIM;
                    else if (temp_char.compareTo("analysis") == 0)
                        /* Analyze the data */
                        action = ACTION_CODE_ANALYZE;
                    else if (temp_char.compareTo("filling") == 0)
                        /* Hole poking evaluation */
                        action = ACTION_CODE_FILLING;
                    else if (temp_char.compareTo("sim-filling") == 0)
                        /* Simulating missing entries */
                        action = ACTION_CODE_SIM_FILL;
                    else if (temp_char.compareTo("prediction") == 0)
                        /* Computing predictive ability */
                        action = ACTION_CODE_PREDICT;
                    else if (temp_char.compareTo("init") == 0)
                        /* Creating initial sets of parameters */
                        action = ACTION_CODE_INIT;
                    else if (temp_char.compareTo("KL") == 0)
                        /* Computing Kullback-Leibler divergence */
                        action = ACTION_CODE_KL;
                    else if (temp_char.compareTo("debug") == 0)
                        /* Testing recently implemented functionality */
                        action = ACTION_CODE_DEBUG;
                    else
                        /* None of the above */
                        action = ACTION_CODE_UNKNOWN;

                    temp_char = null;
                    if (INPUT_VERBOSE) {
                        System.out.format("Action: ");
                        switch (action) {
                            case ACTION_CODE_LEARN:
                                System.out.format("learning\n");
                                break;
                            case ACTION_CODE_VITERBI:
                                System.out.format("viterbi\n");
                                break;
                            case ACTION_CODE_LL:
                                System.out.format("log-likelihood\n");
                                break;
                            case ACTION_CODE_LLTRAIN:
                                System.out.format("log-likelihood of the training set\n");
                                break;
                            case ACTION_CODE_SIM:
                                System.out.format("simulation\n");
                                break;
                            case ACTION_CODE_ANALYZE:
                                System.out.format("data analysis\n");
                                break;
                            case ACTION_CODE_FILLING:
                                System.out.format("hole filling evaluation\n");
                                break;
                            case ACTION_CODE_SIM_FILL:
                                System.out.format("simulation of data for missing data\n");
                                break;
                            case ACTION_CODE_PREDICT:
                                System.out.format("prediction\n");
                                break;
                            case ACTION_CODE_INIT:
                                System.out.format("parameter initialization\n");
                                break;
                            case ACTION_CODE_KL:
                                System.out.format("Kullback-Leibler divergence\n");
                                break;
                            case ACTION_CODE_DEBUG:
                                System.out.format("debugging\n");
                                break;
                            default:
                                System.out.format("unknown\n");
                        }
                    }
                } /* Action */ else if (param_name.compareTo("emission") == 0) { /* Reading in the emission parameters */
                    if (INPUT_VERBOSE) {
                        System.out.format("Reading in the emission distributions' specifications for each of %d variables.\n", num_variables);
                    }
                    for (int i = 0; i < num_variables; i++) {
                        if (INPUT_VERBOSE) {
                            System.out.format("Distribution for variable %d:\n", i + 1);
                        }
                        if (emission[i] != null)
                            emission[i] = null;
                        emission[i] = ReadDistribution(input);
                    }
                } /* Reading in the emission parameters */ else if (param_name.compareTo("transition") == 0) { /* Reading in the transition distribution */
                    if (transition != null)
                        transition = null;
                    transition = ReadDistribution(input);
                } /* Reading in the transition distribution */ else if (param_name.compareTo("data") == 0) { /* Data filename */
                    if (output_data_filename != null)
                        output_data_filename = null;
                    output_data_filename = readFile.read_word();

                    if (INPUT_VERBOSE) {
                        System.out.format("Data file name: %s\n", output_data_filename);
                    }
                } /* Data filename */ else if (param_name.compareTo("num_discrete_data_components") == 0) { /* Number of finite-valued components in the output data */
                    num_ddata_components = readFile.read_long();
                    if (num_ddata_components < 0) {
                        System.err.format("Warning on line %d: number of output data finite-valued components must be at non-negative.  Setting to 0.\n",
                                readFile.line_number);
                        num_ddata_components = 0;
                    }
                    if (INPUT_VERBOSE) {
                        System.out.format("Number of finite-valued vector data components: %d\n", num_ddata_components);
                    }
                } /* Number of finite-valued components in the output data */ else if (param_name.compareTo("num_real_data_components") == 0) { /* Number of real-valued components in the output data */
                    num_rdata_components = readFile.read_long();
                    if (num_rdata_components < 0) {
                        System.err.format("Warning on line %d: number of output data real-valued components must be at non-negative.  Setting to 0.\n",
                                readFile.line_number);
                        num_rdata_components = 0;
                    }
                    if (INPUT_VERBOSE) {
                        System.out.format("Number of real-valued vector data components: %d\n", num_rdata_components);
                    }
                } /* Number of real-valued components in the output data */ else if (param_name.compareTo("num_data_sequences") == 0) { /* Number of data sequences */
                    num_data_seqs = readFile.read_long();
                    if (num_data_seqs < 1) {
                        System.err.format("Error on line %d: number of data sequences must be at least 1.  Aborting.\n",
                                readFile.line_number);
                        System.exit(-1);
                    }
                    if (length_data_seq != null) {
                        length_data_seq = null;
                    }
                    if (INPUT_VERBOSE) {
                        System.out.format("Number of data sequences: %d\n", num_data_seqs);
                    }
                } /* Number of data sequences */ else if (param_name.compareTo("data_sequence_length") == 0) { /* Lengths of each data sequence */
                    /* !!! Assumes that lengths of all sequences are the same !!! */

                    /* Reading in the number of entries in each sequence */
                    length_data_seqs = readFile.read_long();
                    if (length_data_seqs < 1) {
                        System.err.format("Error on line %d: lengths of data sequences must be at least 1.  Aborting.\n",
                                readFile.line_number);
                        System.exit(-1);
                    }
                    if (length_data_seq != null) {
                        length_data_seq = null;
                    }
                    if (INPUT_VERBOSE) {
                        System.out.format("Number of data points per sequence: %d\n", length_data_seqs);
                    }
                } /* Lengths of each data sequence */ else if (param_name.compareTo("data_sequence_length_distinct") == 0) { /* Possibly different length for each data sequence */
                    if (length_data_seq != null)
                        length_data_seq = null;
                    length_data_seq = new int[num_data_seqs];
                    for (int i = 0; i < num_data_seqs; i++) {
                        length_data_seq[i] = readFile.read_long();
                        if (length_data_seq[i] < 1) {
                            System.err.format("Error on line %d: length of data sequences must be at least 1. Aborting.\n",
                                    readFile.line_number);
                            System.exit(-1);
                        }
                    }
                    if (INPUT_VERBOSE) {
                        for (int i = 0; i < num_data_seqs; i++)
                            System.out.format("Number of data point for data sequence %d: %d\n", i + 1, length_data_seq[i]);
                    }
                } /* Possibly different length for each data sequence */ else if (param_name.compareTo("input_dimensionality") == 0) { /* Reading in input distribution */
                    if (INPUT_VERBOSE) {
                        System.out.format("Reading in the number of input components:\n");
                    }
                    input_dim = readFile.read_long();
                    if (input_dim < 0) {
                        System.err.format("Error on line %d: number of input components must be at least 0.  Resetting to 0.\n",
                                readFile.line_number);
                        input_dim = 0;
                    }
                    if (INPUT_VERBOSE) {
                        System.out.format("Number of input components: %d\n", input_dim);
                    }
                } /* Reading in input distribution */ else if (param_name.compareTo("input_filename") == 0) { /* Input data filename */
                    if (input_filename != null)
                        input_filename = null;
                    input_filename = readFile.read_word();

                    if (INPUT_VERBOSE) {
                        System.out.format("Input file name : %s\n", input_filename);
                    }
                } /* Input data filename */ else if (param_name.compareTo("num_discrete_input_components") == 0) { /* Number of finite-valued components in the input data */
                    num_dinput_components = readFile.read_long();
                    if (num_dinput_components < 0) {
                        System.err.format("Warning on line %d: number of input data discrete-valued components must be at non-negative.  Setting to 0.\n",
                                readFile.line_number);
                        num_dinput_components = 0;
                    }
                    if (INPUT_VERBOSE) {
                        System.out.format("Number of finite-valued input vector data components: %d\n", num_dinput_components);
                    }
                } /* Number of finite-valued components in the input data */ else if (param_name.compareTo("num_real_input_components") == 0) { /* Number of real-valued components in the input data */
                    num_rinput_components = readFile.read_long();
                    if (num_rinput_components < 0) {
                        System.err.format("Warning on line %d: number of input data real-valued components must be at non-negative.  Setting to 0.\n",
                                readFile.line_number);
                        num_rinput_components = 0;
                    }
                    if (INPUT_VERBOSE) {
                        System.out.format("Number of real-valued vector input data components: %d\n", num_rinput_components);
                    }
                } /* Number of real-valued components in the input data */ else if (param_name.compareTo("num_input_sequences") == 0) { /* Number of input sequences */
                    num_input_seqs = readFile.read_long();
                    if (num_input_seqs < 1) {
                        System.err.format("Error on line %d: number of input sequences must be at least 1.  Aborting.\n",
                                readFile.line_number);
                        System.exit(-1);
                    }
                    if (length_input_seq != null) {
                        length_input_seq = null;
                    }
                    if (INPUT_VERBOSE) {
                        System.out.format("Number of input sequences: %d\n", num_input_seqs);
                    }
                } /* Number of input sequences */ else if (param_name.compareTo("input_sequence_length") == 0) { /* Lengths of each input sequence */
                    /* !!! Assumes that lengths of all sequences are the same !!! */

                    /* Reading in the number of entries in each sequence */
                    length_input_seqs = readFile.read_long();
                    if (length_input_seqs < 1) {
                        System.err.format("Error on line %d: lengths of input sequences must be at least 1.  Aborting.\n",
                                readFile.line_number);
                        System.exit(-1);
                    }
                    if (length_input_seq != null) {
                        length_input_seq = null;
                    }
                    if (INPUT_VERBOSE) {
                        System.out.format("Number of data points per input sequence: %d\n", length_input_seqs);
                    }
                } /* Lengths of each input sequence */ else if (param_name.compareTo("input_sequence_length_distinct") == 0) { /* Possibly different length for each input sequence */
                    if (length_input_seq != null)
                        length_input_seq = null;
                    length_input_seq = new int[num_input_seqs];
                    for (int i = 0; i < num_input_seqs; i++) {
                        length_input_seq[i] = readFile.read_long();
                        if (length_input_seq[i] < 1) {
                            System.err.format("Error on line %d: length of input sequences must be at least 1. Aborting.\n",
                                    readFile.line_number);
                            System.exit(-1);
                        }
                    }
                    if (INPUT_VERBOSE) {
                        for (int i = 0; i < num_input_seqs; i++)
                            System.out.format("Number of data point for input sequence %d: %d\n", i + 1, length_input_seq[i]);
                    }
                } /* Possibly different length for each input sequence */ else if (param_name.compareTo("extra_filename") == 0) { /* Extra data filename */
                    if (extra_data_filename != null)
                        extra_data_filename = null;
                    extra_data_filename = readFile.read_word();

                    if (INPUT_VERBOSE) {
                        System.out.format("Extra data file name : %s\n", extra_data_filename);
                    }
                } /* Extra data filename */ else if (param_name.compareTo("num_discrete_extra_components") == 0) { /* Number of finite-valued components in the extra data */
                    num_dextra_components = readFile.read_long();
                    if (num_dextra_components < 0) {
                        System.err.format("Warning on line %d: number of extra data discrete-valued components must be at non-negative.  Setting to 0.\n",
                                readFile.line_number);
                        num_dextra_components = 0;
                    }
                    if (INPUT_VERBOSE) {
                        System.out.format("Number of finite-valued extra vector data components: %d\n", num_dextra_components);
                    }
                } /* Number of finite-valued components in the extra data */ else if (param_name.compareTo("num_real_extra_components") == 0) { /* Number of real-valued components in the extra data */
                    num_rextra_components = readFile.read_long();
                    if (num_rextra_components < 0) {
                        System.err.format("Warning on line %d: number of extra data real-valued components must be at non-negative.  Setting to 0.\n",
                                readFile.line_number);
                        num_rextra_components = 0;
                    }
                    if (INPUT_VERBOSE) {
                        System.out.format("Number of real-valued vector extra data components: %d\n", num_rextra_components);
                    }
                } /* Number of real-valued components in the extra data */ else if (param_name.compareTo("num_extra_sequences") == 0) { /* Number of input sequences */
                    num_extra_seqs = readFile.read_long();
                    if (num_extra_seqs < 1) {
                        System.err.format("Error on line %d: number of extra sequences must be at least 1.  Aborting.\n",
                                readFile.line_number);
                        System.exit(-1);
                    }
                    if (INPUT_VERBOSE) {
                        System.out.format("Number of extra sequences: %d\n", num_extra_seqs);
                    }
                } /* Number of extra sequences */ else if (param_name.compareTo("extra_sequence_length") == 0) { /* Lengths of each extra sequence */
                    /* !!! Assumes that lengths of all sequences are the same !!! */

                    /* Reading in the number of entries in each sequence */
                    length_extra_seqs = readFile.read_long();
                    if (length_extra_seqs < 1) {
                        System.err.format("Error on line %d: lengths of extra sequences must be at least 1.  Aborting.\n",
                                readFile.line_number);
                        System.exit(-1);
                    }
                    if (INPUT_VERBOSE) {
                        System.out.format("Number of data points per extra sequence: %d\n", length_extra_seqs);
                    }
                } /* Lengths of each extra sequence */ else if (param_name.compareTo("xval_type") == 0) { /* Type of cross-validation */
                    temp_char = readFile.read_word();
                    if (temp_char.compareTo("none") == 0)
                        /* No cross-validation */
                        xval_type = XVAL_NONE;
                    else if (temp_char.compareTo("leave_n_out") == 0)
                        /* Leave-n-out cross-validation */
                        xval_type = XVAL_LEAVENOUT;
                    else
                        /* Default */
                        xval_type = XVAL_NONE;

                    temp_char = null;
                    if (INPUT_VERBOSE) {
                        System.out.format("Cross-validation type: ");
                        switch (xval_type) {
                            case XVAL_NONE:
                                System.out.format("none\n");
                                break;
                            case XVAL_LEAVENOUT:
                                System.out.format("leave-n-out\n");
                                break;
                            default:
                                System.out.format("unknown\n");
                        }
                    }
                } /* Type of cross-validation */ else if (param_name.compareTo("examples_out") == 0) { /* Number of examples to leave out for cross-validation */
                    number_out = readFile.read_long();
                    if (number_out < 1) {
                        System.err.format("Number of examples to leave out is too small.  Setting it to 1.\n");
                        number_out = 1;
                    }
                    if (INPUT_VERBOSE) {
                        System.out.format("Number of examples to leave out: %d\n", number_out);
                    }
                } /* Number of examples to leave out for cross-validation */ else if (param_name.compareTo("model_filename") == 0) { /* Model */

                    /* Reading in the filename of the file with the model */
                    if (model_filename != null)
                        model_filename = null;
                    model_filename = readFile.read_word();

                    if (INPUT_VERBOSE) {
                        System.out.format("Model filename %s\n", model_filename);
                    }
                } /* Model */ else if (param_name.compareTo("num_models") == 0) { /* Number of models in the model file */
                    num_models = readFile.read_long();
                    if (num_models < 1) {
                        System.err.format("Warning on line %d: number of models in the model file must be at least 1.  Setting to 1.\n",
                                readFile.line_number);
                        num_models = 1;
                    }
                    if (INPUT_VERBOSE) {
                        System.out.format("Number of models in the model file: %d (may be adjusted later)\n", num_models);
                    }
                } /* Number of models in the model file */ else if (param_name.compareTo("output") == 0) { /* Output file */
                    output_filename = readFile.read_word();
                    if (INPUT_VERBOSE) {
                        System.out.format("Output filename: %s\n", output_filename);
                    }
                } /* Output file */ else if (param_name.compareTo("num_restarts") == 0) { /* Number of runs for EM */
                    num_restarts = readFile.read_long();
                    if (num_restarts < 1) {
                        System.err.format("Warning on line %d: number of random restarts must be at least 1.  Setting to 1.\n",
                                readFile.line_number);
                        num_restarts = 1;
                    }
                    if (INPUT_VERBOSE) {
                        System.out.format("Number of EM random restarts: %d\n", num_restarts);
                    }
                } /* Number of runs for EM */ else if (param_name.compareTo("em_precision") == 0) { /* Precision of EM algorithm */
                    em_precision = readFile.read_double();
                    if (em_precision < 0.0) {
                        System.err.format("Warning on line %d: EM precision must be positive.  Resetting to default.\n",
                                readFile.line_number);
                        em_precision = EM_EPSILON;
                    }
                    if (INPUT_VERBOSE) {
                        System.out.format("EM with precision %.12f\n", em_precision);
                    }
                } /* Precision of EM algorithm */ else if (param_name.compareTo("em_verbose") == 0) { /* Flag whether to display verbose messages with EM */
                    em_verbose = true;
                    if (INPUT_VERBOSE) {
                        System.out.format("Displaying intermediate messages with EM\n");
                    }
                } /* Flag whether to display verbose messages with EM */ else if (param_name.compareTo("initialization") == 0) { /* Initialization type */
                    temp_char = readFile.read_word();
                    if (temp_char.compareTo("random") == 0)
                        /* Random initial parameter assignment */
                        initialization_type = INIT_RANDOM;
                    else if (temp_char.compareTo("em") == 0)
                        /* EM on a simpler model */
                        initialization_type = INIT_EM;
                    else
                        /* Default */
                        initialization_type = INIT_RANDOM;

                    temp_char = null;
                    if (INPUT_VERBOSE) {
                        System.out.format("Model parameter initialization type: ");
                        switch (initialization_type) {
                            case INIT_RANDOM:
                                System.out.format("random\n");
                                break;
                            case INIT_EM:
                                System.out.format("EM on a simpler model\n");
                                break;
                            default:
                                System.out.format("unknown\n");
                        }
                    }
                } /* Initialization type */ else if (param_name.compareTo("num_simulations") == 0) { /* Number of simulations per run */
                    num_simulations = readFile.read_long();
                    if (num_simulations < 1) {
                        System.err.format("Warning on line %d: number of simulated sequences must be at least 1.  Setting to 1.\n",
                                readFile.line_number);
                        num_simulations = 1;
                    }
                    if (INPUT_VERBOSE) {
                        System.out.format("Number of simulated sequences: %d\n", num_simulations);
                    }
                } /* Number of simulations per run */ else if (param_name.compareTo("state_filename") == 0) { /* State filename */
                    if (state_filename != null)
                        state_filename = null;
                    state_filename = readFile.read_word();
                    if (INPUT_VERBOSE) {
                        System.out.format("Filename for the states %s\n", state_filename);
                    }
                } /* State filename */ else if (param_name.compareTo("analysis") == 0) { /* Type of analysis */
                    temp_char = readFile.read_word();
                    if (temp_char.compareTo("mean") == 0)
                        /* Mean */
                        analysis_type = ANALYSIS_MEAN;
                    else if (temp_char.compareTo("correlation") == 0)
                        /* Correlation */
                        analysis_type = ANALYSIS_CORR;
                    else if (temp_char.compareTo("persistence") == 0)
                        /* Persistence */
                        analysis_type = ANALYSIS_PERS;
                    else if (temp_char.compareTo("dry") == 0)
                        /* Dry spell */
                        analysis_type = ANALYSIS_DRY;
                    else if (temp_char.compareTo("wet") == 0)
                        /* Wet spell */
                        analysis_type = ANALYSIS_WET;
                    else if (temp_char.compareTo("information") == 0)
                        /* Mutual information */
                        analysis_type = ANALYSIS_MI;
                    else if (temp_char.compareTo("logodds") == 0)
                        /* Log-odds ratio */
                        analysis_type = ANALYSIS_LO;
                    else if (temp_char.compareTo("comparison") == 0)
                        /* Comparison of true set to the filled-in set */
                        analysis_type = ANALYSIS_COMP;
                    else
                        /* Default */
                        analysis_type = ANALYSIS_MEAN;

                    temp_char = null;
                    if (INPUT_VERBOSE) {
                        System.out.format("Analysis type: ");
                        switch (analysis_type) {
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
                            case ANALYSIS_COMP:
                                System.out.format("comparison\n");
                                break;
                            default:
                                System.out.format("unknown\n");
                        }
                    }
                } /* Type of analysis */ else if (param_name.compareTo("filling") == 0) { /* Type of hole-filling analysis */
                    temp_char = readFile.read_word();
                    if (temp_char.compareTo("log_p") == 0)
                        /* Log-probability */
                        poking_type = POKING_TYPE_LOGP;
                    else if (temp_char.compareTo("prediction") == 0)
                        /* Prediction */
                        poking_type = POKING_TYPE_PRED;
                    else if (temp_char.compareTo("missing-probabilities") == 0)
                        /* Marginal probabilities for missing values */
                        poking_type = MISSING_IND_PROB;
                    else if (temp_char.compareTo("missing-predictions") == 0)
                        /* Most likely predictions for missing probabilities */
                        poking_type = MISSING_IND_PRED;
                    else if (temp_char.compareTo("hidden-states") == 0)
                        /* Probabilities for hidden states given the data */
                        poking_type = HIDDEN_STATE_PROB;
                    else if (temp_char.compareTo("fill-in") == 0)
                        /* Filling with maximum probability values */
                        poking_type = VITERBI_FILL_IN;
                    else
                        /* Default */
                        poking_type = POKING_TYPE_LOGP;

                    temp_char = null;
                    if (INPUT_VERBOSE) {
                        System.out.format("Hole-filling analysis type: ");
                        switch (poking_type) {
                            case POKING_TYPE_LOGP:
                                System.out.format("log-probability\n");
                                break;
                            case POKING_TYPE_PRED:
                                System.out.format("prediction accuracy\n");
                                break;
                            case MISSING_IND_PROB:
                                System.out.format("marginal probabilities for missng values\n");
                                break;
                            case MISSING_IND_PRED:
                                System.out.format("marginal best-probability predictions for missing values\n");
                                break;
                            case HIDDEN_STATE_PROB:
                                System.out.format("probabilities for hidden states\n");
                                break;
                            case VITERBI_FILL_IN:
                                System.out.format("filling in with maximum probability values\n");
                                break;
                            default:
                                System.out.format("unknown\n");
                        }
                    }
                } /* Type of hole-filling analysis */ else if (param_name.compareTo("robust_first_state") == 0) { /* Robust first state */
                    robust_first_state = 1;
                    if (INPUT_VERBOSE) {
                        System.out.format("Using robust first state probabilities estimation\n");
                    }
                } /* Robust first state */ else if (param_name.compareTo("cg-epsilon") == 0) { /* Minimum conjugate gradient change */
                    cg_epsilon = readFile.read_double();
                    if (cg_epsilon < 0.0) {
                        System.err.format("Warning on line %d: minimum epsilon for conjugate gradient is negative.  Resetting to default.\n",
                                readFile.line_number);
                        cg_epsilon = CONJ_GRAD_EPSILON;
                    }
                    if (INPUT_VERBOSE) {
                        System.out.format("Epsilon for conjugate gradient: %f\n", cg_epsilon);
                    }
                } /* Minimum conjugate gradient change */ else if (param_name.compareTo("maxent-epsilon") == 0) { /* Minimum maxent log-likelihood change */
                    maxent_epsilon = readFile.read_double();
                    if (maxent_epsilon < 0.0) {
                        System.err.format("Warning on line %d: minimum epsilon for maxent log-likelihood is negative.  Resetting to default.\n",
                                readFile.line_number);
                        maxent_epsilon = MIN_ME_LL_CHANGE;
                    }
                    if (INPUT_VERBOSE) {
                        System.out.format("Epsilon for maxent log-likelihood: %f\n", cg_epsilon);
                    }
                } /* Minimum maxent log-likelihood change */ else if (param_name.compareTo("lookahead") == 0) { /* Look-ahead */
                    lookahead = readFile.read_long();
                    if (lookahead < 1) {
                        System.err.format("Warning on line %d: lookahead is less than 1.  Resetting to default\n", readFile.line_number);
                        lookahead = 1;
                    }
                } /* Look-ahead */ else if (param_name.compareTo("short-display") == 0)
                    /* !!! DEFAULT !!! */ { /* Not outputting/reading dimension indices for distributions */
                    short_dist_display = true;
                    if (INPUT_VERBOSE) {
                        System.out.format("Not outputting dimension indices for distributions\n");
                    }
                } /* Not outputting/reading dimension indices for distributions */ else if (param_name.compareTo("dim-index-display") == 0) { /* Outputting/reading dimension indices for distributions */
                    short_dist_display = false;
                    if (INPUT_VERBOSE) {
                        System.out.format("Outputting dimension indices for distributions\n");
                    }
                } /* Outputting/reading dimension indices for distributions */ else if (param_name.compareTo("bare-display") == 0) { /* Outputting parameters only, no comments */
                    bare_display = true;
                    if (INPUT_VERBOSE) {
                        System.out.format("Outtping parameters only, no comments\n");
                    }
                } /* Outputting parameters only, no comments */ else
                    System.err.format("Unknown parameter name on line %d: %s\nSkipping\n", readFile.line_number, param_name);
            }

            param_name = null;
        }
        while (!readFile.EOF_TRUE);

    }

}
