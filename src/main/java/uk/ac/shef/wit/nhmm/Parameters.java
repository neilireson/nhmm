package uk.ac.shef.wit.nhmm;

import java.io.File;
import java.io.IOException;
import java.util.Scanner;

import static uk.ac.shef.wit.nhmm.Constants.*;
import static uk.ac.shef.wit.nhmm.Distribution.ReadDistribution;

public class Parameters {
    /* Contains possible parameters to the program which can be obtained from
       the parameter file.
    */
    /* Read parameters */
    int num_variables;            // Number of latent variables
    int num_states;               // Number of states for each latent variable
    ModelType model_type;               // Type of the model
    ActionCode action;            // Action for the program to take
    Distribution transition;      // Transition distribution
    Distribution[] emission;      // Emission distributions
    String output_data_filename;  // Name of the file with the data for the outputs
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
        model_type = ModelType.hmm;
        action = ActionCode.unknown;
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
        String token;

        Scanner scanner = new Scanner(input);
        while (scanner.hasNext()) {
            param_name = scanner.next();
            if (param_name == null) break;
            else if (param_name.charAt(0) == COMMENT_SYMBOL) {
                scanner.nextLine();
            } else {
                switch (param_name) {
                    case "num_variables": { /* Reading in the number of states */
                        num_variables = scanner.nextInt();
                        if (num_variables < 1) {
                            System.err.format("Error: number of hidden variables must be at least 1. Setting it to 1.\n");
                            num_variables = 1;
                        } else if (num_variables > MAXNUMVARIABLES) {
                            System.err.format("Error: number of hidden variables has a ceiling of %d. Setting it to the ceiling.\n", MAXNUMVARIABLES);
                            num_variables = MAXNUMVARIABLES;
                        }
                        if (INPUT_VERBOSE) {
                            System.out.format("Number of latent variables: %d\n", num_variables);
                        }
                    } /* Reading in the number of states */
                    break;
                    case "num_states": { /* Reading in the number of states */
                        num_states = scanner.nextInt();
                        if (num_states < 1) {
                            System.err.format("Error: number of hidden states must be at least 1. Aborting.\n");
                            System.exit(-1);
                        }
                        if (INPUT_VERBOSE) {
                            System.out.format("Number of states: %d\n", num_states);
                        }
                    } /* Reading in the number of states */
                    break;
                    case "model_type": { /* Reading in the type of the model */
                        token = scanner.next();
                        model_type = ModelType.valueOf(token);

                        if (INPUT_VERBOSE) {
                            System.out.format("Type of the model: %s%n", model_type.getDescription());
                        }
                    } /* Reading in the type of the model */
                    break;
                    case "action": { /* Action */
                        token = scanner.next();
                        action = ActionCode.valueOf(token);

                        if (INPUT_VERBOSE) {
                            System.out.format("Action: %s%n", action.getDescription());
                        }
                    } /* Action */
                    break;
                    case "emission": { /* Reading in the emission parameters */
                        if (INPUT_VERBOSE) {
                            System.out.format("Reading in the emission distributions' specifications for each of %d variables.\n", num_variables);
                        }
                        for (int i = 0; i < num_variables; i++) {
                            if (INPUT_VERBOSE) {
                                System.out.format("Distribution for variable %d:\n", i + 1);
                            }
                            if (emission[i] != null)
                                emission[i] = null;
                            emission[i] = ReadDistribution(scanner);
                        }
                    } /* Reading in the emission parameters */
                    break;
                    case "transition": { /* Reading in the transition distribution */
                        if (transition != null)
                            transition = null;
                        transition = ReadDistribution(scanner);
                    } /* Reading in the transition distribution */
                    break;
                    case "data": { /* Data filename */
                        if (output_data_filename != null)
                            output_data_filename = null;
                        output_data_filename = scanner.next();

                        if (INPUT_VERBOSE) {
                            System.out.format("Data file name: %s\n", output_data_filename);
                        }
                    } /* Data filename */
                    break;
                    case "num_discrete_data_components": { /* Number of finite-valued components in the output data */
                        num_ddata_components = scanner.nextInt();
                        if (num_ddata_components < 0) {
                            System.err.format("Warning: number of output data finite-valued components must be at non-negative.  Setting to 0.\n");
                            num_ddata_components = 0;
                        }
                        if (INPUT_VERBOSE) {
                            System.out.format("Number of finite-valued vector data components: %d\n", num_ddata_components);
                        }
                    } /* Number of finite-valued components in the output data */
                    break;
                    case "num_real_data_components": { /* Number of real-valued components in the output data */
                        num_rdata_components = scanner.nextInt();
                        if (num_rdata_components < 0) {
                            System.err.format("Warning: number of output data real-valued components must be at non-negative.  Setting to 0.\n");
                            num_rdata_components = 0;
                        }
                        if (INPUT_VERBOSE) {
                            System.out.format("Number of real-valued vector data components: %d\n", num_rdata_components);
                        }
                    } /* Number of real-valued components in the output data */
                    break;
                    case "num_data_sequences": { /* Number of data sequences */
                        num_data_seqs = scanner.nextInt();
                        if (num_data_seqs < 1) {
                            System.err.format("Error: number of data sequences must be at least 1.  Aborting.\n");
                            System.exit(-1);
                        }
                        if (length_data_seq != null) {
                            length_data_seq = null;
                        }
                        if (INPUT_VERBOSE) {
                            System.out.format("Number of data sequences: %d\n", num_data_seqs);
                        }
                    } /* Number of data sequences */
                    break;
                    case "data_sequence_length": { /* Lengths of each data sequence */
                        /* !!! Assumes that lengths of all sequences are the same !!! */

                        /* Reading in the number of entries in each sequence */
                        length_data_seqs = scanner.nextInt();
                        if (length_data_seqs < 1) {
                            System.err.format("Error: lengths of data sequences must be at least 1.  Aborting.\n");
                            System.exit(-1);
                        }
                        if (length_data_seq != null) {
                            length_data_seq = null;
                        }
                        if (INPUT_VERBOSE) {
                            System.out.format("Number of data points per sequence: %d\n", length_data_seqs);
                        }
                    } /* Lengths of each data sequence */
                    break;
                    case "data_sequence_length_distinct": { /* Possibly different length for each data sequence */
                        if (length_data_seq != null)
                            length_data_seq = null;
                        length_data_seq = new int[num_data_seqs];
                        for (int i = 0; i < num_data_seqs; i++) {
                            length_data_seq[i] = scanner.nextInt();
                            if (length_data_seq[i] < 1) {
                                System.err.format("Error: length of data sequences must be at least 1. Aborting.\n");
                                System.exit(-1);
                            }
                        }
                        if (INPUT_VERBOSE) {
                            for (int i = 0; i < num_data_seqs; i++)
                                System.out.format("Number of data point for data sequence %d: %d\n", i + 1, length_data_seq[i]);
                        }
                    } /* Possibly different length for each data sequence */
                    break;
                    case "input_dimensionality": { /* Reading in input distribution */
                        if (INPUT_VERBOSE) {
                            System.out.format("Reading in the number of input components:\n");
                        }
                        input_dim = scanner.nextInt();
                        if (input_dim < 0) {
                            System.err.format("Error: number of input components must be at least 0.  Resetting to 0.\n");
                            input_dim = 0;
                        }
                        if (INPUT_VERBOSE) {
                            System.out.format("Number of input components: %d\n", input_dim);
                        }
                    } /* Reading in input distribution */
                    break;
                    case "input_filename": { /* Input data filename */
                        if (input_filename != null)
                            input_filename = null;
                        input_filename = scanner.next();

                        if (INPUT_VERBOSE) {
                            System.out.format("Input file name : %s\n", input_filename);
                        }
                    } /* Input data filename */
                    break;
                    case "num_discrete_input_components": { /* Number of finite-valued components in the input data */
                        num_dinput_components = scanner.nextInt();
                        if (num_dinput_components < 0) {
                            System.err.format("Warning: number of input data discrete-valued components must be at non-negative.  Setting to 0.\n");
                            num_dinput_components = 0;
                        }
                        if (INPUT_VERBOSE) {
                            System.out.format("Number of finite-valued input vector data components: %d\n", num_dinput_components);
                        }
                    } /* Number of finite-valued components in the input data */
                    break;
                    case "num_real_input_components": { /* Number of real-valued components in the input data */
                        num_rinput_components = scanner.nextInt();
                        if (num_rinput_components < 0) {
                            System.err.format("Warning: number of input data real-valued components must be at non-negative.  Setting to 0.\n");
                            num_rinput_components = 0;
                        }
                        if (INPUT_VERBOSE) {
                            System.out.format("Number of real-valued vector input data components: %d\n", num_rinput_components);
                        }
                    } /* Number of real-valued components in the input data */
                    break;
                    case "num_input_sequences": { /* Number of input sequences */
                        num_input_seqs = scanner.nextInt();
                        if (num_input_seqs < 1) {
                            System.err.format("Error: number of input sequences must be at least 1.  Aborting.\n");
                            System.exit(-1);
                        }
                        if (length_input_seq != null) {
                            length_input_seq = null;
                        }
                        if (INPUT_VERBOSE) {
                            System.out.format("Number of input sequences: %d\n", num_input_seqs);
                        }
                    } /* Number of input sequences */
                    break;
                    case "input_sequence_length": { /* Lengths of each input sequence */
                        /* !!! Assumes that lengths of all sequences are the same !!! */

                        /* Reading in the number of entries in each sequence */
                        length_input_seqs = scanner.nextInt();
                        if (length_input_seqs < 1) {
                            System.err.format("Error: lengths of input sequences must be at least 1.  Aborting.\n");
                            System.exit(-1);
                        }
                        if (length_input_seq != null) {
                            length_input_seq = null;
                        }
                        if (INPUT_VERBOSE) {
                            System.out.format("Number of data points per input sequence: %d\n", length_input_seqs);
                        }
                    } /* Lengths of each input sequence */
                    break;
                    case "input_sequence_length_distinct": { /* Possibly different length for each input sequence */
                        if (length_input_seq != null)
                            length_input_seq = null;
                        length_input_seq = new int[num_input_seqs];
                        for (int i = 0; i < num_input_seqs; i++) {
                            length_input_seq[i] = scanner.nextInt();
                            if (length_input_seq[i] < 1) {
                                System.err.format("Error: length of input sequences must be at least 1. Aborting.\n");
                                System.exit(-1);
                            }
                        }
                        if (INPUT_VERBOSE) {
                            for (int i = 0; i < num_input_seqs; i++)
                                System.out.format("Number of data point for input sequence %d: %d\n", i + 1, length_input_seq[i]);
                        }
                    } /* Possibly different length for each input sequence */
                    break;
                    case "extra_filename": { /* Extra data filename */
                        if (extra_data_filename != null)
                            extra_data_filename = null;
                        extra_data_filename = scanner.next();

                        if (INPUT_VERBOSE) {
                            System.out.format("Extra data file name : %s\n", extra_data_filename);
                        }
                    } /* Extra data filename */
                    break;
                    case "num_discrete_extra_components": { /* Number of finite-valued components in the extra data */
                        num_dextra_components = scanner.nextInt();
                        if (num_dextra_components < 0) {
                            System.err.format("Warning: number of extra data discrete-valued components must be at non-negative.  Setting to 0.\n");
                            num_dextra_components = 0;
                        }
                        if (INPUT_VERBOSE) {
                            System.out.format("Number of finite-valued extra vector data components: %d\n", num_dextra_components);
                        }
                    } /* Number of finite-valued components in the extra data */
                    break;
                    case "num_real_extra_components": { /* Number of real-valued components in the extra data */
                        num_rextra_components = scanner.nextInt();
                        if (num_rextra_components < 0) {
                            System.err.format("Warning: number of extra data real-valued components must be at non-negative.  Setting to 0.\n");
                            num_rextra_components = 0;
                        }
                        if (INPUT_VERBOSE) {
                            System.out.format("Number of real-valued vector extra data components: %d\n", num_rextra_components);
                        }
                    } /* Number of real-valued components in the extra data */
                    break;
                    case "num_extra_sequences": { /* Number of input sequences */
                        num_extra_seqs = scanner.nextInt();
                        if (num_extra_seqs < 1) {
                            System.err.format("Error: number of extra sequences must be at least 1.  Aborting.\n");
                            System.exit(-1);
                        }
                        if (INPUT_VERBOSE) {
                            System.out.format("Number of extra sequences: %d\n", num_extra_seqs);
                        }
                    } /* Number of extra sequences */
                    break;
                    case "extra_sequence_length": { /* Lengths of each extra sequence */
                        /* !!! Assumes that lengths of all sequences are the same !!! */

                        /* Reading in the number of entries in each sequence */
                        length_extra_seqs = scanner.nextInt();
                        if (length_extra_seqs < 1) {
                            System.err.format("Error: lengths of extra sequences must be at least 1.  Aborting.\n");
                            System.exit(-1);
                        }
                        if (INPUT_VERBOSE) {
                            System.out.format("Number of data points per extra sequence: %d\n", length_extra_seqs);
                        }
                    } /* Lengths of each extra sequence */
                    break;
                    case "xval_type": { /* Type of cross-validation */
                        token = scanner.next();
                        switch (token) {
                            case "none":
                                /* No cross-validation */
                                xval_type = XVAL_NONE;
                                break;
                            case "leave_n_out":
                                /* Leave-n-out cross-validation */
                                xval_type = XVAL_LEAVENOUT;
                            default:
                                /* Default */
                                xval_type = XVAL_NONE;
                        }
                        token = null;
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
                    } /* Type of cross-validation */
                    break;
                    case "examples_out": { /* Number of examples to leave out for cross-validation */
                        number_out = scanner.nextInt();
                        if (number_out < 1) {
                            System.err.format("Number of examples to leave out is too small.  Setting it to 1.\n");
                            number_out = 1;
                        }
                        if (INPUT_VERBOSE) {
                            System.out.format("Number of examples to leave out: %d\n", number_out);
                        }
                    } /* Number of examples to leave out for cross-validation */
                    break;
                    case "model_filename": { /* Model */

                        /* Reading in the filename of the file with the model */
                        if (model_filename != null)
                            model_filename = null;
                        model_filename = scanner.next();

                        if (INPUT_VERBOSE) {
                            System.out.format("Model filename %s\n", model_filename);
                        }
                    } /* Model */
                    break;
                    case "num_models": { /* Number of models in the model file */
                        num_models = scanner.nextInt();
                        if (num_models < 1) {
                            System.err.format("Warning: number of models in the model file must be at least 1.  Setting to 1.\n");
                            num_models = 1;
                        }
                        if (INPUT_VERBOSE) {
                            System.out.format("Number of models in the model file: %d (may be adjusted later)\n", num_models);
                        }
                    } /* Number of models in the model file */
                    break;
                    case "output": { /* Output file */
                        output_filename = scanner.next();
                        if (INPUT_VERBOSE) {
                            System.out.format("Output filename: %s\n", output_filename);
                        }
                    } /* Output file */
                    break;
                    case "num_restarts": { /* Number of runs for EM */
                        num_restarts = scanner.nextInt();
                        if (num_restarts < 1) {
                            System.err.format("Warning: number of random restarts must be at least 1.  Setting to 1.\n");
                            num_restarts = 1;
                        }
                        if (INPUT_VERBOSE) {
                            System.out.format("Number of EM random restarts: %d\n", num_restarts);
                        }
                    } /* Number of runs for EM */
                    break;
                    case "em_precision": { /* Precision of EM algorithm */
                        em_precision = scanner.nextDouble();
                        if (em_precision < 0.0) {
                            System.err.format("Warning: EM precision must be positive.  Resetting to default.\n");
                            em_precision = EM_EPSILON;
                        }
                        if (INPUT_VERBOSE) {
                            System.out.format("EM with precision %.12f\n", em_precision);
                        }
                    } /* Precision of EM algorithm */
                    break;
                    case "em_verbose": { /* Flag whether to display verbose messages with EM */
                        em_verbose = true;
                        if (INPUT_VERBOSE) {
                            System.out.format("Displaying intermediate messages with EM\n");
                        }
                    } /* Flag whether to display verbose messages with EM */
                    break;
                    case "initialization": { /* Initialization type */
                        token = scanner.next();
                        switch (token) {
                            case "random":
                                /* Random initial parameter assignment */
                                initialization_type = INIT_RANDOM;
                                break;
                            case "em":
                                /* EM on a simpler model */
                                initialization_type = INIT_EM;
                                break;
                            default:
                                /* Default */
                                initialization_type = INIT_RANDOM;
                        }
                        token = null;
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
                    } /* Initialization type */
                    break;
                    case "num_simulations": { /* Number of simulations per run */
                        num_simulations = scanner.nextInt();
                        if (num_simulations < 1) {
                            System.err.format("Warning: number of simulated sequences must be at least 1.  Setting to 1.\n");
                            num_simulations = 1;
                        }
                        if (INPUT_VERBOSE) {
                            System.out.format("Number of simulated sequences: %d\n", num_simulations);
                        }
                    } /* Number of simulations per run */
                    break;
                    case "state_filename": { /* State filename */
                        if (state_filename != null)
                            state_filename = null;
                        state_filename = scanner.next();
                        if (INPUT_VERBOSE) {
                            System.out.format("Filename for the states %s\n", state_filename);
                        }
                    } /* State filename */
                    break;
                    case "analysis": { /* Type of analysis */
                        token = scanner.next();
                        switch (token) {
                            case "mean":
                                /* Mean */
                                analysis_type = ANALYSIS_MEAN;
                                break;
                            case "correlation":
                                /* Correlation */
                                analysis_type = ANALYSIS_CORR;
                                break;
                            case "persistence":
                                /* Persistence */
                                analysis_type = ANALYSIS_PERS;
                                break;
                            case "dry":
                                /* Dry spell */
                                analysis_type = ANALYSIS_DRY;
                                break;
                            case "wet":
                                /* Wet spell */
                                analysis_type = ANALYSIS_WET;
                                break;
                            case "information":
                                /* Mutual information */
                                analysis_type = ANALYSIS_MI;
                                break;
                            case "logodds":
                                /* Log-odds ratio */
                                analysis_type = ANALYSIS_LO;
                                break;
                            case "comparison":
                                /* Comparison of true set to the filled-in set */
                                analysis_type = ANALYSIS_COMP;
                            default:
                                /* Default */
                                analysis_type = ANALYSIS_MEAN;
                        }
                        token = null;
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
                    } /* Type of analysis */
                    break;
                    case "filling": { /* Type of hole-filling analysis */
                        token = scanner.next();
                        switch (token) {
                            case "log_p":
                                /* Log-probability */
                                poking_type = POKING_TYPE_LOGP;
                                break;
                            case "prediction":
                                /* Prediction */
                                poking_type = POKING_TYPE_PRED;
                                break;
                            case "missing-probabilities":
                                /* Marginal probabilities for missing values */
                                poking_type = MISSING_IND_PROB;
                                break;
                            case "missing-predictions":
                                /* Most likely predictions for missing probabilities */
                                poking_type = MISSING_IND_PRED;
                                break;
                            case "hidden-states":
                                /* Probabilities for hidden states given the data */
                                poking_type = HIDDEN_STATE_PROB;
                                break;
                            case "fill-in":
                                /* Filling with maximum probability values */
                                poking_type = VITERBI_FILL_IN;
                            default:
                                /* Default */
                                poking_type = POKING_TYPE_LOGP;
                        }
                        token = null;
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
                    } /* Type of hole-filling analysis */
                    break;
                    case "robust_first_state": { /* Robust first state */
                        robust_first_state = 1;
                        if (INPUT_VERBOSE) {
                            System.out.format("Using robust first state probabilities estimation\n");
                        }
                    } /* Robust first state */
                    break;
                    case "cg-epsilon": { /* Minimum conjugate gradient change */
                        cg_epsilon = scanner.nextDouble();
                        if (cg_epsilon < 0.0) {
                            System.err.format("Warning: minimum epsilon for conjugate gradient is negative.  Resetting to default.\n");
                            cg_epsilon = CONJ_GRAD_EPSILON;
                        }
                        if (INPUT_VERBOSE) {
                            System.out.format("Epsilon for conjugate gradient: %f\n", cg_epsilon);
                        }
                    } /* Minimum conjugate gradient change */
                    break;
                    case "maxent-epsilon": { /* Minimum maxent log-likelihood change */
                        maxent_epsilon = scanner.nextDouble();
                        if (maxent_epsilon < 0.0) {
                            System.err.format("Warning: minimum epsilon for maxent log-likelihood is negative.  Resetting to default.\n");
                            maxent_epsilon = MIN_ME_LL_CHANGE;
                        }
                        if (INPUT_VERBOSE) {
                            System.out.format("Epsilon for maxent log-likelihood: %f\n", cg_epsilon);
                        }
                    } /* Minimum maxent log-likelihood change */
                    break;
                    case "lookahead": { /* Look-ahead */
                        lookahead = scanner.nextInt();
                        if (lookahead < 1) {
                            System.err.format("Warning: lookahead is less than 1.  Resetting to default\n");
                            lookahead = 1;
                        }
                    } /* Look-ahead */
                    break;
                    case "short-display":
                        /* !!! DEFAULT !!! */
                    { /* Not outputting/reading dimension indices for distributions */
                        short_dist_display = true;
                        if (INPUT_VERBOSE) {
                            System.out.format("Not outputting dimension indices for distributions\n");
                        }
                    } /* Not outputting/reading dimension indices for distributions */
                    break;
                    case "dim-index-display": { /* Outputting/reading dimension indices for distributions */
                        short_dist_display = false;
                        if (INPUT_VERBOSE) {
                            System.out.format("Outputting dimension indices for distributions\n");
                        }
                    } /* Outputting/reading dimension indices for distributions */
                    break;
                    case "bare-display": { /* Outputting parameters only, no comments */
                        bare_display = true;
                        if (INPUT_VERBOSE) {
                            System.out.format("Outtping parameters only, no comments\n");
                        }
                    }
                    break;
                    default:
                        /* Outputting parameters only, no comments */
                        System.err.format("Unknown parameter name: %s\nSkipping\n", param_name);
                }
            }
        }

    }

}
