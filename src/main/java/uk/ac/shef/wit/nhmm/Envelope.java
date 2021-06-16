package uk.ac.shef.wit.nhmm;

import java.io.PrintStream;

import static uk.ac.shef.wit.nhmm.Constants.*;

public class Envelope {
    /* Structure to keep track of chains of missing values */
    public int[] node;
    public int[] edge;
    public int num_nodes;
    public int num_edges;
    public boolean is_missing;


    Envelope(int max_num) {
        node = new int[max_num];
        edge = new int[max_num];
        num_nodes = 0;
        num_edges = 0;
        is_missing = true;
    }


    void close() {
        node = null;
        edge = null;
    }

    static void RunLogLikelihoodData(Data output_data, Data input_data, HMM theta, PrintStream out) {

        /* Allocating necessary structures */
        theta.AllocateEMStructures(output_data, input_data);

        /* Calculating the log-likelihood of the data */
        out.format("%.12f\n", theta.log_likelihood(output_data, input_data));

        /* Deallocating arrays */
        theta.DeallocateEMStructures(output_data, input_data);
    }

    static void RunSimulateData(Data input_data,
                                HMM theta,
                                Parameters params,
                                PrintStream state_output_file,
                                PrintStream output_file,
                                int num_seqs) {

        int[][] sim_states; // Simulated states
        Data output_data; // Simulated data

        /* State sequences first */
        sim_states = theta.simulate_states(input_data,
                num_seqs,
                params.length_data_seqs);

        if (state_output_file != null) { /* Displaying the states */
            for (int n = 0; n < num_seqs; n++)
                for (int t = 0; t < params.length_data_seqs; t++)
                    state_output_file.format("%d\n", sim_states[n][t]);
        } /* Displaying the states */

        /* Having generated the state sequences, can generate the data */
        output_data = new Data(num_seqs);

        /* Simulating the outputs */
        output_data.Simulate(theta.emission[0], sim_states, params.length_data_seqs, params.output_datum);

        output_data.WriteToFile(output_file);

        /* Deallocating states */
        for (int n = 0; n < num_seqs; n++)
            sim_states[n] = null;
        sim_states = null;

        /* Deallocating the data */
        output_data = null;
    }

    static void RunHoleData(Data output_data, Data input_data, HMM theta, PrintStream out, int hole_type) {

        Data fillin_data;

        /* Allocating necessary structures */
        //theta.AllocateAlpha( output_data, input_data );
        theta.AllocateEMStructures(output_data, input_data);

        switch (hole_type) {
            case POKING_TYPE_LOGP:
            case POKING_TYPE_PRED:
                /* Calculating the predictive score of the model */
                out.format("%.12f\n", theta.hole_score(output_data, input_data, hole_type));
                break;
            case MISSING_IND_PROB:
                theta.MissingIndividualMarginals(output_data, input_data, out, 0);
                break;
            case MISSING_IND_PRED:
                theta.MissingIndividualMarginals(output_data, input_data, out, 1);
                break;
            case HIDDEN_STATE_PROB:
                theta.StateProbabilities(output_data, input_data, out);
                break;
            case VITERBI_FILL_IN:
                fillin_data = theta.FillIn(output_data);
                if (fillin_data != null)
                    fillin_data.WriteToFile(out);

                if (fillin_data != null)
                    fillin_data = null;
                break;
        }

        /* Deallocating arrays */
        //  theta.DeallocateAlpha( output_data, input_data );
        theta.DeallocateEMStructures(output_data, input_data);

    }

    static void RunPredictionData(Data data, Data input, HMM theta, int lookahead, PrintStream out) {
        double ll;

        theta.AllocateEMStructures(data, input);

        for (int n = 0; n < data.num_seqs; n++) {
            /* Calculating emission probabilities */
            theta.CalculateProbEmissionsSequence(data.sequence[n], n, 0);

            /* Calculating P(S_nt|S_n,t-1, X_nt) */
            if (input != null)
                theta.transition.CalculateTransitionValuesSequence(input.sequence[n], n, 0);
            else
                theta.transition.CalculateTransitionValuesSequence(null, n, 0);

            /* Calculating scaled alpha values */
            theta.transition.CalculateForwardUpdatesSequence(data.sequence[n], n, 0, theta.log_pb);

            for (int t = 0; t < data.sequence[n].seq_length - lookahead + 1;
                 t++) {
                ll = 0.0;
                for (int tau = t; tau < t + lookahead; tau++)
                    ll -= theta.transition.log_upd_scale[n][t];
                System.out.format("%.12f\n", ll);
            }
        }

        theta.DeallocateEMStructures(data, input);

        return;
    }


    static void RunGenerateParameters(HMM theta,
                                      int num_models,
                                      boolean bare_display,
                                      PrintStream output_file) {

        /* Initializing parameters */

        for (int i = 0; i < num_models; i++) {
            theta.Initialize(null, null);
            if (bare_display)
                theta.WriteToFileBare(output_file);
            else {
                System.out.format("%c Parameter set #%d\n", COMMENT_SYMBOL, i + 1);
                theta.WriteToFile(output_file);
            }
        }

    }
}
