package uk.ac.shef.wit.nhmm;

import java.io.PrintStream;

import static java.lang.Math.floor;
import static uk.ac.shef.wit.nhmm.Constants.MAXBESTSEQ;
import static uk.ac.shef.wit.nhmm.Constants.NEG_INF;

public class StateSequence {
    double VITERBI_EPSILON = 1E-12;

    int[][] state = new int[MAXBESTSEQ][];         // Sequences collected
    double[] ll = new double[MAXBESTSEQ];           // Log-likelihood of the sequence
    int seq_length;                 // Length of the sequences
    int num_best;                   // Number of best sequences collected
    int default_num_best;           // Number of best sequences requested

    /* Routines for the structures of state sequences */
    StateSequence(int sl, int def) {

        seq_length = sl;
        default_num_best = def;
        if (def > MAXBESTSEQ)
            def = MAXBESTSEQ;
        num_best = 0;
    }

    void close() {
        for (int i = 0; i < num_best; i++)
            state[i] = null;
    }

    void AddSequence(int[] new_seq, double new_ll) {
        /* Adding a sequence to the list or discards (deallocates) it */


        /* !!! Need to insert hooks not to go over MAXBESTSEQ sequences !!! */
        if (new_ll > NEG_INF) {
            int new_index = find_insert_index(ll, new_ll, num_best);
            if (new_index < default_num_best || (num_best > 0 && new_ll >= ll[num_best - 1] - VITERBI_EPSILON)) { /* Adding the entry */

                /* Shifting the tail */
                if (new_index < num_best) { /* Shifting is needed */
                    if (!(num_best < MAXBESTSEQ)) { /* !!! !!! */
                        System.exit(-1);
                    } /* !!! !!! */

                    for (int i = num_best; i > new_index; i--) {
                        state[i] = state[i - 1];
                        ll[i] = ll[i - 1];
                    }
                }

                /* Inserting new sequence */
                state[new_index] = new_seq;
                ll[new_index] = new_ll;
                num_best++;

                /* Chopping the tail */
                if (num_best > default_num_best && ll[default_num_best - 1] - VITERBI_EPSILON > ll[default_num_best]) { /* Need to chop the tail */
                    for (int i = default_num_best; i < num_best; i++)
                        state[i] = null;
                    num_best = default_num_best;
                } /* Need to chop the tail */
            } /* Adding the entry */ else
                /* Deallocating the entry */
                new_seq = null;
        } else
            /* Deallocating the entry */
            new_seq = null;
    }

    int find_insert_index(double[] ll_array, double ll, int ll_length) {
        /* Returns the index indicating where the new entry should go in the decreasing-order array */
        /* Uses binary insertion */

        int high_index;
        int low_index;
        int mid_index;

        high_index = ll_length;
        low_index = 0;
        mid_index = (int) floor(0.5 * (double) (high_index + low_index));
        while (low_index < high_index) {
            if (ll > ll_array[mid_index])
                high_index = mid_index;
            else
                low_index = mid_index + 1;

            mid_index = (int) floor(0.5 * (double) (high_index + low_index));
        }

        return (mid_index);
    }

    void WriteToFile(PrintStream out) {

        /* Displaying all best sequences */
        for (int j = 0; j < num_best; j++) {
            for (int i = 0; i < seq_length; i++)
                out.format(" %d", state[j][i]);
            /* Every sequences ends with a new line */
            out.format("\n");
        }

        out.flush();
    }
}
