package uk.ac.shef.wit.nhmm;

import java.io.File;
import java.io.IOException;
import java.io.PrintStream;

import static java.lang.Math.abs;
import static java.lang.Math.log;
import static uk.ac.shef.wit.nhmm.Constants.*;

public class Data {

    Sequence[] sequence; /* Array of sequences */
    int num_seqs;
    String nametag;

    private boolean is_primary;              // Indicator whether the set is primary or not
    // Primary set points to sequences from other sets
    // and need not deallocate its sequences

    Data(int n_seqs) {
        this(n_seqs, null, false);
    }
        Data(int n_seqs, String tag, boolean is_p) {
  /* Individual sequences are initialized in other functions:
     ReadData
  */

        num_seqs = n_seqs;
        is_primary = is_p;

        /* Allocating the array of sequences */
        sequence = new Sequence[num_seqs];
        for (int n = 0; n < num_seqs; n++)
            sequence[n] = null;

        /* Name tag */
        nametag = tag;
    }

    void close() {
        if (!is_primary)
            for (int i = 0; i < num_seqs; i++)
                if (sequence[i] != null)
                    sequence[i] = null;

        sequence = null;
    }

    void WriteToFile(PrintStream out) {

        for (int i = 0; i < num_seqs; i++)
            if (sequence[i] != null)
                sequence[i].WriteToFile(out);

        out.flush();
    }

    int num_points() {
        /* Returns the number of vectors in the data set */
        int num = 0;

        for (int i = 0; i < num_seqs; i++)
            if (sequence[i] != null)
                num += sequence[i].seq_length;

        return (num);
    }

    int total_entries() {
        int num_total_entries = 0;

        for (int n = 0; n < num_seqs; n++)
            if (sequence[n] != null)
                num_total_entries += sequence[n].total_entries();

        return (num_total_entries);
    }

    double weighted_total_entries(double[][] weight) {
        double total = 0.0;

        for (int n = 0; n < num_seqs; n++)
            if (sequence[n] != null)
                total += sequence[n].weighted_total_entries(weight[n]);

        return (total);
    }

    void MissingEntryIndicators() {
        /* Creating indicators for missing entries */

        for (int n = 0; n < num_seqs; n++)
            sequence[n].MissingEntryIndicators();
    }

    public static boolean is_missing(double d) {
        return (Double.isNaN(d));
    }

    public static boolean is_missing(int d) {
        return (d == CONST_MISSING_VALUE);
    }

    public static double missing_value(double d) {
        return Double.NaN;
    }

    public static int missing_value(int d) {
        return CONST_MISSING_VALUE;
    }

    double[][] log_odds_ratio() {
        /* Calculating log-odds ratio for each pair of dimensions */
        double[][][][] counts;
        double[][] lor;
        int dim;

        dim = sequence[0].entry[0].ddim;

        /* Allocating the arrays of counts */
        counts = new double[dim][][][];
        for (int i = 0; i < dim; i++) {
            counts[i] = new double[i + 1][][];
            for (int j = 0; j < i + 1; j++) {
                counts[i][j] = new double[2][]; /* !!! !!! */
                for (int k = 0; k < 2 /* !!! */; k++)
                    counts[i][j][k] = new double[2]; /* !!! !!! */
            }
        }

        /* Allocating the MI matrix */
        lor = new double[dim][];
        for (int i = 0; i < dim; i++)
            lor[i] = new double[dim];

        /* Initializing the counts */
        for (int i = 0; i < dim; i++)
            for (int j = 0; j < i + 1; j++)
                for (int k = 0; k < 2 /* !!! !!! */; k++)
                    for (int l = 0; l < 2 /* !!! !!! */; l++)
                        counts[i][j][k][l] = 0.0;

        /* Calculating the counts */
        for (int k = 0; k < num_seqs; k++)
            for (int l = 0; l < sequence[k].seq_length; l++)
                for (int i = 0; i < dim; i++)
                    for (int j = 0; j < i + 1; j++)
                        if (!is_missing(sequence[k].entry[l].ddata[i]) &&
                                !is_missing(sequence[k].entry[l].ddata[j]))
                            counts[i][j][sequence[k].entry[l].ddata[i]][sequence[k].entry[l].ddata[j]] += 1.0;

        /* Calculating mutual information */
        for (int i = 0; i < dim; i++)
            for (int j = 0; j < i; j++) {
                /* !!! Works only for binary data !!! */
                /* Log-odds ratio = log(11)+log(00)-log(01)-log(10) */
                lor[i][j] = log(counts[i][j][1][1]) + log(counts[i][j][0][0])
                        - log(counts[i][j][0][1]) - log(counts[i][j][1][0]);

                lor[j][i] = lor[i][j];
            }

        for (int i = 0; i < dim; i++)
            /* Diagonal entries cannot be made sense of */
            lor[i][i] = 0.0;

        /* Deallocating the array of counts */
        for (int i = 0; i < dim; i++) {
            for (int j = 0; j < i + 1; j++) {
                for (int k = 0; k < 2 /* !!! */; k++)
                    counts[i][j][k] = null;
                counts[i][j] = null;
            }
            counts[i] = null;
        }
        counts = null;

        return (lor);
    }

    void ReadData(File data_file, int start_position, int end_position, int seq_length, DataPoint datum )
            throws IOException {

        for (int i =start_position; i<end_position; i++ )
        {
            /* Initializing and reading in the sequence i */
            if( sequence[i] != null)
                sequence[i] = null;
            sequence[i]=new Sequence( seq_length );
            sequence[i].ReadData( data_file, datum );
            if (INPUT_VERBOSE) {
                System.out.format( "Read sequence #%d\n", i+1 );
            }
        }

    }

    void ReadDataDistinct( File data_file, int start_position, int end_position, int[] seq_length, DataPoint datum ) throws IOException {

        for (int i =start_position; i<end_position; i++ )
        {
            /* Initializing and reading in the sequence i */
            if( sequence[i] != null)
                sequence[i] = null;
            sequence[i]=new Sequence( seq_length[i] );
            sequence[i].ReadData( data_file, datum );
            if (INPUT_VERBOSE) {
                System.out.format( "Read sequence #%d\n", i+1 );
            }
        }
    }

    void Simulate(Distribution[]dist, int[][]states, int num_entries, DataPoint datum) {
        /* Simulating sequences of data according to distribution dist and already generated hidden states */

        for (int n = 0; n < num_seqs; n++) {
            /* Initializing sequence n */
            if (sequence[n]!=null)
                sequence[n] = null;
            sequence[n] = new Sequence(num_entries);

            if (SIM_VERBOSE) {
                System.out.format("Sequence %d", n);
            }
            /* Simulating sequence n */
            sequence[n].Simulate(dist, states[n], datum);
            if (SIM_VERBOSE) {
                System.out.format("Done\n");
            }
        }
    }

    double[] mean() {
        /* Calculates percentage of rain occurences for each station */

        int ddim, rdim;
        int[]num_present;  /* Number of observations for each component */
        double[]total;      /* Running sum per component */

        /* Initializing hits and total arrays */
        /* !!! Assumes same dimensionality for all entries !!! */
        ddim = sequence[0].entry[0].ddim;
        rdim = sequence[0].entry[0].rdim;

        num_present = new int[ddim + rdim];
        total = new double[ddim + rdim];
        for (int i = 0; i < ddim + rdim; i++) {
            num_present[i] = 0;
            total[i] = 0.0;
        }

        for (int n = 0; n < num_seqs; n++)
            for (int t = 0; t < sequence[n].seq_length; t++) {
                for (int i = 0; i < ddim; i++)
                    if (!is_missing(sequence[n].entry[t].ddata[i])) {
                        num_present[i]++;
                        if (sequence[n].entry[t].ddata[i] == 1)
                            total[i] += 1.0;
                    }
                for (int i = 0; i < rdim; i++)
                    if (!is_missing(sequence[n].entry[t].rdata[i])) {
                        num_present[ddim + i]++;
                        total[ddim + i] += sequence[n].entry[t].rdata[i];
                    }
            }

        /* Computing the array of outputs */
        for (int i = 0; i < ddim + rdim; i++)
            total[i] /= (double) num_present[i];

        /* Deallocating hits arrays */
        num_present = null;

        return (total);
    }

    double[][][] covariance() {
        /* Calculating covariance */
        double[]mu;
        double[][][]cov;
        int[][]num_present_d = null;
        int[][]num_present_r = null;
        int ddim, rdim;

        /* First, need to know the mean */
        mu = mean();

        /* Initializing covariance matrix */
        /* !!! Assumes same dimensionality for all entries !!! */
        ddim = sequence[0].entry[0].ddim;
        rdim = sequence[0].entry[0].rdim;

        cov = new double [2][][];
        if (ddim > 0) {
            num_present_d = new int [ddim][];
            for (int i = 0; i < ddim; i++) {
                num_present_d[i] = new int[ddim];
                for (int j = 0; j < ddim; j++)
                    num_present_d[i][j] = 0;
            }

            cov[0] = new double [ddim][];
            for (int i = 0; i < ddim; i++) {
                cov[0][i] = new double[ddim];
                for (int j = 0; j < ddim; j++)
                    cov[0][i][j] = 0.0;
            }
        } else
            cov[0] = null;

        if (rdim > 0) {
            num_present_r = new int [rdim][];
            for (int i = 0; i < rdim; i++) {
                num_present_r[i] = new int[rdim];
                for (int j = 0; j < rdim; j++)
                    num_present_r[i][j] = 0;
            }

            cov[1] = new double [rdim][];
            for (int i = 0; i < rdim; i++) {
                cov[1][i] = new double[rdim];
                for (int j = 0; j < rdim; j++)
                    cov[1][i][j] = 0.0;
            }
        } else
            cov[1] = null;

        for (int n = 0; n < num_seqs; n++)
            for (int t = 0; t < sequence[n].seq_length; t++) {
                for (int i = 0; i < ddim; i++)
                    for (int j = 0; j <= i; j++)
                        if (!is_missing(sequence[n].entry[t].ddata[i]) &&
                                !is_missing(sequence[n].entry[t].ddata[j])) {
                            num_present_d[i][j] += 1;
                            cov[0][i][j] += ((double) sequence[n].entry[t].ddata[i] - mu[i]) *
                                    ((double) sequence[n].entry[t].ddata[j] - mu[j]);
                        }
                for (int i = 0; i < rdim; i++)
                    for (int j = 0; j <= i; j++)
                        if (!is_missing(sequence[n].entry[t].rdata[i]) &&
                                !is_missing(sequence[n].entry[t].rdata[j])) {
                            num_present_r[i][j] += 1;
                            cov[1][i][j] += (sequence[n].entry[t].rdata[i] - mu[ddim + i]) *
                                    (sequence[n].entry[t].rdata[j] - mu[ddim + j]);
                        }
            }

        if (ddim > 0) {
            for (int i = 0; i < ddim; i++)
                for (int j = 0; j <= i; j++) {
                    cov[0][i][j] /= (double) num_present_d[i][j];
                    cov[0][j][i] = cov[0][i][j];
                }
            for (int i = 0; i < ddim; i++)
                num_present_d[i] = null;
            num_present_d = null;
        }

        if (rdim > 0) {
            for (int i = 0; i < rdim; i++)
                for (int j = 0; j <= i; j++) {
                    cov[1][i][j] /= (double) num_present_r[i][j];
                    cov[1][j][i] = cov[1][j][i];
                }
            for (int i = 0; i < rdim; i++)
                num_present_r[i] = null;
            num_present_r = null;
        }

        mu = null;

        return (cov);
    }

    double[][] persistence() {
        /* Calculating the persistence of values for each component */

        int dim;
        int[][]num_present;
        double[][]pers;

        /* Allocating and initializing arrays */
        dim = sequence[0].entry[0].ddim;
        num_present = new int [dim][];
        for (int i = 0; i < dim; i++) {
            num_present[i] = new int[2]; /* !!! !!! */
            for (int j = 0; j < 2; j++)
                num_present[i][j] = 0;
        }
        pers = new double [dim][];
        for (int i = 0; i < dim; i++) {
            pers[i] = new double[2]; /* !!! !!! */
            for (int j = 0; j < 2; j++)
                pers[i][j] = 0.0;
        }

        /* Calculating counts */
        for (int n = 0; n < num_seqs; n++)
            for (int t = 1; t < sequence[n].seq_length; t++)
                for (int i = 0; i < dim; i++)
                    if (!is_missing(sequence[n].entry[t - 1].ddata[i]) && !is_missing(sequence[n].entry[t].ddata[i])) {
                        if (sequence[n].entry[t - 1].ddata[i] == 0) {
                            num_present[i][0]++;
                            if (sequence[n].entry[t].ddata[i] == 0)
                                pers[i][0] += 1.0;
                        } else if (sequence[n].entry[t - 1].ddata[i] == 1) {
                            num_present[i][1]++;
                            if (sequence[n].entry[t].ddata[i] == 1)
                                pers[i][1] += 1.0;
                        }
                    }

        /* Calculating persistence probabilities */
        for (int i = 0; i < dim; i++)
            for (int j = 0; j < 2 /* !!! !!! */; j++)
                pers[i][j] /= (double) num_present[i][j];

        /* Deallocating counts */
        for (int i = 0; i < dim; i++)
            num_present[i] = null;
        num_present = null;

        return (pers);
    }

    int[][] spell(int outcome) {

        int dim;
        int[][]length_dist;
        int spell_length;

        /* Allocating and initializing the output array */
        dim = sequence[0].entry[0].ddim;

        /* First dimension is the components, second is the length of the spell */
        /* !!! Second component max size is the length of the first sequence !!! */
        length_dist = new int [dim][];
        for (int i = 0; i < dim; i++)
            length_dist[i] = new int[sequence[0].seq_length];

        for (int i = 0; i < dim; i++)
            for (int j = 0; j < sequence[0].seq_length; j++)
                length_dist[i][j] = 0;

        for (int i = 0; i < dim; i++) { /* For each component (station) */
            spell_length = 0;

            for (int j = 0; j < num_seqs; j++) { /* For each sequence */
                for (int k = 0; k < sequence[j].seq_length; k++)
                    /* For each entry in the sequence */
                    if (sequence[j].entry[k].ddata[i] == outcome)
                        /* Spell continues */
                        spell_length++;
                    else if (spell_length > 0) { /* Spell ended */

                        /* Updating the counts */
                        length_dist[i][spell_length - 1]++;

                        /* Resetting the length of the current spell */
                        spell_length = 0;
                    } /* Spell ended */

                /* Checking whether the sequence has ended in a spell */
                if (spell_length > 0)
                    /* Updating the counts */
                    length_dist[i][spell_length - 1]++;

                /* Resetting the length */
                spell_length = 0;

            } /* For each sequence */

        } /* For each component (station) */

        return (length_dist);
    }

    double[][] mutual_information() {
        /* Calculating mutual information for each pair of dimensions */
        double[][][][]counts;
        double[][]MI;
        int dim;

        /* Temporary variable(s) */
        double sum;

        dim = sequence[0].entry[0].ddim;

        /* Allocating the arrays of counts */
        counts = new double [dim][][][];
        for (int i = 0; i < dim; i++) {
            counts[i] = new double [i + 1][][];
            for (int j = 0; j < i + 1; j++) {
                counts[i][j] = new double [2][]; /* !!! !!! */
                for (int k = 0; k < 2 /* !!! */; k++)
                    counts[i][j][k] = new double[2]; /* !!! !!! */
            }
        }

        /* Allocating the MI matrix */
        MI = new double [dim][];
        for (int i = 0; i < dim; i++)
            MI[i] = new double[dim];

        /* Initializing the counts */
        for (int i = 0; i < dim; i++)
            for (int j = 0; j < i + 1; j++)
                for (int k = 0; k < 2 /* !!! !!! */; k++)
                    for (int l = 0; l < 2 /* !!! !!! */; l++)
                        counts[i][j][k][l] = 0.0;

        /* Calculating the counts */
        for (int k = 0; k < num_seqs; k++)
            for (int l = 0; l < sequence[k].seq_length; l++)
                for (int i = 0; i < dim; i++)
                    for (int j = 0; j < i + 1; j++)
                        if (!is_missing(sequence[k].entry[l].ddata[i]) &&
                                !is_missing(sequence[k].entry[l].ddata[j]))
                            counts[i][j][sequence[k].entry[l].ddata[i]][sequence[k].entry[l].ddata[j]] += 1.0;

        /* Calculating mutual information */
        for (int i = 0; i < dim; i++)
            for (int j = 0; j < i + 1; j++) {
                MI[i][j] = 0.0;

                /* !!! Assuming binary data !!! */
                sum = 0.0;
                for (int k = 0; k < 2; k++)
                    for (int l = 0; l < 2; l++)
                        sum += counts[i][j][k][l];

                if (abs(sum) > COMP_THRESHOLD) {
                    /* Making probabilities out of the counts */
                    for (int k = 0; k < 2; k++)
                        for (int l = 0; l < 2; l++)
                            counts[i][j][k][l] /= sum;

                    /* Calculating MI */
                    MI[i][j] += xlogx(counts[i][j][0][0]);
                    MI[i][j] += xlogx(counts[i][j][0][1]);
                    MI[i][j] += xlogx(counts[i][j][1][0]);
                    MI[i][j] += xlogx(counts[i][j][1][1]);
                    MI[i][j] -= xlogx(counts[i][j][0][0] + counts[i][j][0][1]);
                    MI[i][j] -= xlogx(counts[i][j][1][0] + counts[i][j][1][1]);
                    MI[i][j] -= xlogx(counts[i][j][0][0] + counts[i][j][1][0]);
                    MI[i][j] -= xlogx(counts[i][j][0][1] + counts[i][j][1][1]);
                }
                MI[j][i] = MI[i][j];
            }

        /* Deallocating the array of counts */
        for (int i = 0; i < dim; i++) {
            for (int j = 0; j < i + 1; j++) {
                for (int k = 0; k < 2 /* !!! */; k++)
                    counts[i][j][k] = null;
                counts[i][j] = null;
            }
            counts[i] = null;
        }
        counts = null;

        return (MI);
    }

}