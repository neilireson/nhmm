package uk.ac.shef.wit.nhmm;

import java.io.File;
import java.io.IOException;
import java.io.PrintStream;

import static uk.ac.shef.wit.nhmm.Constants.SIM_VERBOSE;

public class Sequence {

    int seq_length;
    DataPoint[] entry;
    int num_missing_discrete_entries;
    int[][] missing_discrete;
    private int num_total_entries; // Number of initialized dimensions

    Sequence(int length) {
        seq_length = length;
        entry = new DataPoint[seq_length];
        /* Initializing entries */
        for (int t = 0; t < seq_length; t++)
            entry[t] = null;

        num_total_entries = 0;
        num_missing_discrete_entries = 0;
        missing_discrete = null;
    }

    void close() {

        for (int i = 0; i < seq_length; i++)
            if (entry[i] != null)
                entry[i] = null;

        if (num_missing_discrete_entries > 0) {
            for (int i = 0; i < num_missing_discrete_entries; i++)
                missing_discrete[i] = null;
            missing_discrete = null;
        }

        entry = null;
    }

    void WriteToFile(PrintStream out) {
        for (int i = 0; i < seq_length; i++) {
            if (entry[i] != null)
                entry[i].WriteToFile(out);
            out.format("\n");
        }
    }

    int total_entries() {
        if (num_total_entries == 0)
            for (int t = 0; t < seq_length; t++)
                if (entry[t] != null)
                    num_total_entries += entry[t].total_entries();

        return (num_total_entries);
    }

    double weighted_total_entries(double[] weight) {
        double total = 0.0;

        for (int t = 0; t < seq_length; t++)
            if (entry[t] != null)
                total += weight[t] * (double) (entry[t].total_entries());

        return (total);
    }

    void CatalogueMissingEntries() {

        int[][] temp;

        /* Allocating dummy array to store missing entries */
        temp = new int[seq_length * entry[0].ddim][];

        if (num_missing_discrete_entries > 0) { /* Resetting the list of missing variables */
            for (int i = 0; i < num_missing_discrete_entries; i++)
                missing_discrete[i] = null;
            missing_discrete = null;

            num_missing_discrete_entries = 0;
        } /* Resetting the list of missing variables */

        /* Counting the number of missing entries */
        for (int t = 0; t < seq_length; t++)
            for (int i = 0; i < entry[t].ddim;
                 i++)
                if (Data.is_missing(entry[t].ddata[i])) {
                    temp[num_missing_discrete_entries] = new int[2];
                    temp[num_missing_discrete_entries][0] = t;
                    temp[num_missing_discrete_entries][1] = i;
                    num_missing_discrete_entries++;

                }

        if (num_missing_discrete_entries > 0) {
            missing_discrete = new int[num_missing_discrete_entries][];
            for (int i = 0; i < num_missing_discrete_entries; i++)
                missing_discrete[i] = temp[i];
        }

        temp = null;
    }

    void MissingEntryIndicators() {
        /* Creating indicators for missing entries */

        for (int t = 0; t < seq_length; t++)
            entry[t].MissingEntryIndicators();
    }

    void ReadData(File input, DataPoint datum)
            throws IOException {

        for (int i = 0; i < seq_length; i++) {
            /* Initializing and reading in the observation i */
            if (entry[i] != null)
                entry[i] = null;
            entry[i] = new DataPoint(datum.ddim, datum.rdim);
            entry[i].ReadData(input);
        }
    }

    void Simulate(Distribution[]dist, int[]states, DataPoint datum) {
        /* Simulating a sequence of data according to distribution dist and already generated hidden states */

        for (int t = 0; t < seq_length; t++) {
            /* Initializing observation t */
            if (entry[t]!=null)
                entry[t] = null;
            entry[t] = new DataPoint(datum.ddim, datum.rdim);

            /* Simulating observation t */
            if (t == 0)
                dist[states[t]].Simulate(entry[t], null);
            else
                dist[states[t]].Simulate(entry[t], entry[t - 1]);
            if (SIM_VERBOSE) {
                System.out.format(".");
            }
        }
    }

}
