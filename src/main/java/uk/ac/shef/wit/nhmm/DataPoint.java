package uk.ac.shef.wit.nhmm;

import java.io.DataInputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.PrintStream;
import java.util.Scanner;

import static uk.ac.shef.wit.nhmm.Data.is_missing;

public class DataPoint {
    public int ddim;       // Dimensionality of the discrete-valued variables
    public int rdim;       // Dimensionality of the real-valued variables
    public int[] ddata;     // Discrete-valued data vector
    public double[] rdata;   // Real-valued data vector
    /* For simulation */
    boolean[] ddata_ind; // Indicator of whether the datum should be missing
    boolean[] rdata_ind; // Indicator of whether the datum should be missing

    DataPoint(int dim1, int dim2) {
        /* !!! No safety checking !!! */
        ddim = dim1;
        rdim = dim2;

        /* Allocating data vectors */
        if (ddim > 0)
            ddata = new int[ddim];
        else
            ddata = null;

        if (rdim > 0)
            rdata = new double[rdim];
        else
            rdata = null;

        ddata_ind = null;
        rdata_ind = null;
    }

    void close() {
        /* Deallocating data structures */
        if (ddata != null)
            ddata = null;

        if (rdata != null)
            rdata = null;

        if (ddata_ind != null)
            ddata_ind = null;

        if (rdata_ind != null)
            rdata_ind = null;
    }

    void WriteToFile(PrintStream out) {
        /* Writing the entry into a file */

        for (int i = 0; i < ddim; i++)
            if (is_missing(ddata[i]))
                out.format("NaN ");
            else
                out.format("%d ", ddata[i]);

        for (int i = 0; i < rdim; i++)
            if (is_missing(rdata[i]))
                out.format("NaN ");
            else
                out.format("%.12f ", rdata[i]);
    }

    int total_entries() {
        /* Calculating the total number of initialized components */
        int num_total_entries = 0;

        for (int i = 0; i < ddim; i++)
            if (!is_missing(ddata[i]))
                num_total_entries++;

        for (int i = 0; i < rdim; i++)
            if (!is_missing(rdata[i]))
                num_total_entries++;

        return (num_total_entries);
    }

    void MissingEntryIndicators() {
        /* Creating indicators for missing entries */

        if (ddata_ind == null)
            ddata_ind = new boolean[ddim];

        for (int i = 0; i < ddim; i++)
            if (is_missing(ddata[i]))
                ddata_ind[i] = true;
            else
                ddata_ind[i] = false;

        if (rdata_ind == null)
            rdata_ind = new boolean[rdim];

        for (int i = 0; i < rdim; i++)
            if (is_missing(rdata[i]))
                rdata_ind[i] = true;
            else
                rdata_ind[i] = false;
    }

    void ReadData(File input) throws IOException {

        Scanner scanner = new Scanner(input);
        for (int i = 0; i < ddim; i++)
            ddata[i] = scanner.nextInt();

        for (int i = 0; i < rdim; i++)
            rdata[i] = scanner.nextDouble();
    }

}
