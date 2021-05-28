package uk.ac.shef.wit.nhmm;

import java.io.PrintStream;

public class StateData {
    public StateSequence[] sequence;
    public int num_seqs;

 StateData(int ns) {

        num_seqs = ns;
        sequence = new StateSequence [num_seqs];
        for (int i = 0; i < num_seqs; i++)
            sequence[i] = null;

    }

    void close() {

        for (int i = 0; i < num_seqs; i++)
            if (sequence[i] != null)
                sequence[i] = null;
        sequence = null;
    }

    void  WriteToFile(PrintStream out) {
        for (int i = 0; i < num_seqs; i++) {
            sequence[i].WriteToFile(out);
            /* Each set of sequences is separated from the next by a line */
            out.format("\n");
        }
    }
}
