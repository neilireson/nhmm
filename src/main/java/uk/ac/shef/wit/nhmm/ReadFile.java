package uk.ac.shef.wit.nhmm;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.PushbackInputStream;

import static uk.ac.shef.wit.nhmm.Constants.COMMENT_SYMBOL;

public class ReadFile {

    int line_number;
    boolean EOF_TRUE;
    int em_verbose = 0;   // Flag whether to display verbose information with EM
    int short_dist_display = 1; // Flag for not writing the params with dimension indices
    boolean bare_display = false;       // Flag for outputting parameters only

    private final PushbackInputStream in;

    private ReadFile() {
        throw new UnsupportedOperationException();
    }

    public ReadFile(File file) throws FileNotFoundException {
        in = new PushbackInputStream(new FileInputStream(file));
    }

    int read_long() throws IOException {
        char[] output = new char[100];

        int pos = 0;

        eat_blanks();
        if (!EOF_TRUE)
            while (is_number_char(output[pos] = (char) in.read()))
                if (pos < 100 - 1)
                    pos++;
                else
                    System.err.format("Number is too int on line %d\n", line_number);

        in.unread(output[pos]);
        output[pos] = '\0';

        String value = String.valueOf(output, 0, pos);
        try {
            double temp = Double.parseDouble(value);
            if (Data.is_missing(temp))
                return Data.missing_value((int) 0);
            else
                return (int) Double.doubleToLongBits(temp);
        } catch (Exception e) {
            throw new RuntimeException(String.format("Line %d: problem converting input string %s into int\n", line_number, value), e);
        }
    }


    double read_double() throws IOException {
        double output_double;
        char[] output = new char[100];

        int pos = 0;

        eat_blanks();
        if (!EOF_TRUE)
            while (is_number_char(output[pos] = (char) in.read()))
                if (pos < 100 - 1)
                    pos++;
                else
                    System.err.format("String is too int on line %d\n", line_number);

        if (!EOF_TRUE)
            in.unread(output[pos]);

        output[pos] = '\0';

        String value = String.valueOf(output, 0, pos);
        try {
            return Double.parseDouble(value);
        } catch (Exception e) {
            throw new RuntimeException(String.format("Line %d: problem converting input string %s into double\n", line_number, value), e);
        }
    }


    boolean is_number_char(char c) {
        return (Character.isDigit(c) || c == 'e' || c == 'E' || c == '.' || c == '+' || c == '-'
                || c == 'N' || c == 'n' || c == 'a' || c == 'A');
    }

    String read_word() throws IOException {
        int totalChars = 256;
        char[] output = new char[totalChars];
        int pos = 0;

        eat_blanks();
        if (!EOF_TRUE) {
            while (is_word_char(output[pos] = (char) in.read()))
                if (pos < totalChars - 1)
                    pos++;
                else
                    System.err.format("String is too long on line %d\n", line_number);

            in.unread(output[pos]);
            output[pos] = '\0';
        }

        return String.valueOf(output,0,pos);
    }

    boolean is_word_char(char c) {
        return (Character.isLetterOrDigit(c) || c == '_' || c == '/' || c == 92 || c == ':' || c == '.' || c == '-');
    }

    void eat_blanks() throws IOException {
        int c;

        do {
            c = in.read();
            if (c == -1) {
                EOF_TRUE = true;
                return;
            } else {
                switch (c) {
                    case '\n':
                        line_number++;
                        break;
                    //	  case EOF:
                    //EOF_TRUE=1;
                    //break;
                    case COMMENT_SYMBOL:
                        eat_line();
                        c = ' ';
                        break;
                    default:
                        ;
                }
            }
        }
        while (Character.isSpaceChar(c));
        in.unread(c);

    }

    void eat_line() throws IOException {

        int c;
        while ((c = in.read()) != '\n')
            if (c == -1) {
                EOF_TRUE = true;
                return;
            }

        line_number++;
    }
}