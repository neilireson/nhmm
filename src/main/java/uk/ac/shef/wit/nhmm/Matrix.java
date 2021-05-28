package uk.ac.shef.wit.nhmm;

import static java.lang.Math.abs;
import static java.lang.Math.sqrt;

public class Matrix {
    /* Constants */
    public static final double MATRIX_TINY = 1E-10;      // Smallest value allowed on the diagonal of the LU-decomposition

    /* Variables */
//    boolean SINGULAR_FAIL;

    static double find_det(double[][] matrix, int dim) {
        /* Calculating the determinant of a matrix */
        double[][] lu;
        double output = 1.0;

        lu = new double[dim][];
        for (int i = 0; i < dim; i++)
            lu[i] = new double[dim];

        if (!ludcmp(matrix, dim, null, null, lu))
            /* Matrix is singualr or close to it */
            return (0.0);

        for (int i = 0; i < dim; i++)
            output *= lu[i][i];

        /* Deallocating the decomposition */
        for (int i = 0; i < dim; i++)
            lu[i] = null;
        lu = null;

        return (output);
    }

    static boolean find_inv(double[][] a, int dim, double[][] output) {
        double[][] lu;                 // LU decomposition of the matrix
        double[] col;
        double[] temp_col;

        lu = new double[dim][];
        for (int i = 0; i < dim; i++)
            lu[i] = new double[dim];

        if (!ludcmp(a, dim, null, null, lu))
            /* Problem with decomposition */
            return false;

        /* Initializing column vector */
        col = new double[dim];
        for (int i = 0; i < dim; i++)
            col[i] = 0.0;

        temp_col = new double[dim];

        for (int j = 0; j < dim; j++) { /* Finding inverse of a column */
            col[j] = 1.0;
            if (!lubksb(lu, dim, null, col, temp_col))
                /* Problem with back-substitution */
                return false;

            /* Copying the values of the inverse column */
            for (int i = 0; i < dim; i++)
                output[i][j] = temp_col[i];

            col[j] = 0.0;
        } /* Finding inverse of a column */

        temp_col = null;
        col = null;

        for (int i = 0; i < dim; i++)
            lu[i] = null;
        lu = null;

        return true;
    }

    static boolean ludcmp(double[][] a, int dim, int[] perm, double[] flip_perm, double[][] output) {
  /* LU decomposition using Crout's method as described on pp. 46-47
     in Numerical Recipes in C, 2nd edition, Cambridge University Press,
     1992
  */

        /* copying the values of the original matrix into the new one */
        for (int i = 0; i < dim; i++)
            for (int j = 0; j < dim; j++)
                output[i][j] = a[i][j];
  
  /* The output for both lower-triangular and upper-triangular matrices
     is stored in the same matrix. */

        /* Coefficients are calculated column by column */
        for (int j = 0; j < dim; j++) { /* For each column */
      
      /* Handling of the rows is broken down into three cases:
	 upper triangular part, diagonal, and lower triangular part. */

            /* Case 1 -- row<column (upper-triangle)) */
            for (int i = 0; i < j; i++)
                /* Calculating entries of the upper triangular matrix for row j */
                for (int k = 0; k < i; k++)
                    output[i][j] -= (output[i][k] * output[k][j]);
      
      /* Both diagonal and lower triangular entries requires same initial
	 calculations.  Lower triangular entries also require rescaling by the
	 appropriate diagonal element. */

            /* Calculating unscaled values */
            for (int i = j; i < dim; i++)
                for (int k = 0; k < j; k++)
                    output[i][j] -= (output[i][k] * output[k][j]);

            /* !!! No pivoting in this version !!! */
            /* Pivoting should be inserted here */

            /* Dividing the lower triangular part by the diagonal entry */
            if (abs(output[j][j]) < MATRIX_TINY) {
                // System.err.format( "Need pivoting in LU decomposition!\n" );
                return false;
            }

            for (int i = j + 1; i < dim; i++)
                output[i][j] /= output[j][j];

        } /* For each column */

        return true;
    }

    static boolean lubksb(double[][] a, int dim, int[] perm, double[] b, double[] output) {
        /* LU back-substitution procedure */

//        SINGULAR_FAIL = false;

        /* Forward substitution first */
        for (int i = 0; i < dim; i++) {
            output[i] = b[i];
            for (int j = 0; j < i; j++)
                output[i] -= (a[i][j] * output[j]);
        }

        /* Back-substitution */
        for (int i = dim - 1; i >= 0; i--) {
            for (int j = i + 1; j < dim; j++)
                output[i] -= (a[i][j] * output[j]);

            if (abs(a[i][i]) < MATRIX_TINY) {
                //  System.err.format( "Matrix is too close to being singular.  May need pivoting.\n" );
//                SINGULAR_FAIL = true;
                return false;
            }

            output[i] /= a[i][i];
        }
        return true;
    }

    void cholesky(double[][] a, int dim, double[][] output) {
  /* Find Cholesky decomposition C of a positive definite matrix A.
     C is a upper-triangular matrix (in agreement with Matlab) such that 
     C'*C=A. 
  */


        for (int i = 0; i < dim; i++) {
            output[i][i] = 0.0;
            for (int k = 0; k < i; k++)
                output[i][i] += output[k][i] * output[k][i];
            output[i][i] = sqrt(a[i][i] - output[i][i]);

            for (int j = i + 1; j < dim; j++) {
                output[i][j] = 0.0;
                output[j][i] = 0.0;
                for (int k = 0; k < i; k++)
                    output[i][j] += output[k][j] * output[k][i];
                output[i][j] = (a[i][j] - output[i][j]) / output[i][i];
            }
        }
    }
}