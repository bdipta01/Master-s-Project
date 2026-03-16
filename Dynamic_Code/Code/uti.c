#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "uti.h"

int count_line(FILE *const file){          //Count number of lines in a given file
    int ch = EOF;
    int lines = 0;
    int prev = '\n';

    while((ch = getc(file)) != EOF){
        if(ch == '\n'){
            lines++;
        }
        prev = ch;
    }

    if(prev!='\n'){
        lines++;
    }
    rewind(file);  //moves the file pointer back to the beginning
    return lines;
}

char *mAllocCh(FILE * const file){  //allocating memory for character of each line
    int ch = EOF;
    int length = 0;
    int size = 128;
    char *line = (char *)malloc((size)*sizeof(char));

    if(!line){
        fprintf(stderr, "Memory allocation failed for line buffer. \n");
        exit(EXIT_FAILURE);
    }

    while((ch = getc(file)) != '\n' && ch != EOF){
        line[length++] = ch;
        if(length +1 == size){
            char *temp = (char *)realloc((line),(size *= 2)*sizeof(char));
            if(!temp){
                free(line);
                fprintf(stderr, "Memory reallocation failed. \n");
                exit(EXIT_FAILURE);
            }
            line = temp;
        }
    }

    line[length] = '\0';
    return line;
}

float ***matrix3d(const int rows, const int col){
    int input_row = sqrt(col);
    int input_col = sqrt(col);
    float ***mat3 = (float ***)malloc((rows) * sizeof(float **));
    if(!mat3){
        fprintf(stderr, "Memory allocation failed for matrix3d rows. \n");
        exit(EXIT_FAILURE);
    }

    for(int i=0; i<rows; i++){
        mat3[i] = (float **)malloc((input_row) * sizeof(float *));
        if(!mat3[i]){
            fprintf(stderr, "Memory allocation failed for matrix3d input row %d. \n", i);
            for(int k=0; k<i; k++){
                free(mat3[k]);
            }
            free(mat3);
            exit(EXIT_FAILURE);
        }
        for(int j=0; j<input_row; j++){
            mat3[i][j] = (float *)malloc((input_col) * sizeof(float));
            if(!mat3[i][j]){
                fprintf(stderr, "Memory allocation failed for matrix3d input col %d %d. \n", i, j);
                for(int l=0; l<j; l++){
                    free(mat3[i][l]);
                }
                for(int m=0; m<j; m++){
                    free(mat3[m]);
                }
                free(mat3);
                exit(EXIT_FAILURE);
            }
        }
    }
    return mat3;
}

float **matrix2d(const int rows, const int cols){  //allocating memory for matrix
    float **mat = (float **)malloc((rows) * sizeof(float *));
    if(!mat){
        fprintf(stderr, "Memory allocation failed for matrix2d rows. \n");
        exit(EXIT_FAILURE);
    }

    for(int i=0; i<rows;i++){
        mat[i] = (float *)malloc((cols) * sizeof(float));
        if(!mat[i]){
            fprintf(stderr, "Memory allocation failed for matrix row %d. \n", i);
            for(int j=0; j<i; j++){
                free(mat[j]);
            }
            free(mat);
            exit(EXIT_FAILURE);
        }
    }
    return mat;
}

Data ndata(const int nips, const int nops, const int rows){  //create data structure for holding training and testing data
    const Data data = {
        matrix3d(rows, nips),
        matrix2d(rows, nops),
        nips,
        nops,
        rows
    };

    return data;  //return the structure
}

void parse(const Data data, char *line, const int row){  //extract numerical value from a line of text
    char *token;
    char *rest = line;
    const int cols = data.nips + data.nops;

    for(int col=0; col<cols; col++){
        token = strtok_r(col == 0? line : NULL, " ", &rest);  //extract value from a string

        if(!token){   //Handle error if value is NULL
            fprintf(stderr, "Error parsing line: %s\n", line);
            exit(EXIT_FAILURE);
        }

        float val = atof(token);  //convert to float

        if(col < data.nips){
            int num_row_col = sqrt(data.nips);
            int input_row = col / num_row_col;
            int input_col = col % num_row_col;
            data.in[row][input_row][input_col] = val;  //store as an input feature
        }
        else{
            data.tg[row][col - data.nips] = val;  //store as a target label
        }
    }
}

void data_free(const Data d){  //free the allocated memory
    for(int row=0; row<d.rows; row++){
        free(d.in[row]);
        free(d.tg[row]);
    }

    free(d.in);
    free(d.tg);
}

void data_shuffle(const Data d){
    for(int i = d.rows - 1; i>0; i--){
        int j = rand() % (i+1);

        float *temp_tg = d.tg[i];
        d.tg[i] = d.tg[j];
        d.tg[j] = temp_tg;

        float **temp_in = d.in[i];
        d.in[i] = d.in[j];
        d.in[j] = temp_in;
    }
}

Data build(const char *path, const int nips, const int nops){
    FILE * file = fopen(path,"r");    //open the file in read mode
    if(!file){
        fprintf(stderr, "File Error: %s\n", path);
        exit(EXIT_FAILURE);
    }

    const int rows = count_line(file);     //line counts(number of rows) in the file
    Data data = ndata(nips, nops, rows);   //making user-defined data structure holding two 3D array for input feature, three variable
    for(int row=0; row<rows; row++){
        char *line = mAllocCh(file);    //reading a character of each line
        parse(data, line, row);  //extract numerical value from a line of text
        free(line);
    }

    fclose(file);
    return data;
}

//following utility section is for conv_layer, pooling_layer, flatten_layer

float **allocate_2d_matrix(int size){
    float **matrix_2d = (float **)malloc(size * sizeof(float *));
    for(int i=0; i<size; i++){
        matrix_2d[i] = (float *)malloc(size * sizeof(float));
    }
    return matrix_2d;
}

void free_2d_matrix(float **matrix, int size){
    for(int i=0; i<size; i++){
        free(matrix[i]);
    }
    free(matrix);
}

int ***allocate_3d_matrix(int size){
    int*** matrix3d = (int ***)malloc(size * sizeof(int **));
    for(int i=0; i<size; i++){
        matrix3d[i] = (int **)malloc(size * sizeof(int *));
        for(int j=0; j<size; j++){
            matrix3d[i][j] = (int *)malloc(2 * sizeof(int));
        }
    }
    return matrix3d;
}

void free_3d_matrix(int ***matrix, int size){
    for(int i=0; i<size; i++){
        for(int j=0; j<size; j++){
            free(matrix[i][j]);
        }
        free(matrix[i]);
    }
    free(matrix);
}