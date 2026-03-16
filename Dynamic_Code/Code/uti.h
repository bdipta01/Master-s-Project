typedef struct
{
    float *** in;   //3D array for input
    float ** tg;   //2D array for target
    int nips;      //Number of inputs
    int nops;      //Number of outputs
    int rows;      //Number of rows
}Data;

int count_line(FILE *const file);
char *mAllocCh(FILE * const file);
float ***matrix3d(const int rows, const int col);
float **matrix2d(const int rows, const int cols);
Data ndata(const int nips, const int nops, const int rows);
void parse(const Data data, char *line, const int row);
void data_free(const Data d);
void data_shuffle(const Data d);
Data build(const char *path, const int nips, const int nops);
float **allocate_2d_matrix(int size);
void free_2d_matrix(float **matrix, int size);
int ***allocate_3d_matrix(int size);
void free_3d_matrix(int ***matrix, int size);