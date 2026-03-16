#define MAX_SAMPLES 2000
#define MAX_INPUTS 256
#define MAX_OUTPUTS 10
#define MAX_LINE_LEN 2048
#define MAX_ROWS 16
#define MAX_COLS 16

typedef struct
{
    float in[MAX_SAMPLES][MAX_ROWS][MAX_COLS];   //3D array for input
    float tg[MAX_SAMPLES][MAX_OUTPUTS];   //2D array for target
    int nips;      //Number of inputs
    int nops;      //Number of outputs
    int rows;      //Number of rows
}Data;

void data_shuffle(Data *d);