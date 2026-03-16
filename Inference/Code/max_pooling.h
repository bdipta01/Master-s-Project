#include "config.h"

typedef struct{
    int input_size;
    int filter_size;
    int output_size;
    int pool_size;
    int stride_size;
    int pool_output_size;    
}Pool_Layer;

void pool_layer_create(Pool_Layer* layer,int input_size, int filter_size, int pool_size, int stride_size);
void max_pooling_forward(Pool_Layer* layer, float input[OUTPUT_SIZE][OUTPUT_SIZE], float output[POOL_OUTPUT_SIZE][POOL_OUTPUT_SIZE]);