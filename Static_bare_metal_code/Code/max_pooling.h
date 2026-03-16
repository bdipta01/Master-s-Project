#include "config.h"

typedef struct{
    int input_size;
    int filter_size;
    int output_size;
    int pool_size;
    int stride_size;
    int pool_output_size;
    
    float grad_output[OUTPUT_SIZE][OUTPUT_SIZE];
    float input[OUTPUT_SIZE][OUTPUT_SIZE];
    float b_pool[POOL_OUTPUT_SIZE][POOL_OUTPUT_SIZE];
    float pool_out[POOL_OUTPUT_SIZE][POOL_OUTPUT_SIZE];
    int max_index[POOL_OUTPUT_SIZE][POOL_OUTPUT_SIZE][2];
    
}Pool_Layer;

void pool_layer_create(Pool_Layer* pool_layer,int input_size, int filter_size, int pool_size, int stride_size);
void max_pooling_forward(Pool_Layer* pool_layer, float input[OUTPUT_SIZE][OUTPUT_SIZE]);
void max_pooling_backward(Pool_Layer* pool_layer, float input[POOL_OUTPUT_SIZE][POOL_OUTPUT_SIZE]);