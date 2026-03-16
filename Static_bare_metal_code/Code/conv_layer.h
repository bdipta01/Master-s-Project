#include "config.h"

typedef struct{
    int input_size;
    int filter_size;
    int output_size;
    
    float filter[FILTER_SIZE][FILTER_SIZE];
    float input[INPUT_SIZE][INPUT_SIZE];
    float conv_layer_out[OUTPUT_SIZE][OUTPUT_SIZE];


    float grad_output[OUTPUT_SIZE][OUTPUT_SIZE];
    float grad_filter[FILTER_SIZE][FILTER_SIZE];
}Conv_Layer;

void conv_layer_create(Conv_Layer* layer, int input_size, int filter_size);
void conv_layer_forward(Conv_Layer* conv_layer);
void conv_layer_backward(Conv_Layer* conv_layer, float input_file[OUTPUT_SIZE][OUTPUT_SIZE], float rate);