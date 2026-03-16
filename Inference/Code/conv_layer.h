#include "config.h"

typedef struct{
    int input_size;
    int filter_size;
    int output_size;
    
    float filter[FILTER_SIZE][FILTER_SIZE];
}Conv_Layer;

void conv_layer_create(Conv_Layer* layer, int input_size, int filter_size);
void conv_layer_forward(Conv_Layer* layer, float input[INPUT_SIZE][INPUT_SIZE], float output[OUTPUT_SIZE][OUTPUT_SIZE]);