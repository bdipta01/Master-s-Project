#include "config.h"

typedef struct{
    int input_size;
    int filter_size;
    int pool_size;
    int output_size;
    int pool_output_size;
    int conv_size;

    float flat_out[FLAT_SIZE];
    float b_pool[POOL_OUTPUT_SIZE][POOL_OUTPUT_SIZE];
    float b_flat[FLAT_SIZE];

}Flatten_Layer;

void flatten_layer_create(Flatten_Layer* flatten_layer, int input_size, int filter_size, int pool_size);
void flatten(Flatten_Layer* flatten_layer, float input[POOL_OUTPUT_SIZE][POOL_OUTPUT_SIZE]);
void unflatten(Flatten_Layer* flatten_layer);