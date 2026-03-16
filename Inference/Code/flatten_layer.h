#include "config.h"

typedef struct{
    int pool_output_size;
    int flat_size;
}Flatten_Layer;

void flatten_layer_create(Flatten_Layer* flatten_layer, int input_size, int filter_size, int pool_size);
void flatten(Flatten_Layer* flatten_layer, float input[POOL_OUTPUT_SIZE][POOL_OUTPUT_SIZE], float output[FLAT_SIZE]);