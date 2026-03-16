#include<stdio.h>
#include<stdlib.h>
#include"flatten_layer.h"
#include "uti.h"

void flatten_layer_create(Flatten_Layer* flatten_layer, int input_size, int filter_size, int pool_size){

    int output_size = input_size - filter_size + 1;
    
    flatten_layer->pool_output_size = output_size/pool_size;
    flatten_layer->flat_size = flatten_layer->pool_output_size * flatten_layer->pool_output_size;
}

void flatten(Flatten_Layer* flatten_layer, float input[POOL_OUTPUT_SIZE][POOL_OUTPUT_SIZE], float output[FLAT_SIZE]){
    int idx = 0;
    for(int i=0; i<flatten_layer->pool_output_size; i++){
        for(int j=0; j<flatten_layer->pool_output_size; j++){
            output[idx++] = input[i][j];
        }
    }
}