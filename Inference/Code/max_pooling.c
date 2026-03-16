#include<stdio.h>
#include<stdlib.h>
#include"max_pooling.h"

void pool_layer_create(Pool_Layer* layer,int input_size, int filter_size, int pool_size, int stride_size){
    layer->input_size = input_size;
    layer->filter_size = filter_size;
    layer->output_size = input_size - filter_size + 1;
    layer->pool_size = pool_size;
    layer->stride_size = stride_size;
    layer->pool_output_size = layer->output_size / layer->pool_size;
}

void max_pooling_forward(Pool_Layer* layer, float input[OUTPUT_SIZE][OUTPUT_SIZE], float output[POOL_OUTPUT_SIZE][POOL_OUTPUT_SIZE]){
    for(int i=0; i<layer->output_size; i+=layer->stride_size){
        for(int j=0; j<layer->output_size; j+=layer->stride_size){
            float max_val = input[i][j];

            for(int m=0; m<layer->pool_size; m++){
                for(int n=0; n<layer->pool_size; n++){
                    float val = input[i+m][j+n];
                    if(val > max_val){
                        max_val = val;
                    }
                }
            }
            output[i/layer->stride_size][j/layer->stride_size] = max_val;
        }
    }
}