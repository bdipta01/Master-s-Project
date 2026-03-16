#include<stdio.h>
#include<stdlib.h>
#include"max_pooling.h"
#include "uti.h"

void pool_layer_create(Pool_Layer* pool_layer,int input_size, int filter_size, int pool_size, int stride_size){
    pool_layer->input_size = input_size;
    pool_layer->filter_size = filter_size;
    pool_layer->output_size = input_size - filter_size + 1;
    pool_layer->pool_size = pool_size;
    pool_layer->stride_size = stride_size;
    pool_layer->pool_output_size = pool_layer->output_size / pool_layer->pool_size;

    for(int i=0; i<pool_layer->input_size; i++){
        for(int j=0; j<pool_layer->input_size; j++){
            pool_layer->input[i][j] = 0.0f;
            pool_layer->grad_output[i][j] = 0.0f;
        }
    }

    for(int i=0; i<pool_layer->pool_output_size; i++){
        for(int j=0; j<pool_layer->pool_output_size; j++){
            pool_layer->pool_out[i][j] = 0.0f;
            pool_layer->b_pool[i][j] = 0.0f;
            pool_layer->max_index[i][j][0] = 0;
            pool_layer->max_index[i][j][1] = 0;
        }
    }
}

void max_pooling_forward(Pool_Layer* pool_layer, float input[OUTPUT_SIZE][OUTPUT_SIZE]){
    for(int i=0; i<pool_layer->output_size; i+=pool_layer->stride_size){
        for(int j=0; j<pool_layer->output_size; j+=pool_layer->stride_size){
            float max_val = input[i][j];
            int mi = i, mj = j;

            for(int m=0; m<pool_layer->pool_size; m++){
                for(int n=0; n<pool_layer->pool_size; n++){
                    float val = input[i+m][j+n];
                    if(val > max_val){
                        max_val = val;
                        mi = i + m;
                        mj = j + n;
                    }
                }
            }
            pool_layer->pool_out[i/2][j/2] = max_val;
            pool_layer->max_index[i / 2][j / 2][0] = mi;
            pool_layer->max_index[i / 2][j / 2][1] = mj;
        }
    }
}

void max_pooling_backward(Pool_Layer* pool_layer, float input[POOL_OUTPUT_SIZE][POOL_OUTPUT_SIZE]){
    for(int i=0; i<pool_layer->output_size; i++){
        for(int j=0; j<pool_layer->output_size; j++){
            pool_layer->grad_output[i][j] = 0.0f;
        }
    }

    for(int i=0; i<pool_layer->pool_output_size; i++){
        for(int j=0; j<pool_layer->pool_output_size; j++){
            int mi = pool_layer->max_index[i][j][0];
            int mj = pool_layer->max_index[i][j][1];
            pool_layer->grad_output[mi][mj] = input[i][j];
        }
    }
}