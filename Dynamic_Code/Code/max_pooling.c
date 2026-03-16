#include<stdio.h>
#include<stdlib.h>
#include"max_pooling.h"
#include "uti.h"

Pool_Layer* pool_layer_create(int input_size, int filter_size, int pool_size, int stride_size){
    Pool_Layer* pool_layer = (Pool_Layer*)malloc(sizeof(Pool_Layer));
    pool_layer->input_size = input_size;
    pool_layer->filter_size = filter_size;
    pool_layer->output_size = input_size - filter_size + 1;
    pool_layer->pool_size = pool_size;
    pool_layer->stride_size = stride_size;
    pool_layer->pool_output_size = pool_layer->output_size / pool_layer->pool_size;

    pool_layer->grad_output = allocate_2d_matrix(pool_layer->output_size);
    pool_layer->input = allocate_2d_matrix(pool_layer->input_size);
    pool_layer->pool_out = allocate_2d_matrix((pool_layer->input_size-2)/2);
    pool_layer->b_pool = allocate_2d_matrix((pool_layer->input_size-2)/2);
    pool_layer->max_index = allocate_3d_matrix(pool_layer->pool_output_size);
    return pool_layer;
}

void pool_layer_free(Pool_Layer* pool_layer){
    free_2d_matrix(pool_layer->input, pool_layer->input_size);
    free_2d_matrix(pool_layer->pool_out, ((pool_layer->input_size-2)/2));
    free_3d_matrix(pool_layer->max_index, pool_layer->pool_output_size);
    free_2d_matrix(pool_layer->grad_output, pool_layer->output_size);
    free_2d_matrix(pool_layer->b_pool, ((pool_layer->input_size-2)/2));
    free(pool_layer);
}

void max_pooling_forward(Pool_Layer* pool_layer, float** input){
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

void max_pooling_backward(Pool_Layer* pool_layer, float **input){
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