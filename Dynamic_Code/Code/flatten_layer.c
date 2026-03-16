#include<stdio.h>
#include<stdlib.h>
#include"flatten_layer.h"
#include "uti.h"

Flatten_Layer* flatten_layer_create(int input_size, int filter_size, int pool_size){
    Flatten_Layer* flatten_layer = (Flatten_Layer *)malloc(sizeof(Flatten_Layer));

    flatten_layer->input_size = input_size;
    flatten_layer->filter_size = filter_size;
    flatten_layer->pool_size = pool_size;
    flatten_layer->output_size = input_size - filter_size + 1;
    flatten_layer->pool_output_size = flatten_layer->output_size / flatten_layer->pool_size;
    flatten_layer->conv_size = flatten_layer->pool_output_size * flatten_layer->pool_output_size;

    flatten_layer->flat_out = (float *)malloc(flatten_layer->conv_size * sizeof(float));
    flatten_layer->b_pool = allocate_2d_matrix(flatten_layer->pool_output_size);
    flatten_layer->b_flat = (float *)malloc(flatten_layer->conv_size * sizeof(float));  

    return flatten_layer;
}

void flatten_layer_free(Flatten_Layer* flatten_layer){
    free(flatten_layer->flat_out);
    free_2d_matrix(flatten_layer->b_pool, ((flatten_layer->input_size-2)/2));
    free(flatten_layer->b_flat);
    free(flatten_layer);
}

void flatten(Flatten_Layer* flatten_layer, float** input){
    for(int i=0; i<flatten_layer->pool_output_size; i++){
        for(int j=0; j<flatten_layer->pool_output_size; j++){
            flatten_layer->flat_out[i*flatten_layer->pool_output_size+j] = input[i][j];
        }
    }
}

void unflatten(Flatten_Layer* flatten_layer){
    for(int i=0; i<flatten_layer->pool_output_size; i++){
        for(int j=0; j<flatten_layer->pool_output_size; j++){
            flatten_layer->b_pool[i][j] = flatten_layer->b_flat[i * flatten_layer->pool_output_size + j];
        }
    }
}