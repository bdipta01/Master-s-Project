#include<stdio.h>
#include<stdlib.h>
#include"flatten_layer.h"
#include "uti.h"

void flatten_layer_create(Flatten_Layer* flatten_layer, int input_size, int filter_size, int pool_size){

    flatten_layer->input_size = input_size;
    flatten_layer->filter_size = filter_size;
    flatten_layer->pool_size = pool_size;
    flatten_layer->output_size = input_size - filter_size + 1;
    flatten_layer->pool_output_size = flatten_layer->output_size / flatten_layer->pool_size;
    flatten_layer->conv_size = flatten_layer->pool_output_size * flatten_layer->pool_output_size;

    for(int i=0; i<flatten_layer->conv_size; i++){
        flatten_layer->flat_out[i] = 0.0f;
        flatten_layer->b_flat[i] = 0.0f;
    }

    for(int i=0; i<flatten_layer->pool_output_size; i++){
        for(int j=0; j<flatten_layer->pool_output_size; j++){
            flatten_layer->b_pool[i][j] = 0.0f;
        }
    }
}

void flatten(Flatten_Layer* flatten_layer, float input[POOL_OUTPUT_SIZE][POOL_OUTPUT_SIZE]){
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