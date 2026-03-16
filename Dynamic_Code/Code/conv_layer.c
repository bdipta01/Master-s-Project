#include<stdio.h>
#include<stdlib.h>
#include"conv_layer.h"
#include "uti.h"

Conv_Layer* conv_layer_create(int input_size, int filter_size){
    Conv_Layer* conv_layer = (Conv_Layer*)malloc(sizeof(Conv_Layer));
    conv_layer->input_size = input_size;
    conv_layer->filter_size = filter_size;
    conv_layer->output_size = input_size - filter_size + 1;
    conv_layer->filter = allocate_2d_matrix(conv_layer->filter_size);
    conv_layer->input = allocate_2d_matrix(conv_layer->input_size);
    conv_layer->conv_layer_out = allocate_2d_matrix(conv_layer->output_size);
    conv_layer->grad_output = allocate_2d_matrix(conv_layer->output_size);
    conv_layer->grad_filter = allocate_2d_matrix(conv_layer->filter_size);
    return conv_layer;
}

void conv_layer_free(Conv_Layer* conv_layer){
    free_2d_matrix(conv_layer->input, conv_layer->input_size);
    free_2d_matrix(conv_layer->filter, conv_layer->filter_size);
    free_2d_matrix(conv_layer->conv_layer_out, conv_layer->output_size);
    free_2d_matrix(conv_layer->grad_filter, conv_layer->filter_size); //grad_filter
    free_2d_matrix(conv_layer->grad_output, conv_layer->output_size);
    free(conv_layer);
}

void conv_layer_forward(Conv_Layer* conv_layer){
    for(int i=0; i<conv_layer->output_size; i++){
        for(int j=0; j<conv_layer->output_size; j++){
            float sum = 0.0;
            for(int m=0; m<conv_layer->filter_size; m++){
                for(int n=0; n<conv_layer->filter_size; n++){
                    sum += conv_layer->input[i+m][j+n] * conv_layer->filter[m][n];
                }
            }
            conv_layer->conv_layer_out[i][j] = sum;
        }
    }
}

void conv_layer_backward(Conv_Layer* conv_layer, float **input_file, float rate){
    for(int m=0; m<conv_layer->filter_size; m++){
        for(int n=0; n<conv_layer->filter_size; n++){
            float grad = 0.0;
            for(int i=0; i<conv_layer->output_size; i++){
                for(int j=0; j<conv_layer->output_size; j++){
                    grad += conv_layer->input[i+m][j+n] * input_file[i][j];
                }
            }
            conv_layer->grad_filter[m][n] = grad;
            conv_layer->filter[m][n] -= rate * conv_layer->grad_filter[m][n];
        }
    }
}
