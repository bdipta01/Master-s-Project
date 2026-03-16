#include<stdio.h>
#include<stdlib.h>
#include"conv_layer.h"
#include "uti.h"

void conv_layer_create(Conv_Layer* layer, int input_size, int filter_size){
    layer->input_size = input_size;
    layer->filter_size = filter_size;
    layer->output_size = input_size - filter_size + 1;
    
    for(int i=0; i<filter_size; i++){
        for(int j=0; j<filter_size; j++){
            layer->filter[i][j] = 0.1f;
            layer->grad_filter[i][j] = 0.0f;
        }
    }

    for(int i=0; i<input_size; i++){
        for(int j=0; j<input_size; j++){
            layer->input[i][j] = 0.0f;
        }
    }
}

void conv_layer_forward(Conv_Layer* conv_layer){
    for(int i=0; i<conv_layer->output_size; i++){
        for(int j=0; j<conv_layer->output_size; j++){
            float sum = 0.0f;
            for(int m=0; m<conv_layer->filter_size; m++){
                for(int n=0; n<conv_layer->filter_size; n++){
                    sum += conv_layer->input[i+m][j+n] * conv_layer->filter[m][n];
                }
            }
            conv_layer->conv_layer_out[i][j] = sum;
        }
    }
}

void conv_layer_backward(Conv_Layer* conv_layer, float input_file[OUTPUT_SIZE][OUTPUT_SIZE], float rate){
    for(int m=0; m<conv_layer->filter_size; m++){
        for(int n=0; n<conv_layer->filter_size; n++){
            float grad = 0.0f;
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
