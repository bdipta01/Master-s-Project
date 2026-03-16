#include<stdio.h>
#include<stdlib.h>
#include"conv_layer.h"

void conv_layer_create(Conv_Layer* layer, int input_size, int filter_size){
    layer->input_size = input_size;
    layer->filter_size = filter_size;
    layer->output_size = input_size - filter_size + 1;
    
    for(int i=0; i<filter_size; i++){
        for(int j=0; j<filter_size; j++){
            layer->filter[i][j] = 0.1f;
        }
    }
}

void conv_layer_forward(Conv_Layer* layer, float input[INPUT_SIZE][INPUT_SIZE], float output[OUTPUT_SIZE][OUTPUT_SIZE]){
    for(int i=0; i<layer->output_size; i++){
        for(int j=0; j<layer->output_size; j++){
            float sum = 0.0f;
            for(int m=0; m<layer->filter_size; m++){
                for(int n=0; n<layer->filter_size; n++){
                    sum += input[i+m][j+n] * layer->filter[m][n];
                }
            }
            output[i][j] = sum;
        }
    }
}
