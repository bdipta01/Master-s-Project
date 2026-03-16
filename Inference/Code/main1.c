#include <stdio.h>
#include <stdlib.h>
#include<math.h>
#include "uti.h"
#include "conv_layer.h"
#include "max_pooling.h"
#include "flatten_layer.h"
#include "dense_layer.h"
#include "data_array.h"
#include "test_data_8.h"
#include "trained_model.h"

int main(){
    const int conv_input = sqrt(256);

    static Data test_data;
    test_data.rows = TEST_NUM_SAMPLES;
    test_data.nips = 256;
    test_data.nops = 10;
    for(int i=0; i<TEST_NUM_SAMPLES; i++){
        for(int j=0; j<16; j++){
            for(int k=0; k<16; k++){
                test_data.in[i][j][k] = test_inputs[i][j][k]; //test_inputs is a train data set. I made it array header file. So that I do not need I/O file system.
            }
        }
    }
    for(int i=0; i<TEST_NUM_SAMPLES; i++){
        for(int j=0; j<10; j++){
                test_data.tg[i][j] = test_outputs[i][j]; ////test_outputs is a train data set. I made it array header file. So that I do not need I/O file system.
        }
    }
    
    Conv_Layer conv_layer;
    conv_layer_create(&conv_layer, conv_input, 3);

    Pool_Layer pool_layer;
    pool_layer_create(&pool_layer, conv_input, 3, 2, 2);

    Flatten_Layer flatten_layer;
    flatten_layer_create(&flatten_layer, conv_input, 3, 2);

    Dense_Layer d_layer;
    dLayer_Load(&d_layer, trained_mode_data); 

    for(int i=0; i<conv_layer.filter_size; i++){
        for(int j=0; j<conv_layer.filter_size; j++){
            conv_layer.filter[i][j] = conv_filter[i][j];
        }
    }

    static float buffer1[INPUT_SIZE][INPUT_SIZE];

    static float buffer2[POOL_OUTPUT_SIZE][POOL_OUTPUT_SIZE];

    static float buffer3[FLAT_SIZE];

    for(int row=0; row<test_data.rows; row++){
        

        for(int i=0; i<conv_layer.input_size; i++){
            for(int j=0; j<conv_layer.input_size; j++){
                buffer1[i][j] = test_data.in[row][i][j] / 255.0f;
            }
        }

        conv_layer_forward(&conv_layer, buffer1, (float(*)[OUTPUT_SIZE])buffer1);
        max_pooling_forward(&pool_layer, (float(*)[OUTPUT_SIZE])buffer1, buffer2);
        flatten(&flatten_layer, buffer2, buffer3);

        const float * const pd = dLayer_Predict(&d_layer, buffer3);
        dLayer_Print(pd, 10);
    }
    return 0;
}