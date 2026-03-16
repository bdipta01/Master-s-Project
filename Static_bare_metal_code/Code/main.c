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

int main(){
    const int nips = 256; //input neuron
    const int nhid = 16;
    const int nops = 10;  //output neuron
    const int conv_input = sqrt(256);

    float rate = 1.0f;  //learning rate
    const float eta = 0.99f;
    const int iterations = 300;

    static Data data;
    data.rows = NUM_SAMPLES;
    data.nips = 256;
    data.nops = 10;
    for(int i=0; i<NUM_SAMPLES; i++){
        for(int j=0; j<16; j++){
            for(int k=0; k<16; k++){
                data.in[i][j][k] = train_inputs[i][j][k];
            }
        }
    }
    for(int i=0; i<NUM_SAMPLES; i++){
        for(int j=0; j<10; j++){
                data.tg[i][j] = train_outputs[i][j];
        }
    }

    static Data test_data;
    test_data.rows = TEST_NUM_SAMPLES;
    test_data.nips = 256;
    test_data.nops = 10;
    for(int i=0; i<TEST_NUM_SAMPLES; i++){
        for(int j=0; j<16; j++){
            for(int k=0; k<16; k++){
                test_data.in[i][j][k] = test_inputs[i][j][k];
            }
        }
    }
    for(int i=0; i<TEST_NUM_SAMPLES; i++){
        for(int j=0; j<10; j++){
                test_data.tg[i][j] = test_outputs[i][j];
        }
    }
    
    Conv_Layer conv_layer;
    conv_layer_create(&conv_layer, conv_input, 3);

    Pool_Layer pool_layer;
    pool_layer_create(&pool_layer, conv_input, 3, 2, 2);

    Flatten_Layer flatten_layer;
    flatten_layer_create(&flatten_layer, conv_input, 3, 2);

    Dense_Layer d_layer;
    dLayer_Build(&d_layer, flatten_layer.conv_size, nhid, nops); 

    for(int i=0; i<conv_layer.filter_size; i++){
        for(int j=0; j<conv_layer.filter_size; j++){
            conv_layer.filter[i][j] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f;
        }
    }


    for(int epoch=1; epoch<=iterations; epoch++){
        data_shuffle(&data);
        float error = 0.0f;
        for(int row=0; row<data.rows; row++){
            

            for(int i=0; i<conv_layer.input_size; i++){
                for(int j=0; j<conv_layer.input_size; j++){
                    conv_layer.input[i][j] = data.in[row][i][j] / 255.0f;
                }
            }

            conv_layer_forward(&conv_layer);
            max_pooling_forward(&pool_layer, conv_layer.conv_layer_out);
            flatten(&flatten_layer, pool_layer.pool_out);

            const float * const in = flatten_layer.flat_out;   
            const float * const tg = data.tg[row];
            error += dLayer_Train(&d_layer, in, tg, rate); 

            for(int i=0; i<flatten_layer.conv_size; i++){
                flatten_layer.b_flat[i] = d_layer.d_in[i];
            }

            unflatten(&flatten_layer);
            max_pooling_backward(&pool_layer, flatten_layer.b_pool);
            conv_layer_backward(&conv_layer, pool_layer.grad_output, rate);
        }

        printf("Iteration: %d Error %.6f :: learning rate %f\n", epoch, (double)error/data.rows, (double)rate);

        rate *= eta;
    }

    //dLayer_save_build(&d_layer);
    float model_data[3+2+MAX_WEIGHT];
    int model_size =dLayer_Save(&d_layer, model_data);  //store the trained weights and biases in mymodel.nn


    Dense_Layer loaded_model;
    dLayer_Load(&loaded_model, model_data);

    Conv_Layer test_conv_layer;
    conv_layer_create(&test_conv_layer, conv_input, 3);

    Pool_Layer test_pool_layer;
    pool_layer_create(&test_pool_layer, conv_input, 3, 2, 2);

    Flatten_Layer test_flatten_layer;
    flatten_layer_create(&test_flatten_layer, conv_input, 3, 2);


    for(int i=0; i<conv_layer.filter_size; i++){
        for(int j=0; j<conv_layer.filter_size; j++){
            test_conv_layer.filter[i][j] = conv_layer.filter[i][j];
        }
    }

    for(int row=0; row<test_data.rows; row++){
        

        for(int i=0; i<test_conv_layer.input_size; i++){
            for(int j=0; j<test_conv_layer.input_size; j++){
                test_conv_layer.input[i][j] = test_data.in[row][i][j] / 255.0f;
            }
        }

        conv_layer_forward(&test_conv_layer);
        max_pooling_forward(&test_pool_layer, test_conv_layer.conv_layer_out);
        flatten(&test_flatten_layer, test_pool_layer.pool_out);

        const float * const in = test_flatten_layer.flat_out; //first test sample
        const float * const pd = dLayer_Predict(&loaded_model, in);
        dLayer_Print(pd, data.nops);
    }
    return 0;
}