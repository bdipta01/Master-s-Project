#include <stdio.h>
#include <stdlib.h>
#include<math.h>
#include "uti.h"
#include "conv_layer.h"
#include "max_pooling.h"
#include "flatten_layer.h"
#include "dense_layer.h"

int main(){
    const int nips = 256; //input neuron
    const int nhid = 16;
    const int nops = 10;  //output neuron
    const int conv_input = sqrt(256);

    float rate = 1.0f;  //learning rate
    const float eta = 0.99f;
    const int iterations = 25;

    Data data = build("train_data_2.data", nips, nops);
    Data test_data = build("test_data_2.data", nips, nops);
    
    Conv_Layer* conv_layer = conv_layer_create(conv_input, 3);
    Pool_Layer* pool_layer = pool_layer_create(conv_input, 3, 2, 2);
    Flatten_Layer* flatten_layer= flatten_layer_create(conv_input, 3, 2);
    const Dense_Layer d_layer = dLayer_Build(flatten_layer->conv_size, nhid, nops); 

    for(int i=0; i<conv_layer->filter_size; i++){
        for(int j=0; j<conv_layer->filter_size; j++){
            conv_layer->filter[i][j] = ((float)rand() / RAND_MAX - 0.5f) * 2;
        }
    }


    for(int epoch=1; epoch<=iterations; epoch++){
        data_shuffle(data);
        float error = 0.0f;
        for(int row=0; row<data.rows; row++){
            

            for(int i=0; i<conv_layer->input_size; i++){
                for(int j=0; j<conv_layer->input_size; j++){
                    conv_layer->input[i][j] = data.in[row][i][j] / 255.0f;
                }
            }

            conv_layer_forward(conv_layer);
            max_pooling_forward(pool_layer, conv_layer->conv_layer_out);
            flatten(flatten_layer, pool_layer->pool_out);

            const float * const in = flatten_layer->flat_out;   
            const float * const tg = data.tg[row];
            error += dLayer_Train(d_layer, in, tg, rate); 

            for(int i=0; i<flatten_layer->conv_size; i++){
                flatten_layer->b_flat[i] = d_layer.d_in[i];
            }

            unflatten(flatten_layer);
            max_pooling_backward(pool_layer, flatten_layer->b_pool);
            conv_layer_backward(conv_layer, pool_layer->grad_output, rate);
        }

        printf("Iteration: %d Error %.6f :: learning rate %f\n", epoch, (double)error/data.rows, (double)rate);

        rate *= eta;
    }

    dLayer_Save(d_layer, "mymodel.nn");  //store the trained weights and biases in mymodel.nn
    dLayer_Free(d_layer);                 //Releases allocated memory 

    const Dense_Layer my_loaded_model = dLayer_Load("mymodel.nn"); //restore the trained model data from mymodel2.nn

    Conv_Layer* test_conv_layer = conv_layer_create(conv_input, 3);
    Pool_Layer* test_pool_layer = pool_layer_create(conv_input, 3, 2, 2);
    Flatten_Layer* test_flatten_layer= flatten_layer_create(conv_input, 3, 2);

    for(int i=0; i<conv_layer->filter_size; i++){
        for(int j=0; j<conv_layer->filter_size; j++){
            test_conv_layer->filter[i][j] = conv_layer->filter[i][j];
        }
    }

    for(int row=0; row<test_data.rows; row++){
        

        for(int i=0; i<test_conv_layer->input_size; i++){
            for(int j=0; j<test_conv_layer->input_size; j++){
                test_conv_layer->input[i][j] = test_data.in[row][i][j] / 255.0f;
            }
        }

        conv_layer_forward(test_conv_layer);
        max_pooling_forward(test_pool_layer, test_conv_layer->conv_layer_out);
        flatten(test_flatten_layer, test_pool_layer->pool_out);

        const float * const in = test_flatten_layer->flat_out; //first test sample
        const float * const pd = dLayer_Predict(my_loaded_model, in);
        dLayer_Print(pd, data.nops);
    }
    
    dLayer_Free(my_loaded_model);
    data_free(data);
    data_free(test_data);

    conv_layer_free(conv_layer);
    pool_layer_free(pool_layer);
    flatten_layer_free(flatten_layer);

    conv_layer_free(test_conv_layer);
    pool_layer_free(test_pool_layer);
    flatten_layer_free(test_flatten_layer);
    return 0;
}