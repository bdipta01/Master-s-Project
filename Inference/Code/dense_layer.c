#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdarg.h>
#include "dense_layer.h"
#include "conv_layer.h"

static float act(const float a){           //Sigmoid activation function
    return 1.0f / (1.0f + expf (-a));
}

static float pdact(const float a){        //Derivative of Activation function
    return a * (1.0f - a);
}

static float frand(){                     //Random floating point number
    return rand()/(float)RAND_MAX-0.5f;
}

static float pderr(const float a, const float b){       //Derivative of error
    return a-b;
}

static void fprop(Dense_Layer *nn, const float * const in){
    /*Compute Hidden Layer neuron values*/
    for(int i=0; i<nn->nhid; i++){
        float sum = 0.0f;
        for(int j=0; j<nn->nips; j++){
            sum += in[j] * nn->w[i*nn->nips+j];  //input * weight of all connected synopses of hidden layer 1
        }
        nn->h[i] = act(sum + nn->b[0]);
    }

    /*Compute Output Layer neuron values*/
    for(int i=0; i<nn->nops; i++){
        float sum = 0.0f;
        for(int j=0; j<nn->nhid; j++){
            sum += nn->h[j] * nn->x[i * nn->nhid + j];
        }
        nn->o[i] = act(sum + nn->b[1]);
    }
}

static void bprop(Dense_Layer *nn, const float * const in, const float * const tg, float rate){
    for(int i=0; i<nn->nhid; i++){
        float sum = 0.0f;
        for(int j=0; j<nn->nops; j++){
            const float a = pderr(nn->o[j], tg[j]);  //nn.o[j] - tg[j] //output value - expected value
            const float b = pdact(nn->o[j]);         //nn.o[j] * (1.0f - nn.o[j]); //applied activation function on output value

            const float err_signal = a * b;
            sum += err_signal * nn->x[j*nn->nhid+i];        // a*b represents the error signal for output neuron j

            nn->x[j * nn->nhid +i] -= rate * err_signal * nn->h[i];  //rate = learning rate determines how much weight should be adjusted
        } 

        for(int j=0; j<nn->nips; j++){
            nn->w[i * nn->nips + j] -= rate * sum * pdact(nn->h[i]) * in[j];
        }
    }

    for(int j=0; j<nn->nips; j++){
        float grad = 0.0f;
        for(int i=0; i<nn->nhid; i++){
            float out_grad = 0.0f;
            for(int k=0; k<nn->nops; k++){
                float a = pderr(nn->o[k], tg[k]);
                float b = pdact(nn->o[k]);
                float err_signal = a*b;
                out_grad += err_signal * nn->x[k * nn->nhid + i];
            }
            grad += out_grad * pdact(nn->h[i]) * nn->w[i * nn->nips + j];
        }
        nn->d_in[j] = grad;
    }
}


float *dLayer_Predict(Dense_Layer *nn, const float *in){
    fprop(nn, in);
    return nn->o;
}

void dLayer_Build(Dense_Layer* nn, int nips, const int nhid, const int nops){
    nn->nips = nips;
    nn->nhid = nhid;
    nn->nops = nops;
    nn->nb = 2;                                          //number of biases
    nn->nw = nhid * (nips + nops);                       //number of weights
    nn->x = nn->w + nhid * nips;                         //Hidden layer to output layer weights

    for(int i=0; i<nn->nw; i++) nn->w[i] = frand();
    for(int i=0; i<nn->nb; i++) nn->b[i] = frand();

    for(int i=0; i<nn->nhid; i++) nn->h[i] = 0.0f;
    for(int i=0; i<nn->nops; i++) nn->o[i] = 0.0f;
    for(int i=0; i<nn->nips; i++) nn->d_in[i] = 0.0f;
}

int dLayer_Save(Dense_Layer *nn, float *arr){
    int idx = 0;

    arr[idx++] = (float)nn->nips;
    arr[idx++] = (float)nn->nhid;
    arr[idx++] = (float)nn->nops;

    for(int i=0; i< nn->nb; i++){
        arr[idx++] = nn->b[i];
    }
    for(int i=0; i< nn->nw; i++){
        arr[idx++] = nn->w[i];
    }

    return idx;
}

void dLayer_Load(Dense_Layer *nn, const float *arr){
    int idx = 0;

    int nips = (int)arr[idx++];
    int nhid = (int)arr[idx++];
    int nops = (int)arr[idx++];
    /*Load the biases*/
    dLayer_Build(nn, nips, nhid, nops);

    for(int i=0; i<nn->nb; i++){
        nn->b[i] = arr[idx++];
    }

    /*Load the weights*/
    for(int i=0; i<nn->nw; i++){
        nn->w[i] = arr[idx++];
    }
}

float dLayer_Train(Dense_Layer *nn, const float *in, const float *tg, float rate){
    fprop(nn, in);  //Compute hidden and output layer
    bprop(nn, in, tg, rate);  //update the weight and biases using the error

    float error = 0.0f;
    for(int i = 0; i < nn->nops; i++){
        float diff = tg[i] - nn->o[i];
        error += 0.5f * diff * diff;
    }

    return error;
}

void dLayer_Print(const float *arr, int size){
    double max = 0.0f;
    int idx = 0;
    
    for(int i=0; i<size; i++){
        printf("%f ", (double)arr[i]);

        if(arr[i] > max){
            idx = i;
            max = arr[i];
        }
    }

    printf("\n");
    printf("The number is : %d\n", idx);
}