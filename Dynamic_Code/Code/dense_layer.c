#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdarg.h>
#include "dense_layer.h"


static float act(const float a){           //Sigmoid activation function
    return 1.0f / (1.0f + expf (-a));
}

static float pdact(const float a){        //Derivative of Activation function
    return a * (1.0f - a);
}

static float frand(){                     //Random floating point number
    return rand()/(float)RAND_MAX;
}

static float pderr(const float a, const float b){       //Derivative of error
    return a-b;
}

static void fprop(const Dense_Layer nn, const float * const in){
    /*Compute Hidden Layer neuron values*/
    for(int i=0; i<nn.nhid; i++){
        float sum = 0.0f;
        for(int j=0; j<nn.nips; j++){
            sum += in[j] * nn.w[i*nn.nips+j];  //input * weight of all connected synopses of hidden layer 1
        }
    nn.h[i] = act(sum + nn.b[0]);
    }

    /*Compute Output Layer neuron values*/
    for(int i=0; i<nn.nops; i++){
        float sum = 0.0f;
        for(int j=0; j<nn.nhid; j++){
            sum += nn.h[j] * nn.x[i * nn.nhid + j];
        }
    nn.o[i] = act(sum + nn.b[1]);
    }
}

static void bprop(const Dense_Layer nn, const float * const in, const float * const tg, float rate){
    for(int i=0; i<nn.nhid; i++){
        float sum = 0.0f;
        for(int j=0; j<nn.nops; j++){
            const float a = pderr(nn.o[j], tg[j]);  //nn.o[j] - tg[j] //output value - expected value
            const float b = pdact(nn.o[j]);         //nn.o[j] * (1.0f - nn.o[j]); //applied activation function on output value

            const float err_signal = a * b;
            sum += err_signal * nn.x[j*nn.nhid+i];        // a*b represents the error signal for output neuron j

            nn.x[j * nn.nhid +i] -= rate * err_signal * nn.h[i];  //rate = learning rate determines how much weight should be adjusted
        } 

        for(int j=0; j<nn.nips; j++){
            nn.w[i * nn.nips + j] -= rate * sum * pdact(nn.h[i]) * in[j];
        }
    }

    for(int j=0; j<nn.nips; j++){
        float grad = 0.0f;
        for(int i=0; i<nn.nhid; i++){
            float out_grad = 0.0f;
            for(int k=0; k<nn.nops; k++){
                float a = pderr(nn.o[k], tg[k]);
                float b = pdact(nn.o[k]);
                float err_signal = a*b;
                out_grad += err_signal * nn.x[k * nn.nhid + i];
            }
            grad += out_grad * pdact(nn.h[i]) * nn.w[i * nn.nips + j];
        }
        nn.d_in[j] = grad;
    }
}


/*Weight bias random*/
static void wbrand(const Dense_Layer nn){
    for(int i=0; i<nn.nw; i++){
        nn.w[i] = frand() - 0.5f;
    }

    for(int i=0; i<nn.nb; i++){
        nn.b[i] = frand() - 0.5f;
    }
}

float *dLayer_Predict(const Dense_Layer nn, const float *in){
    fprop(nn, in);
    return nn.o;
}

Dense_Layer dLayer_Build(const int nips, const int nhid, const int nops){
    Dense_Layer nn;
    nn.nb = 2;                                          //number of biases
    nn.nw = nhid * (nips + nops);                       //number of weights
    nn.w = (float *)malloc(nn.nw * sizeof(float));       //All weights
    nn.x = nn.w + nhid * nips;                          //Hidden layer to output layer weights
    nn.b = (float *)malloc(nn.nb * sizeof(float));       //biases
    nn.h = (float *)malloc(nhid * sizeof(float));        //Hidden layer
    nn.o = (float *)malloc(nops * sizeof(float));        //Output layer
    nn.d_in = (float *)malloc(nips * sizeof(float));

    if(!nn.w || !nn.b || !nn.h || !nn.o || !nn.d_in){
        printf("Memory allocation failed in weight, bias, hidden layer, output.\n");
        exit(EXIT_FAILURE);
    }

    nn.nips = nips;                                     //Number of inputs
    nn.nhid = nhid;                                     //Number of hidden neurons
    nn.nops = nops;                                     //Number of outputs
    wbrand(nn);                 //Random weight and bias value initialize

    return nn;
}

void dLayer_Save(const Dense_Layer nn, const char *path){
    FILE * const file = fopen(path, "w");

    if(!file){
        printf("Failed to open file: %s\n", path);
        return;
    }
    /*Save the header*/
    fprintf(file, "%d %d %d\n", nn.nips, nn.nhid, nn.nops);   //fprintf writes data to a file

    /*Save the biases*/
    for(int i=0; i<nn.nb; i++){
        fprintf(file, "%f\n", (double)nn.b[i]);
    }

    /*Save the weights*/
    for(int i=0; i<nn.nw; i++){
        fprintf(file, "%f\n", (double)nn.w[i]);
    }

    fclose(file);
}

Dense_Layer dLayer_Load(const char *path){
    FILE * const file = fopen(path, "r");

    if(!file){
        printf("Failed to open file: %s\n", path);
        exit(EXIT_FAILURE);
    }

    int nips = 0;
    int nhid = 0;
    int nops = 0;

    /*Load the header*/
    fscanf(file, "%d %d %d\n", &nips, &nhid, &nops);  //read from the file

    const Dense_Layer nn = dLayer_Build(nips, nhid, nops); //allocate memory and initialize the neural network with correct size

    /*Load the biases*/
    for(int i=0; i<nn.nb; i++){
        fscanf(file, "%f\n", &nn.b[i]);
    }

    /*Load the weights*/
    for(int i=0; i<nn.nw; i++){
        fscanf(file, "%f\n", &nn.w[i]);
    }

    fclose(file);

    return nn;
}

float dLayer_Train(const Dense_Layer nn, const float *in, const float *tg, float rate){
    fprop(nn, in);  //Compute hidden and output layer
    bprop(nn, in, tg, rate);  //update the weight and biases using the error

    float error = 0.0f;
    for(int i = 0; i < nn.nops; i++){
        float diff = tg[i] - nn.o[i];
        error += 0.5f * diff * diff;
    }

    return error;
}

void dLayer_Print(const float *arr, const int size){
    double max = 0.0f;
    int idx;
    
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

void dLayer_Free(const Dense_Layer nn){
    free(nn.w);
    free(nn.b);
    free(nn.h);
    free(nn.o);
    free(nn.d_in);
}