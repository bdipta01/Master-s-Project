#define MAX_INPUTS 256
#define MAX_HIDDEN 128
#define MAX_OUTPUTS 10
#define MAX_WEIGHT MAX_HIDDEN*(MAX_INPUTS+MAX_OUTPUTS)
#define LAYER_WEIGHT MAX_WEIGHT+(MAX_HIDDEN*MAX_INPUTS)

typedef struct 
{
    int nips;  //Number of inputs
    int nhid;  //Number of hidden neurons
    int nops;  //Number of outputs
    int nb;    //number of biases
    int nw;    //number of weights

    float w[MAX_WEIGHT];  //All weights
    float *x;  /*hidden layer to output layer weights*/
    float b[2];  /*biases*/
    float h[MAX_HIDDEN];  //hold the values of neurons in the hidden layer
    float o[MAX_OUTPUTS];  //output layer
    float d_in[MAX_INPUTS];  //Gradient of loss with respect to input
}Dense_Layer;


float *dLayer_Predict(Dense_Layer *nn, const float *in);
void dLayer_Build(Dense_Layer* nn, int nips, const int nhid, const int nops);
int dLayer_Save(Dense_Layer *nn, float *arr);
void dLayer_Load(Dense_Layer *nn, const float *arr);
float dLayer_Train(Dense_Layer *nn, const float *in, const float *tg, float rate);
void dLayer_Print(const float *arr, int size);