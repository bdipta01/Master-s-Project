typedef struct 
{
    float *w;  //All weights
    float *x;  /*hidden layer to output layer weights*/
    float *b;  /*biases*/
    float *h;  //hold the values of neurons in the hidden layer
    float *o;  //output layer
    float *d_in;  //Gradient of loss with respect to input
    int nb;    //number of biases
    int nw;    //number of weights
    int nips;  //Number of inputs
    int nhid;  //Number of hidden neurons
    int nops;  //Number of outputs
}Dense_Layer;

float *dLayer_Predict(const Dense_Layer nn, const float *in);
Dense_Layer dLayer_Build(const int nips, const int nhid, const int nops);
float dLayer_Train(const Dense_Layer nn, const float *in, const float *tg, float rate);
void dLayer_Save(const Dense_Layer nn, const char *path);
Dense_Layer dLayer_Load(const char *path);
void dLayer_Print(const float *arr, const int size);
void dLayer_Free(const Dense_Layer nn);