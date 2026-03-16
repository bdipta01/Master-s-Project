typedef struct{
    int input_size;
    int filter_size;
    int output_size;

    float **filter;
    float **input;
    float **conv_layer_out;
    

    float **grad_output;
    float **grad_filter;
}Conv_Layer;

Conv_Layer* conv_layer_create(int input_size, int filter_size);
void conv_layer_free(Conv_Layer* conv_layer);
void conv_layer_forward(Conv_Layer* conv_layer);
void conv_layer_backward(Conv_Layer* conv_layer, float** input_file, float rate);