typedef struct{
    int input_size;
    int filter_size;
    int output_size;
    int pool_size;
    int stride_size;
    int pool_output_size;
    
    float **grad_output;
    float **input;
    float **b_pool;
    float **pool_out;
    int ***max_index;
    
}Pool_Layer;

Pool_Layer* pool_layer_create(int input_size, int filter_size, int pool_size, int stride_size);
void pool_layer_free(Pool_Layer* pool_layer);
void max_pooling_forward(Pool_Layer* pool_layer, float** input);
void max_pooling_backward(Pool_Layer* pool_layer, float **input);