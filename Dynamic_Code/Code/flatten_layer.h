typedef struct{
    int input_size;
    int filter_size;
    int pool_size;
    int output_size;
    int pool_output_size;
    int conv_size;

    float *flat_out;
    float **b_pool;
    float *b_flat;

}Flatten_Layer;

Flatten_Layer* flatten_layer_create(int input_size, int filter_size, int pool_size);
void flatten_layer_free(Flatten_Layer* flatten_layer);
void flatten(Flatten_Layer* flatten_layer, float** input);
void unflatten(Flatten_Layer* flatten_layer);