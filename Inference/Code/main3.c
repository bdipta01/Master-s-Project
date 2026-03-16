#include <stdio.h>
#include <math.h>
#include "uti.h"
#include "conv_layer.h"
#include "max_pooling.h"
#include "flatten_layer.h"
#include "dense_layer.h"
#include "data_array.h"
#include "test_data_8.h"
#include "trained_model.h"  // ← include the trained weights

int main() {
    const int conv_input = sqrt(256);

    Conv_Layer conv_layer;
    conv_layer_create(&conv_layer, conv_input, 3);

    Pool_Layer pool_layer;
    pool_layer_create(&pool_layer, conv_input, 3, 2, 2);

    Flatten_Layer flatten_layer;
    flatten_layer_create(&flatten_layer, conv_input, 3, 2);

    Dense_Layer d_layer;
    dLayer_Load(&d_layer, trained_mode_data);  // ← load weights from header

    // Copy convolution filter weights (you can also save these to header similarly)
    for (int i = 0; i < conv_layer.filter_size; i++) {
        for (int j = 0; j < conv_layer.filter_size; j++) {
            conv_layer.filter[i][j] = 0.1f; // or set from another header
        }
    }

    for (int row = 0; row < TEST_NUM_SAMPLES; row++) {
        for (int i = 0; i < conv_layer.input_size; i++) {
            for (int j = 0; j < conv_layer.input_size; j++) {
                conv_layer.input[i][j] = test_inputs[row][i][j] / 255.0f;
            }
        }

        conv_layer_forward(&conv_layer);
        max_pooling_forward(&pool_layer, conv_layer.conv_layer_out);
        flatten(&flatten_layer, pool_layer.pool_out);

        const float *in = flatten_layer.flat_out;
        const float *pd = dLayer_Predict(&d_layer, in);

        printf("Sample %d -> ", row);
        dLayer_Print(pd, 10);
    }

    return 0;
}
