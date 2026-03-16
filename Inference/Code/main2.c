// File: train.c
// (Includes and other functions remain the same)

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "uti.h"
#include "conv_layer.h"
#include "max_pooling.h"
#include "flatten_layer.h"
#include "dense_layer.h"
#include "data_array.h"
// Note: We don't need test_data here, as we are only training and saving.

// <<< HELPER FUNCTION TO SAVE MODEL TO HEADER FILE >>>
void save_model_to_header(Conv_Layer* conv_layer, Dense_Layer* d_layer) {
    FILE *fp = fopen("trained_model1.h", "w");
    if (fp == NULL) {
        printf("Error opening file!\n");
        return;
    }

    printf("Saving trained model parameters to trained_model.h...\n");

    // --- Header Guard ---
    fprintf(fp, "#ifndef TRAINED_MODEL_H\n");
    fprintf(fp, "#define TRAINED_MODEL_H\n\n");

    // --- Save Convolutional Layer Filter ---
    fprintf(fp, "// Trained Convolutional Layer Filter Weights\n");
    fprintf(fp, "const float trained_conv_filter[%d][%d] = {\n", conv_layer->filter_size, conv_layer->filter_size);
    for (int i = 0; i < conv_layer->filter_size; i++) {
        fprintf(fp, "    {");
        for (int j = 0; j < conv_layer->filter_size; j++) {
            fprintf(fp, "%.8ef", conv_layer->filter[i][j]);
            if (j < conv_layer->filter_size - 1) {
                fprintf(fp, ", ");
            }
        }
        fprintf(fp, "}");
        if (i < conv_layer->filter_size - 1) {
            fprintf(fp, ",\n");
        }
    }
    fprintf(fp, "\n};\n\n");

    // --- Save Dense Layer Weights and Biases ---
    float model_data[3 + 2 + MAX_WEIGHT]; // MAX_WEIGHT should be defined in your dense_layer.h
    int model_size = dLayer_Save(d_layer, model_data);

    fprintf(fp, "// Trained Dense Layer Parameters (weights and biases)\n");
    fprintf(fp, "const int trained_dense_model_size = %d;\n", model_size);
    fprintf(fp, "const float trained_dense_model[] = {\n    ");
    for (int i = 0; i < model_size; i++) {
        fprintf(fp, "%.8ef", model_data[i]);
        if (i < model_size - 1) {
            fprintf(fp, ", ");
        }
        if ((i + 1) % 8 == 0) { // Newline every 8 values for readability
            fprintf(fp, "\n    ");
        }
    }
    fprintf(fp, "\n};\n\n");

    // --- Closing Header Guard ---
    fprintf(fp, "#endif // TRAINED_MODEL_H\n");
    fclose(fp);
    printf("Model saved successfully.\n");
}


int main() {
    // --- All your existing variable declarations and data loading ---
    const int nips = 256;
    const int nhid = 16;
    const int nops = 10;
    const int conv_input = sqrt(256);
    float rate = 1.0f;
    const float eta = 0.99f;
    const int iterations = 25;

    static Data data;
    // ... (your data loading for 'train_inputs' and 'train_outputs' remains here)
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

    // --- All your existing layer creation and initialization ---
    Conv_Layer conv_layer;
    conv_layer_create(&conv_layer, conv_input, 3);
    Pool_Layer pool_layer;
    pool_layer_create(&pool_layer, conv_input, 3, 2, 2);
    Flatten_Layer flatten_layer;
    flatten_layer_create(&flatten_layer, conv_input, 3, 2);
    Dense_Layer d_layer;
    dLayer_Build(&d_layer, flatten_layer.conv_size, nhid, nops);
    // ... (your random filter initialization remains here)
    for(int i=0; i<conv_layer.filter_size; i++){
        for(int j=0; j<conv_layer.filter_size; j++){
            conv_layer.filter[i][j] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f;
        }
    }


    // --- Your entire training loop remains exactly the same ---
    for (int epoch = 1; epoch <= iterations; epoch++) {
        data_shuffle(&data);
        float error = 0.0f;
        for (int row = 0; row < data.rows; row++) {
            // ... (forward pass)
            for(int i=0; i<conv_layer.input_size; i++){
                for(int j=0; j<conv_layer.input_size; j++){
                    conv_layer.input[i][j] = data.in[row][i][j] / 255.0f;
                }
            }
            conv_layer_forward(&conv_layer);
            max_pooling_forward(&pool_layer, conv_layer.conv_layer_out);
            flatten(&flatten_layer, pool_layer.pool_out);
            
            // ... (training step)
            const float * const in = flatten_layer.flat_out;
            const float * const tg = data.tg[row];
            error += dLayer_Train(&d_layer, in, tg, rate);

            // ... (backward pass)
            for(int i=0; i<flatten_layer.conv_size; i++){
                flatten_layer.b_flat[i] = d_layer.d_in[i];
            }
            unflatten(&flatten_layer);
            max_pooling_backward(&pool_layer, flatten_layer.b_pool);
            conv_layer_backward(&conv_layer, pool_layer.grad_output, rate);
        }
        printf("Iteration: %d Error %.6f :: learning rate %f\n", epoch, (double)error / data.rows, (double)rate);
        rate *= eta;
    }

    // --- <<< MODIFIED PART: SAVE THE TRAINED MODEL >>> ---
    // Instead of testing here, we save the final parameters.
    save_model_to_header(&conv_layer, &d_layer);

    // We can remove the entire testing section from this file.
    return 0;
}