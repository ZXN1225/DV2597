/*
 * GPU CUDA Version of LeNet-5
 * Uses GPU for forward propagation, CPU for backpropagation
 */

#include "lenet.h"
#include <stdlib.h>
#include <stdio.h>
#include <time.h>

/* MNIST dataset file paths */
#define FILE_TRAIN_IMAGE    "train-images-idx3-ubyte"
#define FILE_TRAIN_LABEL    "train-labels-idx1-ubyte"
#define FILE_TEST_IMAGE     "t10k-images-idx3-ubyte"
#define FILE_TEST_LABEL     "t10k-labels-idx1-ubyte"
#define COUNT_TRAIN         60000
#define COUNT_TEST          10000

/* Load MNIST images and labels from binary files */
int read_data(unsigned char(*data)[28][28], unsigned char label[], const int count, const char data_file[], const char label_file[])
{
    FILE *fp_image = fopen(data_file, "rb");
    FILE *fp_label = fopen(label_file, "rb");
    if (!fp_image||!fp_label) return 1;
    fseek(fp_image, 16, SEEK_SET);
    fseek(fp_label, 8, SEEK_SET);
    fread(data, sizeof(*data)*count, 1, fp_image);
    fread(label, count, 1, fp_label);
    fclose(fp_image);
    fclose(fp_label);
    return 0;
}

int main()
{
    /* ==================== ALLOCATE MEMORY ==================== */
    image *train_data = (image *)calloc(COUNT_TRAIN, sizeof(image));
    uint8 *train_label = (uint8 *)calloc(COUNT_TRAIN, sizeof(uint8));
    image *test_data = (image *)calloc(COUNT_TEST, sizeof(image));
    uint8 *test_label = (uint8 *)calloc(COUNT_TEST, sizeof(uint8));

    /* ==================== LOAD MNIST DATA ==================== */
    if (read_data(train_data, train_label, COUNT_TRAIN, FILE_TRAIN_IMAGE, FILE_TRAIN_LABEL)) {
        printf("ERROR: Training dataset files not found!\n");
        return 1;
    }
    if (read_data(test_data, test_label, COUNT_TEST, FILE_TEST_IMAGE, FILE_TEST_LABEL)) {
        printf("ERROR: Test dataset files not found!\n");
        return 1;
    }

    /* ==================== INITIALIZE NETWORK ==================== */
    LeNet5 *lenet = (LeNet5 *)malloc(sizeof(LeNet5));
    Initial(lenet);

    int batch_size = 300;
    int confusion_matrix[10][10] = {0};

    printf("============================================================\n");
    printf("GPU CUDA VERSION\n");
    printf("============================================================\n");
    printf("Training: %d images, batch size: %d\n", COUNT_TRAIN, batch_size);

    /* ==================== TRAINING PHASE ==================== */
    /* GPU forward propagation, CPU backpropagation */
    clock_t train_start = clock();


    for (int i = 0, percent = 0; i <= COUNT_TRAIN - batch_size; i += batch_size) {
        TrainBatch_CUDA(lenet, train_data + i, train_label + i, batch_size);
        if (i * 100 / COUNT_TRAIN > percent) {
            percent = i * 100 / COUNT_TRAIN;
            printf("Training: %d%% complete\n", percent);
        }
    }

    clock_t train_end = clock();
    double train_time = (double)(train_end - train_start) / CLOCKS_PER_SEC;
    printf("Training complete.\n");
    printf("GPU Training Time: %.2f seconds\n\n", train_time);

    /* ==================== TESTING PHASE ==================== */
    /* GPU forward propagation in batches */
    printf("Testing: %d images\n", COUNT_TEST);
    clock_t test_start = clock();

    int right = 0;
    uint8 *results = (uint8*)malloc(batch_size * sizeof(uint8));

    for (int i = 0; i < COUNT_TEST; i += batch_size) {
        int current_batch = (COUNT_TEST - i < batch_size) ? (COUNT_TEST - i) : batch_size;

        Predict_Batch_CUDA(lenet, test_data + i, results, current_batch);

        for (int j = 0; j < current_batch; j++) {
            uint8 l = test_label[i + j];
            uint8 p = results[j];
            confusion_matrix[l][p]++;
            if (l == p) right++;
        }
    }

    clock_t test_end = clock();
    double test_time = (double)(test_end - test_start) / CLOCKS_PER_SEC;

    /* ==================== PRINT RESULTS ==================== */
    PrintResult(confusion_matrix);

    printf("\n============================================================\n");
    printf("GPU RESULTS SUMMARY\n");
    printf("============================================================\n");
    printf("Accuracy: %d / %d correct (%.2f%%)\n", right, COUNT_TEST, right * 100.0 / COUNT_TEST);
    printf("GPU Training Time: %.2f seconds\n", train_time);
    printf("GPU Testing Time:  %.2f seconds\n", test_time);
    printf("GPU Total Time:    %.2f seconds\n", train_time + test_time);
    printf("============================================================\n");

    /* ==================== CLEANUP ==================== */
    free(results);
    free(lenet);
    free(train_data);
    free(train_label);
    free(test_data);
    free(test_label);

    return 0;
}