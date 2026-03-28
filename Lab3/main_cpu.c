/*
 * CPU Sequential Version of LeNet-5
 * Baseline implementation for performance comparison
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
    /* Allocate memory for training and test data */
    image *train_data = (image *)calloc(COUNT_TRAIN, sizeof(image));
    uint8 *train_label = (uint8 *)calloc(COUNT_TRAIN, sizeof(uint8));
    image *test_data = (image *)calloc(COUNT_TEST, sizeof(image));
    uint8 *test_label = (uint8 *)calloc(COUNT_TEST, sizeof(uint8));

    /* ==================== LOAD MNIST DATA ==================== */
    /* Load MNIST dataset */
    if (read_data(train_data, train_label, COUNT_TRAIN, FILE_TRAIN_IMAGE, FILE_TRAIN_LABEL)) {
        printf("ERROR: Training dataset files not found!\n");
        return 1;
    }
    if (read_data(test_data, test_label, COUNT_TEST, FILE_TEST_IMAGE, FILE_TEST_LABEL)) {
        printf("ERROR: Test dataset files not found!\n");
        return 1;
    }

    /* ==================== INITIALIZE NETWORK ==================== */
    /* Initialize network with random weights */
    LeNet5 *lenet = (LeNet5 *)malloc(sizeof(LeNet5));
    Initial(lenet);

    int batch_size = 300;
    int confusion_matrix[10][10] = {0};

    printf("============================================================\n");
    printf("CPU SEQUENTIAL VERSION\n");
    printf("============================================================\n");
    printf("Training: %d images, batch size: %d\n", COUNT_TRAIN, batch_size);

    /* ==================== TRAINING PHASE ==================== */
    /* Training phase - forward and backward propagation on CPU */
    clock_t train_start = clock();

    for (int i = 0, percent = 0; i <= COUNT_TRAIN - batch_size; i += batch_size) {
        TrainBatch(lenet, train_data + i, train_label + i, batch_size);
        if (i * 100 / COUNT_TRAIN > percent) {
            percent = i * 100 / COUNT_TRAIN;
            printf("Training: %d%% complete\n", percent);
        }
    }

    clock_t train_end = clock();
    double train_time = (double)(train_end - train_start) / CLOCKS_PER_SEC;
    printf("Training complete.\n");
    printf("CPU Training Time: %.2f seconds\n\n", train_time);


    /* ==================== TESTING PHASE ==================== */
    /* Testing phase - forward propagation only */
    printf("Testing: %d images\n", COUNT_TEST);
    clock_t test_start = clock();

    int right = 0;
    for (int i = 0; i < COUNT_TEST; i++) {
        uint8 l = test_label[i];
        uint8 p = Predict(lenet, test_data[i], 10);
        confusion_matrix[l][p]++;
        if (l == p) right++;
    }

    clock_t test_end = clock();
    double test_time = (double)(test_end - test_start) / CLOCKS_PER_SEC;

    /* ==================== PRINT RESULTS ==================== */
    PrintResult(confusion_matrix);

    printf("\n============================================================\n");
    printf("CPU RESULTS SUMMARY\n");
    printf("============================================================\n");
    printf("Accuracy: %d / %d correct (%.2f%%)\n", right, COUNT_TEST, right * 100.0 / COUNT_TEST);
    printf("CPU Training Time: %.2f seconds\n", train_time);
    printf("CPU Testing Time:  %.2f seconds\n", test_time);
    printf("CPU Total Time:    %.2f seconds\n", train_time + test_time);
    printf("============================================================\n");


    /* ==================== CLEANUP ==================== */
    free(lenet);
    free(train_data);
    free(train_label);
    free(test_data);
    free(test_label);

    return 0;
}