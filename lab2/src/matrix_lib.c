#include "matrix_lib.h"
#include <pthread.h>
#include <string.h>
#include <stdlib.h>
#include <string.h>
#include <immintrin.h> 

typedef struct {
    float scalar;
    float* m_start;
    float* r_start;
    int length;
} scalar_thread_args_ptr;

void* scalar_worker_ptr(void* arg) {
    scalar_thread_args_ptr* args = (scalar_thread_args_ptr*) arg;
    float* m_ptr = args->m_start;
    float* r_ptr = args->r_start;
    int len = args->length;

    for (int i = 0; i < len; i++) {
        *r_ptr = args->scalar * *m_ptr;
        r_ptr++;
        m_ptr++;
    }

    return NULL;
}

int scalar_matrix_mult(float scalar_value, matrix* m, matrix* r) {
    if (!m || !r || !m->values || !r->values) return -1;
    if (m->rows != r->rows || m->cols != r->cols) return -2;

    int num_threads = 8;
    int total = m->rows * m->cols;

    pthread_t threads[num_threads];
    scalar_thread_args_ptr args[num_threads];

    int chunk = total / num_threads;
    int remainder = total % num_threads;

    for (int i = 0; i < num_threads; i++) {
        int current_chunk = chunk + (i < remainder ? 1 : 0);  

        args[i].scalar = scalar_value;
        args[i].m_start = m->values + i * chunk + (i < remainder ? i : remainder);
        args[i].r_start = r->values + i * chunk + (i < remainder ? i : remainder);
        args[i].length = current_chunk;

        pthread_create(&threads[i], NULL, scalar_worker_ptr, &args[i]);
    }

    for (int i = 0; i < num_threads; i++) {
        pthread_join(threads[i], NULL);
    }

    return 0;
}


typedef struct {
    float* m1;
    float* m2;
    float* r;
    int m1_rows;
    int m1_cols;
    int m2_cols;
    int start_row;
    int end_row;
} thread_args;

void* optimized_worker(void* arg) {
    thread_args* args = (thread_args*) arg;

    float* m1_ptr = args->m1 + args->start_row * args->m1_cols;
    float* r_ptr  = args->r  + args->start_row * args->m2_cols;
    float* m2_ptr = args->m2;

    for (int i = args->start_row; i < args->end_row; i++) {
        float* m1_elem_ptr = m1_ptr;
        float* r_row_ptr   = r_ptr;

        for (int j = 0; j < args->m1_cols; j++) {
            float value = *m1_elem_ptr;
            float* m2_row_ptr = m2_ptr + j * args->m2_cols;
            float* r_col_ptr  = r_row_ptr;

            for (int k = 0; k < args->m2_cols; k++) {
                *r_col_ptr += value * *m2_row_ptr;
                r_col_ptr++;
                m2_row_ptr++;
            }
            m1_elem_ptr++;
        }

        m1_ptr += args->m1_cols;
        r_ptr  += args->m2_cols;
    }

    return NULL;
}

int matrix_matrix_mult(matrix* m1, matrix* m2, matrix* r) {
    if (!m1 || !m2 || !r || !m1->values || !m2->values || !r->values) return -1;
    if (m1->cols != m2->rows) return 1;

    int num_threads = 8;

    r->rows = m1->rows;
    r->cols = m2->cols;

    memset(r->values, 0, sizeof(float) * r->rows * r->cols);

    pthread_t threads[num_threads];
    thread_args args[num_threads];

    int chunk = m1->rows / num_threads;

    for (int i = 0; i < num_threads; i++) {
        args[i].m1 = m1->values;
        args[i].m2 = m2->values;
        args[i].r  = r->values;
        args[i].m1_rows = m1->rows;
        args[i].m1_cols = m1->cols;
        args[i].m2_cols = m2->cols;
        args[i].start_row = i * chunk;
        args[i].end_row = (i == num_threads - 1) ? m1->rows : (i + 1) * chunk;
        pthread_create(&threads[i], NULL, optimized_worker, &args[i]);
    }

    for (int i = 0; i < num_threads; i++) {
        pthread_join(threads[i], NULL);
    }

    return 0;
}


// /**
//  * @brief multiplies tow matrices (using intel intrinsics lib)
//  * 
//  * @param m1 pointer to the first matrix
//  * @param m2 pointer to the second matrix
//  * @param r pointer to the result matrix
//  * @return 0 in case of success or the error code
//  */

// int matrix_matrix_mult(matrix* m1, matrix* m2, matrix* r) {
//     if(m1->cols != m2->rows) return 1;

//     r->rows = m1->rows;
//     r->cols = m2->cols;

//     if(!m1 || !m2 || !r || !m1->values || !m2->values) return -1;

//     if(m1->rows % 8 != 0 || m1->cols % 8 != 0 || m2->rows % 8 != 0 || m2->cols % 8 != 0) return -3;

//     memset(r->values, 0, sizeof(float) * r->rows * r->cols);

//     float* m1_ptr = m1->values;
//     float* m2_ptr = m2->values;
//     float* r_ptr = r->values;

//     for(int i = 0; i < m1->rows; i++) {
//         float* m1_elem_ptr = m1_ptr;
//         float* r_row_ptr = r_ptr;

//         for(int j = 0; j < m1->cols; j++) {
//             float value = *m1_elem_ptr;
//             float* m2_row_ptr = m2_ptr + (j * m2->cols);

//             float* r_col_ptr = r_row_ptr;
//             __m256 value_vector = _mm256_set1_ps(value);

//             for(int k = 0; k < m2->cols; k+=8) {
//                 __m256 m2_vals = _mm256_load_ps(m2_row_ptr);
//                 __m256 r_vals = _mm256_load_ps(r_col_ptr);
//                 __m256 mult = _mm256_mul_ps(m2_vals, value_vector);
//                 __m256 sum = _mm256_add_ps(r_vals, mult);
//                 _mm256_store_ps(r_col_ptr, sum);
//                 m2_row_ptr+=8;
//                 r_col_ptr+=8;
//             }
//             m1_elem_ptr++;
//         }
//         m1_ptr += m1->cols;
//         r_ptr += r->cols;
//     }

//     return 0;
// }


// /**
//  * @brief multiplies a matrix by a scalar value (using intel intrinsics lib)
//  * 
//  * @param scalar_value value that multiplies the matrix
//  * @param m pointer to the original matrix
//  * @param r pointer to the result matrix
//  * @return 0 in case of success or the error code
//  */
// int scalar_matrix_mult(float scalar_value, matrix *m, matrix *r){
//     if(!m || !r || !m->values || !r->values) return -1;
//     if (m->rows != r->rows || m->cols != r->cols) return -2;
//     if(m->rows % 8 != 0 || m->cols % 8 != 0) return -3;

//     float* m_ptr = m->values;
//     float* r_ptr = r->values;

//     __m256 scalarArr = _mm256_set1_ps(scalar_value); 
   
//     int total = m->rows * m->cols;
//     for (int i = 0; i < total; i += 8) {
//         __m256 mLine = _mm256_load_ps(m_ptr);    
//         __m256 resLine = _mm256_mul_ps(mLine, scalarArr);  
//         _mm256_store_ps(r_ptr, resLine);
//         m_ptr += 8;
//         r_ptr += 8;
//     }
//     return 0;    
// }

// /**
//  * @brief multiplies a matrix by a scalar value (rows first)
//  * 
//  * @param scalar_value value that multiplies the matrix
//  * @param m pointer to the original matrix
//  * @param r pointer to the result matrix
//  * @return 0 in case of success or the error code
//  */

// int scalar_matrix_mult_lines(float scalar_value, matrix* m, matrix* r) {
//     if(!m || !r || !m->values || !r->values) return -1;
//     if (m->rows != r->rows || m->cols != r->cols) return -2;

//     for(int i = 0; i < m->rows; i++){
//         for(int j = 0; j < m->cols; j++) {
//             int idx = i * m-> cols + j;

//             r->values[idx] = scalar_value * m->values[idx];
//         }
//     }

//     return 0;
// }

// /**
//  * @brief multiplies a matrix by a scalar value (columns first)
//  * 
//  * @param scalar_value value that multiplies the matrix
//  * @param m pointer to the original matrix
//  * @param r pointer to the result matrix
//  * @return 0 in case of success or the error code
//  */

// int scalar_matrix_mult_cols(float scalar_value, matrix *m, matrix *r) {
//     if(!m || !r || !m->values || !r->values) return -1;
//     if (m->rows != r->rows || m->cols != r->cols) return -2;


//     for(int j = 0; j < m->cols; j++){
//         for(int i = 0; i < m->rows; i++){
//             int idx = i * m-> cols + j;

//             r->values[idx] = scalar_value * m->values[idx];
//         }
//     }

//     return 0;
// }

// /**
//  * @brief multiplies a matrix by a scalar value (pointer-based version)
//  * 
//  * @param scalar_value value that multiplies the matrix
//  * @param m pointer to the original matrix
//  * @param r pointer to the result matrix
//  * @return 0 in case of success or the error code
//  */

// int scalar_matrix_mult_ptr(float scalar_value, matrix* m, matrix* r) {
//     if(!m || !r || !m->values || !r->values) return -1; 

//     int size = m->rows * m->cols;

//     float* m_ptr = m->values;
//     float* r_ptr = r->values;
    
//     for(int i = 0; i < size; i++) {
//         *r_ptr = scalar_value * *(m_ptr);
//         m_ptr++;
//         r_ptr++;
//     }

//     return 0;
// }


// /**
//  * @brief multiplies two matrices (traditional version)
//  * 
//  * @param m1 pointer to the first matrix
//  * @param m2 pointer to the second matrix
//  * @param r pointer to the result matrix
//  * @return 0 in case of success or the error code
//  */

// int matrix_matrix_mult_trad(matrix *m1, matrix *m2, matrix *r){
//     if(m1->cols != m2->rows) return 1;
    
//     r->rows = m1->rows;
//     r->cols = m2->cols;

//     if(!r->values) return -1;

//     for(int i1 = 0; i1 < m1->rows; i1++) {
//         for(int j2 = 0; j2 < m2->cols; j2++) {
//             r->values[i1 * r->cols + j2] = 0;

//             for(int j1 = 0; j1 < m1->cols; j1++) {
//                 r->values[i1 * r->cols + j2] += m1->values[i1 * m1->cols + j1] * m2->values[j1 * m2->cols + j2];
//             }
//         }
//     }
    
//     return 0;   

// }

// /**
//  * @brief multiplies two matrices (optimized version)
//  * 
//  * @param m1 pointer to the first matrix
//  * @param m2 pointer to the second matrix
//  * @param r pointer to the result matrix
//  * @return 0 in case of success or the error code
//  */

// int matrix_matrix_mult_opt(matrix* m1, matrix* m2, matrix* r) {
//     if(m1->cols != m2->rows) return 1;

//     r->rows = m1->rows;
//     r->cols = m2->cols;

//     if(!r->values) return -1;

//     memset(r->values, 0, sizeof(float) * r->rows * r->cols);

//     float* m1_ptr = m1->values;
//     float* m2_ptr = m2->values;
//     float* r_ptr = r->values;

//     for(int i = 0; i < m1->rows; i++) {
//         float* m1_elem_ptr = m1_ptr;
//         float* r_row_ptr = r_ptr;

//         for(int j = 0; j < m1->cols; j++) {
//             float value = *m1_elem_ptr;
//             float* m2_row_ptr = m2_ptr + (j * m2->cols);

//             float* r_col_ptr = r_row_ptr;

//             for(int k = 0; k < m2->cols; k++) {
//                 *r_col_ptr += value * *m2_row_ptr;
//                 m2_row_ptr++;
//                 r_col_ptr++;
//             }
//             m1_elem_ptr++;
//         }
//         m1_ptr += m1->cols;
//         r_ptr += r->cols;
//     }


//     return 0;
// }