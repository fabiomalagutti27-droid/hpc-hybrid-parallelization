// File: main_omp.c
// Scopo: versione OpenMP su memoria condivisa; parallelizza il doppio ciclo (i,j).
// Ruolo: baseline intra-nodo per confronto con MPI e ibrido.
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h> 

#define RAGGIO 1
#define MAX_VAL 100.0f

int main(int argc, char *argv[]) {
    int N = 2000; // Default a 2000 per vedere subito la differenza
    int num_threads = 4; // Default thread

    if (argc > 1) N = atoi(argv[1]);
    if (argc > 2) num_threads = atoi(argv[2]);

    // Imposta il numero di thread da usare (memoria condivisa, niente overhead MPI)
    omp_set_num_threads(num_threads);

    printf("AVVIO OpenMP: Matrice %dx%d con %d Thread\n", N, N, num_threads);

    // Allocazione
    float *A = (float *)malloc(N * N * sizeof(float));
    int *T = (int *)malloc(N * N * sizeof(int));

    if (!A || !T) {
        fprintf(stderr, "Errore memoria\n");
        return 1;
    }

    // Inizializzazione (sequenziale, seed fisso per ripetibilità)
    srand(12345);
    for (int i = 0; i < N * N; i++) {
        A[i] = ((float)rand() / RAND_MAX) * MAX_VAL;
    }

    // --- INIZIO CALCOLO PARALLELO ---
    double start = omp_get_wtime(); // Uso il timer di OpenMP (Wall time)

    // Worksharing: divide le righe tra thread, variabili interne restano private
    #pragma omp parallel for
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            
            float somma = 0.0f;
            int conta = 0;

            for (int dx = -RAGGIO; dx <= RAGGIO; dx++) {
                for (int dy = -RAGGIO; dy <= RAGGIO; dy++) {
                    int r = i + dx;
                    int c = j + dy;

                    if (r >= 0 && r < N && c >= 0 && c < N) {
                        somma += A[r * N + c];
                        conta++;
                    }
                }
            }

            if (A[i * N + j] > (somma / conta)) {
                T[i * N + j] = 1;
            } else {
                T[i * N + j] = 0;
            }
        }
    }

double end = omp_get_wtime();

    // --- OUTPUT E PULIZIA ---

    // Calcolo Checksum (Somma di tutti gli 1)
    long long checksum = 0;
    // La matrice T è linearizzata, quindi scorriamo come array 1D
    for (int i = 0; i < N * N; i++) {
        checksum += T[i];
    }

    // STAMPE FORMATTATE PER LO SCRIPT (Non cambiare queste righe!)
    printf("Tempo: %f\n", end - start);
    printf("Checksum: %lld\n", checksum);

    free(A);
    free(T);

    return 0;
}