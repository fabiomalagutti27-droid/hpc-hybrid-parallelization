// File: serial.c
// Scopo: versione sequenziale deterministica; ground truth per misurare speedup.
// Ruolo: riferimento di correttezza (checksum) e confronto con MPI/OpenMP.
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

/* Progetto HPC: Binarizzazione Matrice - Versione Sequenziale
 * Serve da riferimento per verificare le versioni parallele (MPI e OpenMP).
 */

#define RAGGIO 1   // Intorno: 1 cella per lato (quadrato 3x3)
#define MAX_VAL 100.0f

// Helper per stampare matrici piccole (debug)
void print_matrix(float *M, int n, const char *name) {
    if (n > 10) return; // Stampa solo se piccola
    printf("\nMatrice %s:\n", name);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            printf("%6.2f ", M[i*n + j]);
        }
        printf("\n");
    }
}

void print_binary_matrix(int *M, int n, const char *name) {
    if (n > 10) return;
    printf("\nMatrice %s (Binaria):\n", name);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            printf("%d ", M[i*n + j]);
        }
        printf("\n");
    }
}

int main(int argc, char *argv[]) {
    int N = 10; // Dimensione default
    
    // Lettura dimensione da riga di comando (opzionale)
    if (argc > 1) {
        N = atoi(argv[1]);
    }

    printf("AVVIO SERIALE: Matrice %dx%d\n", N, N);

    // 1. Allocazione: array 1D contiguo per cache; A=input, T=output binario
    float *A = (float *)malloc(N * N * sizeof(float));
    int *T = (int *)malloc(N * N * sizeof(int));

    if (!A || !T) {
        fprintf(stderr, "Errore allocazione memoria\n");
        return 1;
    }

    // 2. Inizializzazione (numeri casuali, seed fisso)
    srand(42);
    for (int i = 0; i < N * N; i++) {
        A[i] = ((float)rand() / RAND_MAX) * MAX_VAL;
    }

    // 3. Calcolo Sequenziale
    // Usiamo clock() per misurare il tempo CPU su macchina singola
    clock_t start = clock();

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            
            float somma = 0.0f;
            int conta = 0;

            // Calcolo intorno (gestione bordi inclusa)
            for (int dx = -RAGGIO; dx <= RAGGIO; dx++) {
                for (int dy = -RAGGIO; dy <= RAGGIO; dy++) {
                    
                    int r = i + dx;
                    int c = j + dy;

                    // Controlla se il vicino è dentro la matrice
                    if (r >= 0 && r < N && c >= 0 && c < N) {
                        somma += A[r * N + c];
                        conta++;
                    }
                }
            }

            float media = somma / conta;

            // Binarizzazione
            if (A[i * N + j] > media) {
                T[i * N + j] = 1;
            } else {
                T[i * N + j] = 0;
            }
        }
    }

    clock_t end = clock();
    double tempo_sec = (double)(end - start) / CLOCKS_PER_SEC;

 // 4. Output risultati
    // print_matrix(A, N, "Input");      <-- Commentato per evitare spam su N grandi
    // print_binary_matrix(T, N, "Output");
    
    // CALCOLO CHECKSUM (numero di 1) per confrontare con le versioni parallele
    long long checksum = 0;
    for (int i = 0; i < N * N; i++) {
        checksum += T[i];
    }

    // STAMPE PER IL PARSER (non modificare)
    printf("Tempo: %f\n", tempo_sec);
    printf("Checksum: %lld\n", checksum);

    free(A);
    free(T);
    return 0;
}