// File: main_mpi_final.c
// Scopo: versione ibrida MPI+OpenMP della binarizzazione; righe distribuite con Scatterv/Gatherv,
//        halo exchange non-blocking per overlap comm/calcolo, OpenMP per parallelismo intra-nodo.
// Ruolo: kernel usato nei benchmark strong/weak e test ibrido.
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// Questa parte viene inclusa SOLO se compili con -fopenmp
#ifdef _OPENMP
#include <omp.h>
#endif

// ... resto del codice ...

#define RAGGIO 1
#define MAX_VAL 100.0f
#define TAG_GHOST 10

int main(int argc, char *argv[]) {
    int rank, size;
    int N = 2000; // Default
    int provided;

    // 1. Inizializzazione MPI con supporto thread (FUNNELED: solo il master invoca MPI)
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc > 1) N = atoi(argv[1]);

    if (rank == 0) {
        printf("=== START PROGETTO HPC ===\n");
        printf("Matrice: %dx%d\n", N, N);
        printf("Processi MPI: %d\n", size);
        #pragma omp parallel
        {
            if (omp_get_thread_num() == 0)
                printf("Thread OpenMP per processo: %d\n", omp_get_num_threads());
        }
        if (provided < MPI_THREAD_FUNNELED) {
            printf("Warning: Il supporto thread MPI richiesto non e' disponibile.\n");
        }
    }

    // ---------------------------------------------------------------
    // 2. CALCOLO DELLA PARTIZIONE (Load Balancing per Scatterv)
    // ---------------------------------------------------------------

    // Array per Scatterv/Gatherv (significativi solo su rank 0)
    int *sendcounts = NULL;
    int *displs = NULL;
    
    // Decomposizione per righe; resto distribuito ai primi rank per bilanciare
    int base_rows = N / size;
    int remainder = N % size;
    int my_rows = base_rows + (rank < remainder ? 1 : 0);

    if (rank == 0) {
        sendcounts = (int *)malloc(size * sizeof(int));
        displs = (int *)malloc(size * sizeof(int));
        
        int offset = 0;
        for (int i = 0; i < size; i++) {
            int rows = base_rows + (i < remainder ? 1 : 0);
            sendcounts[i] = rows * N; // Inviamo N float per ogni riga
            displs[i] = offset;
            offset += sendcounts[i];
        }
    }

    // ---------------------------------------------------------------
    // 3. ALLOCAZIONE MEMORIA LOCALE
    // ---------------------------------------------------------------

    // Alloco righe locali + 2 ghost (sopra/sotto) in blocco contiguo
    float *local_A = (float *)malloc((my_rows + 2) * N * sizeof(float));
    int *local_T = (int *)malloc(my_rows * N * sizeof(int)); // Il risultato non ha ghost

    // Master alloca matrice completa
    float *full_A = NULL;
    int *full_T = NULL;

    if (rank == 0) {
        full_A = (float *)malloc(N * N * sizeof(float));
        full_T = (int *)malloc(N * N * sizeof(int));
        
                // Inizializzazione deterministica (first touch)
                srand(42);
        #pragma omp parallel for
        for (int i = 0; i < N * N; i++) {
            full_A[i] = ((float)rand() / RAND_MAX) * MAX_VAL;
        }
    }

    // ---------------------------------------------------------------
    // 4. DISTRIBUZIONE DATI (SCATTERV)
    // ---------------------------------------------------------------

    // &local_A[N]: scriviamo dalla riga 1 lasciando la ghost superiore libera
    MPI_Scatterv(full_A, sendcounts, displs, MPI_FLOAT, 
                 &local_A[N], my_rows * N, MPI_FLOAT, 
                 0, MPI_COMM_WORLD);

    double start_time = MPI_Wtime();

    // ---------------------------------------------------------------
    // 5. HALO EXCHANGE NON-BLOCKING (Overlapping)
    // ---------------------------------------------------------------
    
    MPI_Request reqs[4];
    int n_req = 0;

    // Ricezione asincrona per sovrapporre comunicazione e calcolo
    if (rank > 0) // Ricevo da sopra
        MPI_Irecv(&local_A[0], N, MPI_FLOAT, rank - 1, TAG_GHOST, MPI_COMM_WORLD, &reqs[n_req++]);
    
    if (rank < size - 1) // Ricevo da sotto
        MPI_Irecv(&local_A[(my_rows + 1) * N], N, MPI_FLOAT, rank + 1, TAG_GHOST, MPI_COMM_WORLD, &reqs[n_req++]);

    // Invio asincrono delle righe di bordo
    if (rank > 0) // Invio la mia prima riga a sopra
        MPI_Isend(&local_A[N], N, MPI_FLOAT, rank - 1, TAG_GHOST, MPI_COMM_WORLD, &reqs[n_req++]);

    if (rank < size - 1) // Invio la mia ultima riga a sotto
        MPI_Isend(&local_A[my_rows * N], N, MPI_FLOAT, rank + 1, TAG_GHOST, MPI_COMM_WORLD, &reqs[n_req++]);

    // ---------------------------------------------------------------
    // 6. CALCOLO PARTE INTERNA (BODY) MENTRE I MESSAGGI VIAGGIANO
    // ---------------------------------------------------------------

    // Righe 2..my_rows-1: indipendenti dalle ghost; overlap comm/calcolo
    #pragma omp parallel for collapse(2) schedule(static)
    for (int i = 2; i < my_rows; i++) {
        for (int j = 0; j < N; j++) {
            
            float somma = 0.0f;
            int conta = 0;

            for (int dx = -RAGGIO; dx <= RAGGIO; dx++) {
                for (int dy = -RAGGIO; dy <= RAGGIO; dy++) {
                    int r = i + dx;
                    int c = j + dy;

                    if (c >= 0 && c < N) {
                        somma += local_A[r * N + c];
                        conta++;
                    }
                }
            }
            
            local_T[(i - 1) * N + j] = (local_A[i * N + j] > (somma / conta)) ? 1 : 0;
        }
    }

    // ---------------------------------------------------------------
    // 7. SINCRONIZZAZIONE E CALCOLO BORDI
    // ---------------------------------------------------------------

    // Attende arrivo ghost prima di elaborare le righe di bordo
    MPI_Waitall(n_req, reqs, MPI_STATUSES_IGNORE);

    // Calcolo prima/ultima riga locale
    int rows_to_calc[2];
    int count_border = 0;
    
    if (my_rows >= 1) rows_to_calc[count_border++] = 1;         // Prima riga
    if (my_rows >= 2) rows_to_calc[count_border++] = my_rows;   // Ultima riga
    // Nota: se my_rows è 1, processiamo solo la riga 1 (che è anche l'ultima)

    for (int k = 0; k < count_border; k++) {
        int i = rows_to_calc[k];
        
        #pragma omp parallel for
        for (int j = 0; j < N; j++) {
            float somma = 0.0f;
            int conta = 0;

            for (int dx = -RAGGIO; dx <= RAGGIO; dx++) {
                for (int dy = -RAGGIO; dy <= RAGGIO; dy++) {
                    int r = i + dx;
                    int c = j + dy;

                    // Esclude ghost fuori dominio globale (rank 0 sopra, rank size-1 sotto)
                    int valid_row = 1;
                    
                    if (r == 0 && rank == 0) valid_row = 0;
                    if (r == my_rows + 1 && rank == size - 1) valid_row = 0;

                    if (valid_row && c >= 0 && c < N) {
                        somma += local_A[r * N + c];
                        conta++;
                    }
                }
            }
            local_T[(i - 1) * N + j] = (local_A[i * N + j] > (somma / conta)) ? 1 : 0;
        }
    }

    double end_time = MPI_Wtime();

    // ---------------------------------------------------------------
    // 8. RACCOLTA RISULTATI (GATHERV)
    // ---------------------------------------------------------------
    MPI_Gatherv(local_T, my_rows * N, MPI_INT, 
                full_T, sendcounts, displs, MPI_INT, 
                0, MPI_COMM_WORLD);

 
    // ---------------------------------------------------------------
    // 9. OUTPUT E PULIZIA
    // ---------------------------------------------------------------
    if (rank == 0) {
        // CALCOLO CHECKSUM
        long long checksum = 0;
        for (int i = 0; i < N * N; i++) {
            checksum += full_T[i];
        }
        
        // Output unico
        printf("Tempo: %f\n", end_time - start_time);
        printf("Checksum: %lld\n", checksum);
        fflush(stdout); 

        free(full_A);
        free(full_T);
        free(sendcounts);
        free(displs);
    }

    free(local_A);
    free(local_T);

    MPI_Finalize();
    return 0;
}