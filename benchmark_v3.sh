#!/bin/bash
# File: benchmark_v3.sh
# Scopo: pipeline SLURM per compilare serial/OMP/MPI-ibrido, validare il checksum ed eseguire benchmark strong/weak e test ibrido.
# Ruolo: genera dati di scaling e verifica la correttezza con output CSV già parsabile.
#SBATCH --job-name=final_hpc_run
#SBATCH --output=benchmark_v3.out
#SBATCH --error=benchmark_v3.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --time=00:25:00
#SBATCH --partition=g100_usr_prod
#SBATCH --account=tra25_inginfbo

# === 1. CARICAMENTO MODULI ===
module load openmpi
module load openmp

# === 2. INFO HARDWARE ===
echo "=========================================="
echo "          INFO SISTEMA & HARDWARE         "
echo "=========================================="
echo "Cluster Node: $(hostname)"
echo "CPU Info:"
lscpu | grep -E "Model name|Socket|Thread|Core|MHz"
echo "RAM Info:"
free -h
echo "=========================================="

# === 3. COMPILAZIONE ===
echo "--> Compilazione in corso..."
gcc -o serial serial.c -O3
mpicc -fopenmp -o main_mpi main_mpi_final.c -O3
gcc -fopenmp -o main_omp main_omp.c -O3
echo "Compilazione terminata."
echo ""

# === 4. VALIDAZIONE (CHECKSUM) ===
echo "=========================================="
echo "          FASE DI VALIDAZIONE             "
echo "=========================================="
N_VAL=1000
echo "Generazione baseline SERIALE (N=$N_VAL)..."
./serial $N_VAL > serial_val.txt

echo "Generazione confronto MPI (N=$N_VAL, 4 Processi)..."
export OMP_NUM_THREADS=1
mpirun -np 4 ./main_mpi $N_VAL > mpi_val.txt

# Estrazione Checksum (Assumendo che il codice stampi 'Checksum: 12345')
SUM_SER=$(grep "Checksum:" serial_val.txt | awk '{print $2}')
SUM_MPI=$(grep "Checksum:" mpi_val.txt | awk '{print $2}')

echo "Checksum Seriale: $SUM_SER"
echo "Checksum MPI:     $SUM_MPI"

if [ "$SUM_SER" == "$SUM_MPI" ]; then
    echo ">>> VALIDAZIONE SUPERATA: I risultati coincidono! <<<"
else
    echo ">>> ATTENZIONE: I risultati NON coincidono! <<<"
fi
echo ""

# Intestazione unica per i dati CSV
echo "CSV_DATA_HEADER: Tipo,N,Processi,Thread,Tempo(s)"

# === 5. BENCHMARK OPENMP STRONG SCALING ===
# N fisso, thread variabili
echo "--> Esecuzione OpenMP Strong Scaling..."
N_OMP=4000
for T in 1 4 8 16 32; do
    export OMP_NUM_THREADS=$T
    TIME=$(./main_omp $N_OMP | grep "Tempo:" | awk '{print $2}')
    echo "CSV_DATA: OMP_STRONG,$N_OMP,1,$T,$TIME"
done

# === 6. BENCHMARK MPI STRONG SCALING ===
# N fisso, processi variabili
echo "--> Esecuzione MPI Strong Scaling..."
N_STRONG=4000
export OMP_NUM_THREADS=1
for P in 1 2 4 8 16 32; do
    TIME=$(mpirun -np $P --map-by core ./main_mpi $N_STRONG | grep "Tempo:" | awk '{print $2}')
    echo "CSV_DATA: MPI_STRONG,$N_STRONG,$P,1,$TIME"
done

# === 7. BENCHMARK MPI WEAK SCALING ===
# Carico costante per processo
echo "--> Esecuzione MPI Weak Scaling..."
export OMP_NUM_THREADS=1
# 1 Proc -> N=2000
T1=$(mpirun -np 1 ./main_mpi 2000 | grep "Tempo:" | awk '{print $2}')
echo "CSV_DATA: MPI_WEAK,2000,1,1,$T1"
# 4 Proc -> N=4000
T4=$(mpirun -np 4 ./main_mpi 4000 | grep "Tempo:" | awk '{print $2}')
echo "CSV_DATA: MPI_WEAK,4000,4,1,$T4"
# 16 Proc -> N=8000
T16=$(mpirun -np 16 ./main_mpi 8000 | grep "Tempo:" | awk '{print $2}')
echo "CSV_DATA: MPI_WEAK,8000,16,1,$T16"
# === 8. TEST IBRIDO ===
echo "--> Esecuzione Test Ibrido..."
# 4 processi MPI x 8 thread OMP
export OMP_NUM_THREADS=8

# Salviamo l'output per leggere eventuali errori; mapping semplice per evitare problemi di binding su nodi eterogenei
mpirun -np 4 --bind-to none ./main_mpi 4000 > hybrid_log.txt 2>&1

# Stampiamo il log nel file di output così leggiamo eventuali errori
cat hybrid_log.txt

# Estraiamo il tempo dal file
TIME_HYB=$(grep "Tempo:" hybrid_log.txt | awk '{print $2}')
echo "CSV_DATA: HYBRID_4x8,4000,4,8,$TIME_HYB"