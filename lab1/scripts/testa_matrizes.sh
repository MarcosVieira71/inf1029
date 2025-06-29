#!/bin/bash

mkdir -p ../arquivos

BINARIO="../build/matrix_lib_test"
ARQUIVOS="../arquivos"
CSV_RESULTADOS_ESCALAR="../resultados/resultadoEscalar.csv"
CSV_RESULTADOS_MATRIZ_MATRIZ="../resultados/resultadoMatrizMatriz.csv"

ESCALAR=5.0

echo "linhas_m1,colunas_m1,tempo_scalar_ms,executavel" > "$CSV_RESULTADOS_ESCALAR"
echo "linhas_m1,colunas_m1,linhas_m2,colunas_m2,tempo_matriz_matriz_ms,executavel" > "$CSV_RESULTADOS_MATRIZ_MATRIZ"

dimensoesEscalar=(
    "64 64"                  
    "512 1024"
    "1024 1024"
    "4096 4096"
    "4096 8192"
    "8192 4096"
    "8192 8192" 
    "16384 4096"
    "4096 16384"
    "16384 16384"
    "32000 40000"
)

dimensoesMm=(
    "64 32"
    "64 64"
    "256 128"
    "256 256"
    "512 1024"
    "1024 2048"
    "2048 1024"
    "2048 2048"
    "3200 2048"
    "3200 3200"
    "4096 4096"
)


executaveisMatrizesMatrizes=(
    "../build/matrix_lib_test_mm_trad"
    "../build/matrix_lib_test_mm_opt"
    "../build/matrix_lib_test_mm_threads"
)

executaveisEscalares=(
    "../build/matrix_lib_test_scalar_ptr"
    "../build/matrix_lib_test_scalar_threads"
)


#EXECUTA PARA MULTIPLICACAO DE MATRIZES
for exec in "${executaveisMatrizesMatrizes[@]}"; do
    for dimensao in "${dimensoesMm[@]}"; do
        for i in $(seq 1 3); do
            set -- $dimensao
            linhas_m1=$1
            colunas_m1=$2
            linhas_m2=$colunas_m1
            colunas_m2=$linhas_m1

            matriz1="$ARQUIVOS/matrix_${linhas_m1}x${colunas_m1}.dat"
            matriz2="$ARQUIVOS/matrix_${linhas_m2}x${colunas_m2}.dat"
            result1="$ARQUIVOS/result1.dat"
            result2="$ARQUIVOS/result2.dat"

            echo "Executando teste com matrizes ${linhas_m1}x${colunas_m1} e ${linhas_m2}x${colunas_m2} para $(basename $exec)..."

            output=$($exec -s "$ESCALAR" -r "$linhas_m1" -c "$colunas_m1" -C "$colunas_m2" -m "$matriz1" -M "$matriz2" -o "$result1" -O "$result2")

            tempo_matriz=$(echo "$output" | grep "Matrix product" | awk '{print $(NF-1)}')

            if [ -z "$tempo_matriz" ]; then
                tempo_matriz="erro"
            fi

            echo "$linhas_m1,$colunas_m1,$linhas_m2,$colunas_m2,$tempo_matriz,$(basename $exec)" >> "$CSV_RESULTADOS_MATRIZ_MATRIZ"
        done
    done
done


# EXECUTA PARA ESCALARES
for exec in "${executaveisEscalares[@]}"; do
    for dimensao in "${dimensoesEscalar[@]}"; do
        for i in $(seq 1 3); do
            set -- $dimensao
            linhas_m1=$1
            colunas_m1=$2
            linhas_m2=$colunas_m1
            colunas_m2=$linhas_m1

            matriz1="$ARQUIVOS/matrix_${linhas_m1}x${colunas_m1}.dat"
            matriz2="$ARQUIVOS/matrix_${linhas_m2}x${colunas_m2}.dat"
            result1="$ARQUIVOS/result1.dat"
            result2="$ARQUIVOS/result2.dat"

            echo "Executando teste com matrizes ${linhas_m1}x${colunas_m1} para $(basename $exec)..."

            output=$($exec -s "$ESCALAR" -r "$linhas_m1" -c "$colunas_m1" -C "$colunas_m2" -m "$matriz1" -M "$matriz2" -o "$result1" -O "$result2")

            tempo_scalar=$(echo "$output" | grep "Scalar product" | awk '{print $(NF-1)}')
            if [ -z "$tempo_scalar" ]; then
                tempo_scalar="erro"
            fi
            echo "$linhas_m1,$colunas_m1,$tempo_scalar,$(basename $exec)" >> "$CSV_RESULTADOS_ESCALAR"
        done
    done
done

