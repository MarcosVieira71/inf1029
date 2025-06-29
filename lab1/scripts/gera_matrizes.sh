mkdir -p ../build && gcc  -Wall -o ../build/gera_matrix ../src/gera_matrix.c

linhasM1="40000"
colunasM1="40000"
colunasM2="40000"

if [ $# -ge 1 ]; then
    linhasM1=$1
fi
if [ $# -ge 2 ]; then
    colunasM1=$2
fi
if [ $# -ge 3 ]; then
    colunasM2=$3
fi

mkdir -p ../arquivos

arquivo1="../arquivos/matrix1.dat"
arquivo2="../arquivos/matrix2.dat"

../build/gera_matrix "$arquivo1" "$linhasM1" "$colunasM1"
../build/gera_matrix "$arquivo2" "$colunasM1" "$colunasM2"