names=('GeomGCL' 'MGSSL' 'MG-BERT' 'PAR' 'Uni-Mol' 'MolCLR')
for name in ${names[*]}; do
    cd /work02/home/wlluo/halflife
    mkdir experiment/$name/
    cd experiment/$name/
    mkdir ArnotHL
    cd ArnotHL
    mkdir random
    touch random/run.sh
    mkdir scaffold
    touch scaffold/run.sh
    cd ..
    mkdir ObachHL
    cd ObachHL
    mkdir random
    touch random/run.sh
    mkdir scaffold
    touch scaffold/run.sh
done