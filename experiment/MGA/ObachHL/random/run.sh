root=/work02/home/wlluo/halflife
project=$root/experiment/MGA
exp=$project/ObachHL
subexp=$exp/random
conda activate lwl_deepchem
export PYTHONPATH=$root:$PYTHONPATH
export PYTHONPATH=$project:$PYTHONPATH
export PYTHONPATH=$exp:$PYTHONPATH
cd $root
python -u $subexp/main.py --config="$subexp/config.json" > $subexp/log 2>&1 