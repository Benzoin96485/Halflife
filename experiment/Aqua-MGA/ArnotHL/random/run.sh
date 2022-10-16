root=/work02/home/wlluo/halflife
project=$root/experiment/Aqua-MGA
exp=$project/ArnotHL
subexp=$exp/random
conda activate lwl_deepchem
export PYTHONPATH=$root:$PYTHONPATH
export PYTHONPATH=$project:$PYTHONPATH
export PYTHONPATH=$exp:$PYTHONPATH
cd $root
nohup python -u $subexp/main.py --config="$subexp/config.json" > $subexp/log 2>&1 &