root=/work02/home/wlluo/halflife
project=$root/experiment/MGA
exp=$project/TDCpretrainedpk1HL
subexp=$exp/fix_graph

export PYTHONPATH=$root:$PYTHONPATH
export PYTHONPATH=$project:$PYTHONPATH
export PYTHONPATH=$exp:$PYTHONPATH
cd $root
nohup python -u $subexp/main.py --config="$exp/config.json" > $subexp/log 2>&1 &