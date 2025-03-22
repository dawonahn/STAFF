
# Demo for STAFF - Costco 
config_file=staff_costco
gpu=0

for data in oulad chicago_area lastfm_time; do
for i in 1 2 3 4 5; do # Random seed
python ./src/run_exp.py $config_file $data $gpu $i
done; done;