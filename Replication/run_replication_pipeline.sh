for i in {1..4}; do

    out_dir="Replication/Depth1_Run$i/"
    mkdir -p $out_dir 
    nohup python run_pipeline.py --depth 1 --output $out_dir > $out_dir/pipeline.out 2>&1

done

for i in {1..4}; do

    out_dir="Replication/Depth2_Run$i/"
    mkdir -p $out_dir 
    nohup python run_pipeline.py --depth 2 --output $out_dir > $out_dir/pipeline.out 2>&1

done

