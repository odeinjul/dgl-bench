python3 utils/launch_train.py --workspace ~/dgl-bench/ \
   --num_trainers 2 \
   --num_samplers 1 \
   --num_servers 2 \
   --part_config /data/distemb_datasets/ogbn-products-1part/ogbn-products.json \
   --ip_config utils/ip_config_local.txt \
   "~/venv/bin/python3 graphsage/train_dist.py --num_hidden 256 --graph_name ogbn-products --ip_config utils/ip_config_local.txt --num_gpus 2 --num_epochs 7 --eval_every 5"