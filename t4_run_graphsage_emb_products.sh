python3 utils/launch_train.py --workspace ~/workspace/dgl-bench/ \
   --num_trainers 8 \
   --num_samplers 1 \
   --num_servers 2 \
   --part_config /home/ubuntu/workspace/datasets/products_4part/ogbn-products.json \
   --ip_config utils/ip_config4.txt \
   "~/workspace/embcache_venv/bin/python3 graphsage/train_dist_transductive_node_classification.py --num_hidden 256 --graph_name ogbn-products --ip_config utils/ip_config4.txt --num_gpus 8 --num_epochs 20 --log_acc --eval_acc --dgl_sparse"