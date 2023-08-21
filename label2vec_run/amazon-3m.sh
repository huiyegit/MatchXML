python label2vec.py \
--train_path     ./xmc-base/amazon-3m/Y.trn.txt \
--save_path    ./xmc-base/amazon-3m/label_embed.npy \
--mode         1 \
--sample       0.1 \
--ns_exponent  -0.5 \
--alpha        2.5e-2 \
--alpha_min    1e-4  \
--emb_size     100 \
--negative     20  \
--epochs       20  \
--win_size     100  \
--seed         1   \
--num_label    2812281