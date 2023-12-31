python label2vec.py \
--train_path     ./xmc-base/amazon-670k/Y.trn.txt \
--save_path    ./xmc-base/amazon-670k/label_embed.npy \
--mode         1 \
--sample       0.1 \
--ns_exponent  0.5 \
--alpha        2.5e-2 \
--alpha_min    1e-4  \
--emb_size     100 \
--negative     20  \
--epochs       50  \
--win_size     7  \
--seed         1   \
--num_label    670091
