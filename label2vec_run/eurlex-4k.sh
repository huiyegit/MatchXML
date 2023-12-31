python label2vec.py \
--train_path     ./xmc-base/eurlex-4k/Y.trn.txt \
--save_path    ./xmc-base/eurlex-4k/label_embed.npy \
--mode         1 \
--sample       0.1 \
--ns_exponent  0.5 \
--alpha        2.5e-2 \
--alpha_min    1e-4  \
--emb_size     100 \
--negative     20  \
--epochs       20  \
--win_size     24  \
--seed         1   \
--num_label    3956
