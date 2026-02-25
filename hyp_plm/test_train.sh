export CUDA_VISIBLE_DEVICES=6

python scripts/test_train.py /datapool/data2/home/ruihan/code/reference/Hyperbolic-PLM/hyp_plm/uniref50/uniref50.fasta /datapool/data2/home/ruihan/code/reference/Hyperbolic-PLM/hyp_plm/output/debug \
  --toks_per_batch 4096 \
  --truncation_seq_length 1022