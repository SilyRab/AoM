for sl in  '7.5e-5' #You can change the sl to find the best hyperparameter.
do
		echo ${sl}
		python3.7 MAESC_training.py \
          --dataset twitter15 ./src/data/jsons/twitter15_info.json \
          --checkpoint_dir ./train15 \
          --model_config config/pretrain_base.json \
          --log_dir 15_aesc \
          --num_beams 4 \
          --eval_every 1 \
          --lr ${sl} \
          --batch_size 16  \
          --epochs 35 \
          --grad_clip 5 \
          --warmup 0.1 \
          --seed 57 \
          --checkpoint ./checkpoint/pytorch_model.bin \
          --rank 2 \
          --trc_pretrain_file TRC_ckpt/pytorch_model.bin \
          --nn_attention_on \
          --nn_attention_mode 0\
          --trc_on \
          --gcn_on \
          --dep_mode 2 \
          --sentinet \
          --no_train
done
