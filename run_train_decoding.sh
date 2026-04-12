python train_hira.py \
--peft_type=hira \
--model= your model path \
--r_ab=16 \
--enable_grad_ckpt --epoch=3 --lr=1e-4 --batch=16 \
--dataset=common_170k --seed=36 \
--warmup=100 --eval_strategy=steps --eval_steps=80 \
--output_folder=results_hira --target_modules=q_proj,k_proj,v_proj,up_proj,down_proj