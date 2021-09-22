# test points
#python  test_reg_igndata_block_BN1_point.py   --model pointnet_reg_sml1BN1_point  --log_dir reg_ign_block_sml1B1_point --batch_size 1 --npoint 100  --learning_rate 0.001 --decay_rate 1e-4 --step_size 100 --epoch 200000
## test rgb+ points
python  test_reg_igndata_block_BN1_point.py  --model pointnet_reg_sml1BN1_point  --log_dir rgn_point --batch_size 1 --npoint 100  --learning_rate 0.00001 --decay_rate 1e-5 --step_size 1 --epoch 10 --optimizer sgd
