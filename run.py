import os
os.system("CUDA_VISIBLE_DEVICES=0,1 \
    		python3 main.py \
			--phase guide \
			--dataset guide_show \
			--guide_num 128\
			--batch_size 1 \
			--direction a2b")