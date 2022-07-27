import os
os.system("CUDA_VISIBLE_DEVICES=0,1 \
    		python3 main.py \
			--phase guide \
			--dataset AGLIE \
			--guide_num 130\
			--batch_size 1 \
			--direction a2b") 