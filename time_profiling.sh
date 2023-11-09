#!/bin/bash
rm -rf storage/test && python -m cProfile -o graphics/train.prof scripts/train.py --model test --seed 1 --algo ppo --dialogue --save-interval 100 --log-interval 1 --test-interval 0 --frames-per-proc 40 --multi-modal-babyai11-agent --env SocialAI-GridSearchParamEnv-v1 --clipped-rewards --batch-size 640 --clip-eps 0.2 --recurrence 5 --max-grad-norm 0.5 --epochs 4 --optim-eps 1e-05 --lr 0.0001 --entropy-coef 0.00001 --env-args see_through_walls True --arch original_endpool_res  --env-args max_steps 80  --frames 12800
#snakeviz train.prof
