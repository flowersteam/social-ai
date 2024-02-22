# Repeat for different seeds
SEED=1

######################
## Scaffolding + Formats
######################
python -m scripts.train --frames 50000000 --model formats_50M_CBL/$SEED -seed $SEED  --dialogue --save-interval 100 --log-interval 100 --test-interval 1000 --frames-per-proc 40 --multi-modal-babyai11-agent --env SocialAI-AELangFeedbackTrainFormatsCSParamEnv-v1  --clipped-rewards --batch-size 640 --clip-eps 0.2 --recurrence 5 --max-grad-norm 0.5 --epochs 4 --optim-eps 1e-05 --lr 1e-4 --entropy-coef 0.00001 --test-set-name AEFormatsTestSet --env-args see_through_walls False --arch bow_endpool_res --bAI-lang-model attgru --memory-dim 2048 --procs 64 --exploration-bonus --episodic-exploration-bonus  --exploration-bonus-type lang --exploration-bonus-params  10 50 --exploration-bonus-tanh 0.6
python -m scripts.train --frames 50000000 --model scaffolding_50M_no_acl/$SEED -seed $SEED  --dialogue --save-interval 100 --log-interval 100 --test-interval 1000 --frames-per-proc 40 --multi-modal-babyai11-agent --env SocialAI-AELangFeedbackTrainFormatsCSParamEnv-v1  --clipped-rewards --batch-size 640 --clip-eps 0.2 --recurrence 5 --max-grad-norm 0.5 --epochs 4 --optim-eps 1e-05 --lr 1e-4 --entropy-coef 0.00001 --test-set-name AEFormatsTestSet --env-args see_through_walls False --arch bow_endpool_res --bAI-lang-model attgru --memory-dim 2048 --procs 64
python -m scripts.train --frames 50000000 --model scaffolding_50M_acl_4/$SEED -seed $SEED  --dialogue --save-interval 100 --log-interval 100 --test-interval 1000 --frames-per-proc 40 --multi-modal-babyai11-agent --env SocialAI-AELangFeedbackTrainScaffoldingCSParamEnv-v1 --clipped-rewards --batch-size 640 --clip-eps 0.2 --recurrence 5 --max-grad-norm 0.5 --epochs 4 --optim-eps 1e-05 --lr 1e-4 --entropy-coef 0.00001 --test-set-name AEFormatsTestSet --env-args see_through_walls False --arch bow_endpool_res --bAI-lang-model attgru --memory-dim 2048 --procs 64 --acl --*acl-type intro_seq --acl-thresholds 0.90 0.90 0.90 0.90 --acl-average-interval 500  --acl-minimum-episodes 1000
python -m scripts.train --frames 50000000 --model scaffolding_50M_acl_8/$SEED -seed $SEED  --dialogue --save-interval 100 --log-interval 100 --test-interval 1000 --frames-per-proc 40 --multi-modal-babyai11-agent --env SocialAI-AELangFeedbackTrainScaffoldingCSParamEnv-v1 --clipped-rewards --batch-size 640 --clip-eps 0.2 --recurrence 5 --max-grad-norm 0.5 --epochs 4 --optim-eps 1e-05 --lr 1e-4 --entropy-coef 0.00001 --test-set-name AEFormatsTestSet --env-args see_through_walls False --arch bow_endpool_res --bAI-lang-model attgru --memory-dim 2048 --procs 64 --acl --*acl-type intro_seq_scaf --acl-thresholds 0.90 0.90 0.90 0.90 --acl-average-interval 500  --acl-minimum-episodes 1000

###############
## Pointing
###############
python -m scripts.train --frames 50000000 --model Pointing_CB_heldout_doors/$SEED --seed $SEED  --dialogue --save-interval 100 --log-interval 100 --test-interval 1000 --frames-per-proc 40 --multi-modal-babyai11-agent --env SocialAI-EPointingHeldoutDoorsTrainInformationSeekingParamEnv-v1 --clipped-rewards --batch-size 640 --clip-eps 0.2 --recurrence 5 --max-grad-norm 0.5 --epochs 4 --optim-eps 1e-05 --lr 1e-4 --entropy-coef 0.00001 --test-set-name PointingTestSet --env-args see_through_walls False --arch bow_endpool_res --bAI-lang-model attgru --memory-dim 2048 --procs 64 --exploration-bonus --episodic-exploration-bonus  --exploration-bonus-type cell --exploration-bonus-params  0.25 50 --exploration-bonus-tanh 0.6

###############
## Language Feedback
###############
python -m scripts.train --frames 20000000 --model Feedback_CB_heldout_doors_20M/$SEED --seed $SEED   --dialogue --save-interval 100 --log-interval 100 --test-interval 1000 --frames-per-proc 40 --multi-modal-babyai11-agent --env SocialAI-ELangFeedbackHeldoutDoorsTrainInformationSeekingParamEnv-v1 --clipped-rewards --batch-size 640 --clip-eps 0.2 --recurrence 5 --max-grad-norm 0.5 --epochs 4 --optim-eps 1e-05 --lr 1e-4 --entropy-coef 0.00001 --test-set-name LangFeedbackTestSet --env-args see_through_walls False --arch bow_endpool_res --bAI-lang-model attgru --memory-dim 2048 --procs 64 --exploration-bonus --episodic-exploration-bonus  --exploration-bonus-type lang --exploration-bonus-params  10 50 --exploration-bonus-tanh 0.6

###############
## Language Color
###############
python -m scripts.train --frames 20000000 --model Color_CB_heldout_doors/$SEED --seed $SEED   --dialogue --save-interval 100 --log-interval 100 --test-interval 1000 --frames-per-proc 40 --multi-modal-babyai11-agent --env SocialAI-ELangColorHeldoutDoorsTrainInformationSeekingParamEnv-v1 --clipped-rewards --batch-size 640 --clip-eps 0.2 --recurrence 5 --max-grad-norm 0.5 --epochs 4 --optim-eps 1e-05 --lr 1e-4 --entropy-coef 0.00001 --test-set-name LangColorTestSet --env-args see_through_walls False --arch bow_endpool_res --bAI-lang-model attgru --memory-dim 2048 --procs 64 --exploration-bonus --episodic-exploration-bonus  --exploration-bonus-type lang --exploration-bonus-params  10 50 --exploration-bonus-tanh 0.6

###############
## Joint attention (Color)
###############
python -m scripts.train --frames 20000000 --model JA_Color_CB_heldout_doors/$SEED --seed $SEED  --dialogue --save-interval 100 --log-interval 100 --test-interval 1000 --frames-per-proc 40 --multi-modal-babyai11-agent --env SocialAI-JAELangColorHeldoutDoorsTrainInformationSeekingParamEnv-v1 --clipped-rewards --batch-size 640 --clip-eps 0.2 --recurrence 5 --max-grad-norm 0.5 --epochs 4 --optim-eps 1e-05 --lr 1e-4 --entropy-coef 0.00001 --test-set-name JALangColorTestSet --env-args see_through_walls False --arch bow_endpool_res --bAI-lang-model attgru --memory-dim 2048 --procs 64 --exploration-bonus --episodic-exploration-bonus  --exploration-bonus-type lang --exploration-bonus-params  10 50 --exploration-bonus-tanh 0.6


###############
## Imitation
###############
python -m scripts.train --frames 20000000 --model Imitation_PPO_CB/$SEED --seed $SEED  --dialogue --save-interval 10 --log-interval 10 --test-interval 100 --frames-per-proc 40 --multi-modal-babyai11-agent --env SocialAI-EEmulationNoDistrInformationSeekingParamEnv-v1 --clipped-rewards --batch-size 640 --clip-eps 0.2 --recurrence 5 --max-grad-norm 0.5 --epochs 4 --optim-eps 1e-05 --lr 1e-4 --entropy-coef 0.00001 --env-args see_through_walls False --arch bow_endpool_res --bAI-lang-model attgru --memory-dim 2048 --procs 64 --test-set-name NoDistrEmulationTestSet --exploration-bonus --episodic-exploration-bonus  --*exploration-bonus-type cell --*exploration-bonus-params  0.25 50 --exploration-bonus-tanh 0.6
python -m scripts.train --frames 20000000 --model Imitation_PPO_CB/$SEED --seed $SEED  --dialogue --save-interval 10 --log-interval 10 --test-interval 100 --frames-per-proc 40 --multi-modal-babyai11-agent --env SocialAI-EEmulationNoDistrInformationSeekingParamEnv-v1 --clipped-rewards --batch-size 640 --clip-eps 0.2 --recurrence 5 --max-grad-norm 0.5 --epochs 4 --optim-eps 1e-05 --lr 1e-4 --entropy-coef 0.00001 --env-args see_through_walls False --arch bow_endpool_res --bAI-lang-model attgru --memory-dim 2048 --procs 64 --test-set-name NoDistrEmulationTestSet --exploration-bonus --episodic-exploration-bonus  --*exploration-bonus-type cell --*exploration-bonus-params  0.5 50 --exploration-bonus-tanh 0.6
python -m scripts.train --frames 20000000 --model Imitation_PPO_CB/$SEED --seed $SEED  --dialogue --save-interval 10 --log-interval 10 --test-interval 100 --frames-per-proc 40 --multi-modal-babyai11-agent --env SocialAI-EEmulationNoDistrInformationSeekingParamEnv-v1 --clipped-rewards --batch-size 640 --clip-eps 0.2 --recurrence 5 --max-grad-norm 0.5 --epochs 4 --optim-eps 1e-05 --lr 1e-4 --entropy-coef 0.00001 --env-args see_through_walls False --arch bow_endpool_res --bAI-lang-model attgru --memory-dim 2048 --procs 64 --test-set-name NoDistrEmulationTestSet --exploration-bonus --episodic-exploration-bonus  --*exploration-bonus-type cell --*exploration-bonus-params  1 50 --exploration-bonus-tanh 0.6

##################
## Role Reversal
##################

## SINGLE
##################

# pretrain
python -m scripts.train --frames 4000000 --model RR_single_CB_marble_pass_B_exp/$SEED --seed $SEED  --dialogue --save-interval 10 --log-interval 10 --test-interval 0 --frames-per-proc 40 --multi-modal-babyai11-agent --env SocialAI-MarblePassBCollaborationParamEnv-v1 --clipped-rewards --batch-size 640 --clip-eps 0.2 --recurrence 5 --max-grad-norm 0.5 --epochs 4 --optim-eps 1e-05 --lr 1e-4 --entropy-coef 0.00001 --test-set-name RoleReversalTestSet --env-args see_through_walls False --arch bow_endpool_res --bAI-lang-model attgru --memory-dim 2048 --procs 64 --exploration-bonus --episodic-exploration-bonus  --exploration-bonus-type cell --exploration-bonus-params  2 50 --exploration-bonus-tanh 0.6
python -m scripts.train --frames 4000000 --model RR_single_CB_marble_pass_asoc_contr/$SEED --seed $SEED  --dialogue --save-interval 10 --log-interval 10 --test-interval 0 --frames-per-proc 40 --multi-modal-babyai11-agent --env SocialAI-AsocialMarbleCollaborationParamEnv-v1 --clipped-rewards --batch-size 640 --clip-eps 0.2 --recurrence 5 --max-grad-norm 0.5 --epochs 4 --optim-eps 1e-05 --lr 1e-4 --entropy-coef 0.00001 --test-set-name RoleReversalTestSet --env-args see_through_walls False --arch bow_endpool_res --bAI-lang-model attgru --memory-dim 2048 --procs 64 --exploration-bonus --episodic-exploration-bonus  --exploration-bonus-type cell --exploration-bonus-params  2 50 --exploration-bonus-tanh 0.6

# finetune
python -m scripts.train --frames 1000000 --model RR_ft_single_CB_marble_pass_A_soc_exp/$SEED --seed $SEED  --dialogue --save-interval 1 --log-interval 1 --test-interval 0 --frames-per-proc 40 --multi-modal-babyai11-agent --env SocialAI-MarblePassACollaborationParamEnv-v1 --clipped-rewards --batch-size 640 --clip-eps 0.2 --recurrence 5 --max-grad-norm 0.5 --epochs 4 --optim-eps 1e-05 --lr 1e-4 --entropy-coef 0.00001 --test-set-name RoleReversalTestSet --env-args see_through_walls False --arch bow_endpool_res --bAI-lang-model attgru --memory-dim 2048 --procs 64 --exploration-bonus --episodic-exploration-bonus  --exploration-bonus-type cell --exploration-bonus-params  2 50 --exploration-bonus-tanh 0.6 --finetune-train storage/02-01_RR_single_CB_marble_pass_B_exp
python -m scripts.train --frames 1000000 --model RR_ft_single_CB_marble_pass_A_asoc_contr/$SEED --seed $SEED  --dialogue --save-interval 1 --log-interval 1 --test-interval 0 --frames-per-proc 40 --multi-modal-babyai11-agent --env SocialAI-MarblePassACollaborationParamEnv-v1 --clipped-rewards --batch-size 640 --clip-eps 0.2 --recurrence 5 --max-grad-norm 0.5 --epochs 4 --optim-eps 1e-05 --lr 1e-4 --entropy-coef 0.00001 --test-set-name RoleReversalTestSet --env-args see_through_walls False --arch bow_endpool_res --bAI-lang-model attgru --memory-dim 2048 --procs 64 --exploration-bonus --episodic-exploration-bonus  --exploration-bonus-type cell --exploration-bonus-params  2 50 --exploration-bonus-tanh 0.6 --finetune-train storage/02-01_RR_single_CB_marble_pass_asoc_contr

## GROUP
##################

# pretrain
python -m scripts.train --frames 50000000 --model RR_group_CB_marble_pass_B_exp/$SEED --seed $SEED  --dialogue --save-interval 10 --log-interval 10 --test-interval 100 --frames-per-proc 40 --multi-modal-babyai11-agent --env SocialAI-RoleReversalGroupExperimentalCollaborationParamEnv-v1 --clipped-rewards --batch-size 640 --clip-eps 0.2 --recurrence 5 --max-grad-norm 0.5 --epochs 4 --optim-eps 1e-05 --lr 1e-4 --entropy-coef 0.00001 --test-set-name RoleReversalTestSet --env-args see_through_walls False --arch bow_endpool_res --bAI-lang-model attgru --memory-dim 2048 --procs 64 --exploration-bonus --episodic-exploration-bonus  --exploration-bonus-type cell --exploration-bonus-params  2 50 --exploration-bonus-tanh 0.6
python -m scripts.train --frames 50000000 --model RR_group_CB_marble_pass_asoc_contr/$SEED --seed $SEED  --dialogue --save-interval 10 --log-interval 10 --test-interval 100 --frames-per-proc 40 --multi-modal-babyai11-agent --env SocialAI-RoleReversalGroupControlCollaborationParamEnv-v1 --clipped-rewards --batch-size 640 --clip-eps 0.2 --recurrence 5 --max-grad-norm 0.5 --epochs 4 --optim-eps 1e-05 --lr 1e-4 --entropy-coef 0.00001 --test-set-name RoleReversalTestSet --env-args see_through_walls False --arch bow_endpool_res --bAI-lang-model attgru --memory-dim 2048 --procs 64 --exploration-bonus --episodic-exploration-bonus  --exploration-bonus-type cell --exploration-bonus-params  2 50 --exploration-bonus-tanh 0.6

# finetune
python -m scripts.train --frames 500000 --model RR_ft_group_20M_CB_marble_pass_A_soc_exp/$SEED --seed $SEED  --dialogue --save-interval 1 --log-interval 1 --test-interval 0 --frames-per-proc 40 --multi-modal-babyai11-agent --env SocialAI-MarblePassACollaborationParamEnv-v1 --clipped-rewards --batch-size 640 --clip-eps 0.2 --recurrence 5 --max-grad-norm 0.5 --epochs 4 --optim-eps 1e-05 --lr 1e-4 --entropy-coef 0.00001 --test-set-name RoleReversalTestSet --env-args see_through_walls False --arch bow_endpool_res --bAI-lang-model attgru --memory-dim 2048 --procs 64 --exploration-bonus --episodic-exploration-bonus  --exploration-bonus-type cell --exploration-bonus-params  2 50 --exploration-bonus-tanh 0.6 --finetune-train storage/02-01_RR_group_CB_marble_pass_B_exp
python -m scripts.train --frames 500000 --model RR_ft_group_20M_CB_marble_pass_A_asoc_contr/$SEED --seed $SEED  --dialogue --save-interval 1 --log-interval 1 --test-interval 0 --frames-per-proc 40 --multi-modal-babyai11-agent --env SocialAI-MarblePassACollaborationParamEnv-v1 --clipped-rewards --batch-size 640 --clip-eps 0.2 --recurrence 5 --max-grad-norm 0.5 --epochs 4 --optim-eps 1e-05 --lr 1e-4 --entropy-coef 0.00001 --test-set-name RoleReversalTestSet --env-args see_through_walls False --arch bow_endpool_res --bAI-lang-model attgru --memory-dim 2048 --procs 64 --exploration-bonus --episodic-exploration-bonus  --exploration-bonus-type cell --exploration-bonus-params  2 50 --exploration-bonus-tanh 0.6 --finetune-train storage/02-01_RR_group_CB_marble_pass_asoc_contr

# finetune - 50M
python -m scripts.train --frames 500000 --model RR_ft_group_50M_CB_marble_pass_A_soc_exp/$SEED --seed $SEED  --dialogue --save-interval 1 --log-interval 1 --test-interval 0 --frames-per-proc 40 --multi-modal-babyai11-agent --env SocialAI-MarblePassACollaborationParamEnv-v1 --clipped-rewards --batch-size 640 --clip-eps 0.2 --recurrence 5 --max-grad-norm 0.5 --epochs 4 --optim-eps 1e-05 --lr 1e-4 --entropy-coef 0.00001 --test-set-name RoleReversalTestSet --env-args see_through_walls False --arch bow_endpool_res --bAI-lang-model attgru --memory-dim 2048 --procs 64 --exploration-bonus --episodic-exploration-bonus  --exploration-bonus-type cell --exploration-bonus-params  2 50 --exploration-bonus-tanh 0.6 --finetune-train storage/03-01_RR_group_CB_marble_pass_B_exp
python -m scripts.train --frames 500000 --model RR_ft_group_50M_CB_marble_pass_A_asoc_contr/$SEED --seed $SEED  --dialogue --save-interval 1 --log-interval 1 --test-interval 0 --frames-per-proc 40 --multi-modal-babyai11-agent --env SocialAI-MarblePassACollaborationParamEnv-v1 --clipped-rewards --batch-size 640 --clip-eps 0.2 --recurrence 5 --max-grad-norm 0.5 --epochs 4 --optim-eps 1e-05 --lr 1e-4 --entropy-coef 0.00001 --test-set-name RoleReversalTestSet --env-args see_through_walls False --arch bow_endpool_res --bAI-lang-model attgru --memory-dim 2048 --procs 64 --exploration-bonus --episodic-exploration-bonus  --exploration-bonus-type cell --exploration-bonus-params  2 50 --exploration-bonus-tanh 0.6 --finetune-train storage/03-01_RR_group_CB_marble_pass_asoc_contr

##################
## Adversarial type - AppleStealing
##################

python -m scripts.train --frames 2000000 --model Adversarial_2M_PPO_CB/$SEED --seed $SEED  --dialogue --save-interval 10 --log-interval 10 --test-interval 0 --frames-per-proc 40 --multi-modal-babyai11-agent --env SocialAI-AppleStealingObst_NoParamEnv-v1 --clipped-rewards --batch-size 640 --clip-eps 0.2 --recurrence 5 --max-grad-norm 0.5 --epochs 4 --optim-eps 1e-05 --lr 1e-4 --entropy-coef 0.00001 --test-set-name RoleReversalTestSet --env-args see_through_walls False --arch bow_endpool_res --bAI-lang-model attgru --memory-dim 2048 --procs 64 --exploration-bonus --episodic-exploration-bonus  --exploration-bonus-type cell --exploration-bonus-params  2 50 --exploration-bonus-tanh 0.6
python -m scripts.train --frames 2000000 --model Adversarial_2M_PPO_CB_hidden_npc/$SEED --seed $SEED  --dialogue --save-interval 10 --log-interval 10 --test-interval 0 --frames-per-proc 40 --multi-modal-babyai11-agent --env SocialAI-AppleStealingObst_NoParamEnv-v1 --clipped-rewards --batch-size 640 --clip-eps 0.2 --recurrence 5 --max-grad-norm 0.5 --epochs 4 --optim-eps 1e-05 --lr 1e-4 --entropy-coef 0.00001 --test-set-name RoleReversalTestSet --env-args see_through_walls False --arch bow_endpool_res --bAI-lang-model attgru --memory-dim 2048 --procs 64 --exploration-bonus --episodic-exploration-bonus  --exploration-bonus-type cell --exploration-bonus-params  2 50 --exploration-bonus-tanh 0.6 --env-args hidden_npc True
python -m scripts.train --frames 2000000 --model Adversarial_2M_PPO_CB_asoc/$SEED --seed $SEED  --dialogue --save-interval 10 --log-interval 10 --test-interval 0 --frames-per-proc 40 --multi-modal-babyai11-agent --env SocialAI-AsocialAppleStealingObst_NoParamEnv-v1 --clipped-rewards --batch-size 640 --clip-eps 0.2 --recurrence 5 --max-grad-norm 0.5 --epochs 4 --optim-eps 1e-05 --lr 1e-4 --entropy-coef 0.00001 --test-set-name RoleReversalTestSet --env-args see_through_walls False --arch bow_endpool_res --bAI-lang-model attgru --memory-dim 2048 --procs 64 --exploration-bonus --episodic-exploration-bonus  --exploration-bonus-type cell --exploration-bonus-params  2 50 --exploration-bonus-tanh 0.6

##################
# Adversarial type - AppleStealing - more stumps
##################

python -m scripts.train --frames 5000000 --model Adversarial_5M_Stumps_PPO_CB/$SEED --seed $SEED  --dialogue --save-interval 10 --log-interval 10 --test-interval 0 --frames-per-proc 40 --multi-modal-babyai11-agent --env SocialAI-AppleStealingObst_MediumParamEnv-v1 --clipped-rewards --batch-size 640 --clip-eps 0.2 --recurrence 5 --max-grad-norm 0.5 --epochs 4 --optim-eps 1e-05 --lr 1e-4 --entropy-coef 0.00001 --test-set-name RoleReversalTestSet --env-args see_through_walls False --arch bow_endpool_res --bAI-lang-model attgru --memory-dim 2048 --procs 64 --exploration-bonus --episodic-exploration-bonus  --exploration-bonus-type cell --exploration-bonus-params  2 50 --exploration-bonus-tanh 0.6
python -m scripts.train --frames 5000000 --model Adversarial_5M_Stumps_PPO_CB_hidden_npc/$SEED --seed $SEED  --dialogue --save-interval 10 --log-interval 10 --test-interval 0 --frames-per-proc 40 --multi-modal-babyai11-agent --env SocialAI-AppleStealingObst_MediumParamEnv-v1 --clipped-rewards --batch-size 640 --clip-eps 0.2 --recurrence 5 --max-grad-norm 0.5 --epochs 4 --optim-eps 1e-05 --lr 1e-4 --entropy-coef 0.00001 --test-set-name RoleReversalTestSet --env-args see_through_walls False --arch bow_endpool_res --bAI-lang-model attgru --memory-dim 2048 --procs 64 --exploration-bonus --episodic-exploration-bonus  --exploration-bonus-type cell --exploration-bonus-params  2 50 --exploration-bonus-tanh 0.6 --env-args hidden_npc True
python -m scripts.train --frames 5000000 --model Adversarial_5M_Stumps_PPO_CB_asoc/$SEED --seed $SEED  --dialogue --save-interval 10 --log-interval 10 --test-interval 0 --frames-per-proc 40 --multi-modal-babyai11-agent --env SocialAI-AsocialAppleStealingObst_MediumParamEnv-v1 --clipped-rewards --batch-size 640 --clip-eps 0.2 --recurrence 5 --max-grad-norm 0.5 --epochs 4 --optim-eps 1e-05 --lr 1e-4 --entropy-coef 0.00001 --test-set-name RoleReversalTestSet --env-args see_through_walls False --arch bow_endpool_res --bAI-lang-model attgru --memory-dim 2048 --procs 64 --exploration-bonus --episodic-exploration-bonus  --exploration-bonus-type cell --exploration-bonus-params  2 50 --exploration-bonus-tanh 0.6
