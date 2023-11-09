python -m scripts.visualize \
--model 13-03_VIGIL4_WizardGuide_lang64_mm_baby_short_rec_env_MiniGrid-GoToDoorTalkHardSesameWizardGuideLang64-8x8-v0_multi-modal-babyai11-agent_arch_original_endpool_res_custom-ppo-2_exploration-bonus-params_5_50/0 \
--episodes 3 --seed=5 --gif graphics/gifs/MH-BabyAI-EB-Ablation --pause 0.2
python -m scripts.visualize \
--model 13-03_VIGIL4_WizardGuide_lang64_mm_baby_short_rec_env_MiniGrid-GoToDoorTalkHardSesameWizardGuideLang64-8x8-v0_multi-modal-babyai11-agent_arch_original_endpool_res_custom-ppo-2_exploration-bonus-params_5_50/0 \
--episodes 3 --seed=5 --gif graphics/gifs/MH-BabyAI-EB-Ablation-Deterministic --pause 0.2 --argmax
python -m scripts.visualize \
--model 13-03_VIGIL4_WizardTwoGuides_lang64_mm_baby_short_rec_env_MiniGrid-GoToDoorTalkHardSesameNPCGuidesLang64-8x8-v0_multi-modal-babyai11-agent_arch_original_endpool_res_custom-ppo-2_exploration-bonus-params_5_50/0 \
--episodes 3 --seed=5 --gif graphics/gifs/MH-BabyAI-EB-Original --pause 0.2
python -m scripts.visualize \
--model 13-03_VIGIL4_WizardTwoGuides_lang64_mm_baby_short_rec_env_MiniGrid-GoToDoorTalkHardSesameNPCGuidesLang64-8x8-v0_multi-modal-babyai11-agent_arch_original_endpool_res_custom-ppo-2_exploration-bonus-params_5_50/0 \
--episodes 3 --seed=5 --gif graphics/gifs/MH-BabyAI-EB-Original-Deterministic --pause 0.2 --argmax
# no explo
python -m scripts.visualize \
--model 13-03_VIGIL4_WizardGuide_lang64_no_explo_mm_baby_short_rec_env_MiniGrid-GoToDoorTalkHardSesameWizardGuideLang64-8x8-v0_multi-modal-babyai11-agent_arch_original_endpool_res_custom-ppo-2/0 \
--episodes 3 --seed=5 --gif graphics/gifs/MH-BabyAI-Ablation --pause 0.2
python -m scripts.visualize \
--model 13-03_VIGIL4_WizardGuide_lang64_no_explo_mm_baby_short_rec_env_MiniGrid-GoToDoorTalkHardSesameWizardGuideLang64-8x8-v0_multi-modal-babyai11-agent_arch_original_endpool_res_custom-ppo-2/0 \
--episodes 3 --seed=5 --gif graphics/gifs/MH-BabyAI-Ablation-Deterministic --pause 0.2 --argmax
python -m scripts.visualize \
--model 13-03_VIGIL4_WizardTwoGuides_lang64_no_explo_mm_baby_short_rec_env_MiniGrid-GoToDoorTalkHardSesameNPCGuidesLang64-8x8-v0_multi-modal-babyai11-agent_arch_original_endpool_res_custom-ppo-2/0 \
--episodes 3 --seed=5 --gif graphics/gifs/MH-BabyAI-Original --pause 0.2
python -m scripts.visualize \
--model 13-03_VIGIL4_WizardTwoGuides_lang64_no_explo_mm_baby_short_rec_env_MiniGrid-GoToDoorTalkHardSesameNPCGuidesLang64-8x8-v0_multi-modal-babyai11-agent_arch_original_endpool_res_custom-ppo-2/0 \
--episodes 3 --seed=5 --gif graphics/gifs/MH-BabyAI-Original-Deterministic --pause 0.2 --argmax
