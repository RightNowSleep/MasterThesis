# Tasks

## Phase 1: Models Directory (Core RoPE Implementation)

- [x] Task 1: Standardize models/pe_llama.py comments
  - [x] 1.1: Review and standardize class docstrings (LlamaRotaryEmbedding and all variants)
  - [x] 1.2: Review and standardize method docstrings (__init__, forward, _set_cos_sin_cache, etc.)
  - [x] 1.3: Review and standardize helper function docstrings (_layer_aware_attn_scale)
  - [x] 1.4: Translate any Chinese comments to English (lines 1867-1895 have Chinese comments)
  - [x] 1.5: Ensure inline comments match code logic

- [x] Task 2: Standardize models/modeling_llama.py comments
  - [x] 2.1: Review and standardize class docstrings (LlamaRMSNorm, LlamaMLP, LlamaAttention, etc.)
  - [x] 2.2: Review and standardize method docstrings
  - [x] 2.3: Ensure _init_rope docstring is accurate
  - [x] 2.4: Review inline comments for accuracy

- [x] Task 3: Standardize models/configuration_llama.py comments
  - [x] 3.1: Review LlamaConfig class docstring
  - [x] 3.2: Review _rope_scaling_validation method docstring
  - [x] 3.3: Ensure parameter descriptions are accurate

- [x] Task 4: Standardize models/model_loader.py comments
  - [x] 4.1: Review and standardize function docstrings
  - [x] 4.2: Review inline comments

- [x] Task 5: Standardize models/entropy_stable_scaled_rope.py comments
  - [x] 5.1: Review and standardize class docstrings
  - [x] 5.2: Review and standardize method docstrings

## Phase 2: Training Scripts

- [x] Task 6: Standardize continued_pretrain.py comments
  - [x] 6.1: Review function docstrings (find_all_linear_names, make_collate_fn, etc.)
  - [x] 6.2: Review main function section comments
  - [x] 6.3: Review add_args_continued_pretrain docstrings

- [x] Task 7: Standardize finetune.py comments
  - [x] 7.1: Review add_args_finetune docstring
  - [x] 7.2: Review main function docstring and section comments

## Phase 3: Evaluation Scripts

- [x] Task 8: Standardize eval/ directory Python files
  - [x] 8.1: Standardize eval/quality.py comments
  - [x] 8.2: Standardize eval/perplexity.py comments
  - [x] 8.3: Standardize eval/performance.py comments
  - [x] 8.4: Standardize eval/passkey.py comments
  - [x] 8.5: Standardize eval/plot_entropy.py comments
  - [x] 8.6: Standardize eval/entropy.py comments
  - [x] 8.7: Standardize eval/eval_harness.py comments

## Phase 4: Drawing/Visualization Scripts

- [x] Task 9: Standardize drawer/ directory Python files
  - [x] 9.1: Standardize drawer/position_heatmap.py comments
  - [x] 9.2: Standardize drawer/block_sum.py comments

- [x] Task 10: Standardize root drawing scripts
  - [x] 10.1: Standardize draw_rope_heatmap.py comments
  - [x] 10.2: Standardize draw_attention_scaling.py comments
  - [x] 10.3: Standardize draw.py comments
  - [x] 10.4: Standardize draw_position.py comments

## Phase 5: Utility Scripts

- [x] Task 11: Standardize remaining Python files
  - [x] 11.1: Standardize test.py comments
  - [x] 11.2: Standardize performance_compare.py comments
  - [x] 11.3: Standardize clean_json.py comments

## Phase 6: Shell Scripts

- [x] Task 12: Standardize all .sh files
  - [x] 12.1: Standardize permission.sh comments
  - [x] 12.2: Standardize finetune.sh comments
  - [x] 12.3: Standardize test.sh comments
  - [x] 12.4: Standardize continued_pretrain.sh comments
  - [x] 12.5: Standardize eval2.sh comments
  - [x] 12.6: Standardize entropy.sh comments
  - [x] 12.7: Standardize eval1.sh comments
  - [x] 12.8: Standardize eval_harness.sh comments
  - [x] 12.9: Standardize eval.sh comments

## Phase 7: Final Verification

- [x] Task 13: Verify all files have been processed
  - [x] 13.1: Confirm no Chinese comments remain
  - [x] 13.2: Confirm all docstrings follow Google style
  - [x] 13.3: Confirm all comments match code logic
  - [x] 13.4: Generate diff summary of all changes

# Task Dependencies
- Tasks 1-5 can run in parallel (all in models/ directory)
- Tasks 6-7 can run in parallel (training scripts)
- Tasks 8-11 can run in parallel (evaluation and utility scripts)
- Task 12 can run in parallel with Tasks 8-11
- Task 13 depends on all previous tasks being completed
