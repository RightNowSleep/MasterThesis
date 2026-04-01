# Comment Standardization Spec

## Why
The codebase contains mixed English and Chinese comments, with some comments not following Google-style docstring conventions. This task ensures all comments are standardized to professional English, follow Google-style format, and accurately match the code logic.

## What Changes
- Translate all Chinese comments to professional technical English
- Standardize all Python docstrings to Google-style format (Description, Args, Returns)
- Standardize all Shell script comments to professional English
- Correct any comments that don't match the actual code logic
- Ensure class docstrings include Description and Attributes sections
- Ensure function/method docstrings include Description, Args, and Returns sections

## Impact
- Affected files: All .py and .sh files in the project root and subdirectories
- No changes to code logic, variable names, string constants, or execution behavior
- Only static text modifications to comments

## ADDED Requirements

### Requirement: Python Comment Standardization
The system SHALL ensure all Python files follow Google-style docstring conventions.

#### Scenario: Class docstring format
- **WHEN** a class is defined in a Python file
- **THEN** the class SHALL have a Google-style docstring with Description and Attributes sections

#### Scenario: Function/method docstring format
- **WHEN** a function or method is defined in a Python file
- **THEN** it SHALL have a Google-style docstring with Description, Args, and Returns sections (Returns: None if no return value)

#### Scenario: Inline comment format
- **WHEN** an inline comment is present in Python code
- **THEN** it SHALL be in English and accurately describe the code logic

### Requirement: Shell Script Comment Standardization
The system SHALL ensure all Shell scripts have professional English comments.

#### Scenario: Script header comments
- **WHEN** a .sh file is processed
- **THEN** it SHALL have a header comment block describing the script's purpose, parameters, and usage

#### Scenario: Function comments in shell scripts
- **WHEN** a function is defined in a shell script
- **THEN** it SHALL have a comment describing its purpose, inputs, and outputs

### Requirement: Chinese Comment Translation
The system SHALL translate all Chinese comments to professional technical English.

#### Scenario: Chinese comment detection
- **WHEN** a comment contains Chinese characters
- **THEN** it SHALL be translated to accurate technical English
- **AND** the translation SHALL preserve the original meaning
- **AND** the translation SHALL use appropriate technical terminology

### Requirement: Comment-Code Accuracy
The system SHALL ensure all comments accurately reflect the code they describe.

#### Scenario: Outdated comment correction
- **WHEN** a comment does not match the code logic
- **THEN** the comment SHALL be corrected to accurately describe the code

#### Scenario: Missing comment addition
- **WHEN** complex logic lacks explanatory comments
- **THEN** appropriate English comments SHALL be added

## Constraints
1. **No code execution**: Only static text read/write operations are permitted
2. **No logic changes**: Code logic, variable definitions, string constants, and command statements must not be modified
3. **No Chinese in comments**: All comments must be in English only
4. **No translation of non-comment elements**: Variable names, string constants, and code logic must not be translated

## Files to Process

### Python Files (21 files)
1. models/configuration_llama.py
2. models/pe_llama.py
3. models/modeling_llama.py
4. models/entropy_stable_scaled_rope.py
5. models/model_loader.py
6. draw_rope_heatmap.py
7. draw_attention_scaling.py
8. draw.py
9. draw_position.py
10. test.py
11. continued_pretrain.py
12. finetune.py
13. performance_compare.py
14. clean_json.py
15. eval/quality.py
16. eval/perplexity.py
17. eval/performance.py
18. eval/passkey.py
19. eval/plot_entropy.py
20. eval/entropy.py
21. eval/eval_harness.py
22. drawer/position_heatmap.py
23. drawer/block_sum.py

### Shell Files (9 files)
1. permission.sh
2. finetune.sh
3. test.sh
4. continued_pretrain.sh
5. eval2.sh
6. entropy.sh
7. eval1.sh
8. eval_harness.sh
9. eval.sh
