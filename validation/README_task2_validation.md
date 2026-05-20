# Task2 Validation Protocol

This validation is designed for policies trained on `ETHrobotlearning/tv-task2-clean-aug5p`.
It tests the two main factors that should control the behavior:

- bowl configuration: which color is left, center, and right
- prompt wording: direct color, negation, ordinal, relative, and between prompts

## Generated Sheets

Use the generator:

```bash
python validation/generate_task2_validation_plan.py \
  --profile balanced \
  --repeats 3 \
  --policy-repo-id ETHrobotlearning/YOUR_POLICY_REPO \
  --checkpoint step-50000 \
  --output validation/task2_balanced_validation.csv
```

For a complete sweep of all valid prompt/configuration pairs:

```bash
python validation/generate_task2_validation_plan.py \
  --profile full \
  --repeats 1 \
  --policy-repo-id ETHrobotlearning/YOUR_POLICY_REPO \
  --checkpoint step-50000 \
  --output validation/task2_full_validation.csv
```

## Recommended Evaluation

For a fast evaluation, use the 30-rollout compact sheet:

```bash
python validation/generate_task2_validation_plan.py \
  --profile compact30 \
  --no-shuffle \
  --policy-repo-id ETHrobotlearning/YOUR_POLICY_REPO \
  --checkpoint step-50000 \
  --output validation/task2_30rollout_validation.csv
```

This covers:

- 6 bowl configurations, 5 rollouts each
- 10 target-red, 10 target-green, 10 target-blue trials
- direct color, color paraphrase, negation, ordinal, and spatial relation prompts

Start with the balanced sheet.

- `balanced`, 3 repeats: 270 rollouts
- 6 bowl configurations
- 3 target colors per configuration
- 5 prompt tests per target/configuration

The 5 prompt tests cover:

- direct color
- color paraphrase
- negation
- ordinal/position
- relative or between relation

Run the full sheet only after the balanced evaluation shows promising behavior.

## Scoring

Fill these fields after each rollout:

- `grasp_success`: banana is lifted from the table
- `correct_bowl_motion`: robot moves toward the expected bowl
- `release_success`: gripper opens above a bowl
- `task_success`: banana ends inside the expected bowl
- `actual_bowl_color`: red, green, blue, or none
- `actual_bowl_position`: left, center, right, or none
- `failure_mode`: wrong_bowl, no_grasp, dropped_before_bowl, no_release, timeout, collision

The main metric is `task_success`.
The most diagnostic metric is success grouped by `prompt_family` and `config_name`.
