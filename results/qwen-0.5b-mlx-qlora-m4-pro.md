# Qwen2.5 0.5B MLX QLoRA smoke experiment

This experiment validates Apple-native 4-bit QLoRA training and objective
base-versus-adapter evaluation on an Apple M4 Pro with 48 GiB unified memory.
The held-out result is negative and is reported without post-test tuning.

## Controls

| Control | Value |
| --- | --- |
| MLX model | `mlx-community/Qwen2.5-0.5B-Instruct-4bit` |
| Upstream revision | `a5339a4131f135d0fdc6a5c8b5bbed2753bbe0f3` |
| Stable local snapshot SHA-256 | `8eb92e234f43ed1826f02aafd04c1bd210fe7bb028fc1341935b633bad2b0cd3` |
| Training dataset SHA-256 | `2b8c70228fa1ac01362bc3abd46c09660cc8915d2c90cded9c0c2ef910585c58` |
| Learned weight SHA-256 | `b5ea6ec24bf768292e2c0ff5e79ece3711e5819bf35fc783e66d98ed8f0c932e` |
| MLX-LM | 0.31.3 |
| Seed | 42 |
| Generation | Greedy, maximum 64 tokens |

The repository-owned synthetic smoke corpus contains 12 training, 3
validation, and 6 test questions. Source-document groups are disjoint across
splits: four train groups, one validation group, and two test groups. Data
preparation rejects duplicate normalized questions, documents shared between
splits, and exact normalized answer collisions across splits.

## Training

| Setting | Value |
| --- | ---: |
| Quantization | 4-bit input model, therefore QLoRA |
| Trainable layers | 4 |
| Trainable parameters | 0.733M of 494.033M (0.148%) |
| Iterations | 30 |
| Batch size | 1 |
| Learning rate | 0.00001 |
| Prompt masking | enabled |
| Trained completion tokens | 533 |
| Initial validation loss | 4.577 |
| Final validation loss | 2.631 |
| Final train loss | 1.820 |
| Peak MLX memory | 0.459 GB |
| Wall time | 2.59 seconds |
| Adapter artifact size | 2.8 MiB |

Three independent runs with the same seed and configuration produced
byte-identical `adapters.safetensors` files. MLX's generated
`adapter_config.json` embeds run-specific output paths, so whole-directory
digests differ even when the learned weight bytes are identical.

## Held-Out Evaluation

Base evaluation was completed and frozen before training. The adapter was then
evaluated sequentially, not concurrently, on exactly the same six test items.

| Metric | Base 4-bit model | QLoRA adapter | Delta |
| --- | ---: | ---: | ---: |
| Normalized exact match | 0.000 | 0.000 | 0.000 |
| Mean token F1 | 0.209 | 0.128 | -0.081 |
| Median output tokens | 47.5 | 14.5 | -33.0 |
| Median output token/s | 242.44 | 107.68 | -134.75 |
| Median generation latency | 183.11 ms | 114.44 ms | -68.67 ms |

The lower adapter latency is not a speed improvement: the adapter generated
much shorter answers. Token-normalized throughput was lower, and held-out token
F1 decreased by 0.081. Exact match remained zero because both arms produced
free-form answers rather than exact reference strings.

## Interpretation

The training pipeline works and validation loss improved, but this 30-step,
12-example adapter did not improve the untouched test set. It likely learned
brevity and narrow training vocabulary rather than generalizing to the held-out
pricing and demand-response document groups.

No hyperparameter, prompt, checkpoint, or dataset change was made after reading
test predictions. The six-item synthetic test split is sufficient for pipeline
validation but far too small for a general model-quality claim. A defensible
follow-up requires a larger source-grounded corpus, checkpoint selection using
validation only, and a test set that remains sealed until the training protocol
is finalized.
