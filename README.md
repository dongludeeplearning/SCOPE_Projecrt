# SCOPE Project: Cognitive Detection

## Environment Setup

Create and activate the conda environment:

```bash
conda create --name edu-cognition python=3.9
conda activate edu-cognition
```

Install dependencies:

```bash
pip install dlib
pip install mediapipe
```

## Experiments

### Model Variations

- **v1**: Inception frame-level feature + temporal pooling (transformer encoder) + classifier (No Position Encoding).
- **v2**: v1 + Position Encoding.
- **v3**: v2 + Mixture of Experts (MoE).

### Usage

**Train:**

```bash
python train.py --model_type v1  # Replace v1 with v2 or v3
```

**Test:**

```bash
python test.py --model_type v1   # Replace v1 with v2 or v3
```

### Results Summary

| Model | Description | Task B Accuracy | Task E Accuracy | Task C Accuracy | Task F Accuracy |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **v1** | Inception + Transformer (No PE) | 0.4615 | 0.5216 | 0.7014 | 0.7852 |
| **v2** | v1 + Position Encoding | 0.4627 | 0.5230 | 0.7031 | 0.7846 |
| **v3** | v2 + MoE | 0.3684 | 0.5255 | 0.7016 | 0.7861 |

## TODO

- [ ] Explore different MoE designs to improve performance.
- [ ] Combine AUs (Action Units) and VA (Valence/Arousal) to train v2.
