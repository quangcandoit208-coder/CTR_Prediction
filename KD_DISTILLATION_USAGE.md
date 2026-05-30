# KD-NAFI Distillation Usage

Paper dùng teacher ensemble gồm:

```text
DeepFM + xDeepFM + AutoInt
```

Sau đó distill sang student:

```text
KD-NAFI
```

## 1. Train Teacher Models

Nên dùng output dir riêng cho từng teacher để checkpoint không ghi đè nhau.

```bash
python -m scripts.train \
  --config config/kaggle_t4x2.yaml \
  --model deepfm \
  --processed-dir <processed_dir> \
  --output-dir /kaggle/working/outputs_deepfm
```

Tương tự:

```bash
python -m scripts.train \
  --config config/kaggle_t4x2.yaml \
  --model xdeepfm \
  --processed-dir <processed_dir> \
  --output-dir /kaggle/working/outputs_xdeepfm
```

```bash
python -m scripts.train \
  --config config/kaggle_t4x2.yaml \
  --model autoint \
  --processed-dir <processed_dir> \
  --output-dir /kaggle/working/outputs_autoint
```

## 2. Train Student KD-NAFI

```bash
python -m scripts.train_kd \
  --config config/kaggle_t4x2.yaml \
  --processed-dir <processed_dir> \
  --output-dir /kaggle/working/outputs_kd_nafi \
  --student-model kd_nafi \
  --teacher deepfm:/kaggle/working/outputs_deepfm/checkpoints/best_model.pt \
  --teacher xdeepfm:/kaggle/working/outputs_xdeepfm/checkpoints/best_model.pt \
  --teacher autoint:/kaggle/working/outputs_autoint/checkpoints/best_model.pt \
  --temperature 4.0 \
  --alpha 0.5 \
  --ensemble-mode uniform
```

## 3. Optional Fixed Teacher Weights

Nếu muốn weighted ensemble:

```bash
python -m scripts.train_kd \
  --config config/kaggle_t4x2.yaml \
  --processed-dir <processed_dir> \
  --output-dir /kaggle/working/outputs_kd_nafi \
  --student-model kd_nafi \
  --teacher deepfm:/kaggle/working/outputs_deepfm/checkpoints/best_model.pt \
  --teacher xdeepfm:/kaggle/working/outputs_xdeepfm/checkpoints/best_model.pt \
  --teacher autoint:/kaggle/working/outputs_autoint/checkpoints/best_model.pt \
  --teacher-weight 0.3 0.4 0.3 \
  --temperature 4.0 \
  --alpha 0.5
```

## 4. Optional Confidence Ensemble

This computes per-sample teacher weights from teacher confidence:

```bash
--ensemble-mode confidence
```

## 5. Loss

Training objective:

```text
loss = alpha * BCE(student_logits, hard_labels)
     + (1 - alpha) * T^2 * BCE(student_logits / T, teacher_probs_T)
```

where:

```text
teacher_probs_T = ensemble(sigmoid(teacher_logits / T))
```

## 6. Evaluate

```bash
python -m scripts.evaluate \
  --config config/kaggle_t4x2.yaml \
  --checkpoint /kaggle/working/outputs_kd_nafi/checkpoints/best_model.pt \
  --split test \
  --processed-dir <processed_dir>
```
