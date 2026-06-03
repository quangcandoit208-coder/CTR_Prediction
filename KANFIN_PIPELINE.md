# KANFIN Pipeline

Tai lieu nay mo ta chi tiet pipeline cua `KANFIN` trong project, dua tren:

- `src/models/kanfin.py`
- `src/models/kan.py`
- `src/models/fin.py`

Muc tieu cua `KANFIN` la thay nhanh additive `NAM` trong `NAFI` bang nhanh additive `KAN`, trong khi giu nhanh tuong tac `FIN`.

```text
KANFIN(x) = KANBranch(Embedding(x)) + FINBranch(Embedding(x))
```

Output cuoi cung la logit. Khi can xac suat CTR, dung:

```math
p(click=1|x) = sigmoid(logit)
```

Trong training, project dung `BCEWithLogitsLoss`, nen model tra ve logits, khong sigmoid truc tiep trong forward.

## 1. Input va Embedding

Input cua model la tensor categorical id:

```text
x shape = [batch_size, num_fields]
```

Moi cot la mot feature/field, vi du:

```text
hour, C1, banner_pos, site_id, site_domain, ..., weekday
```

`FeatureEmbedding` bien moi feature id thanh vector embedding:

```math
E_f = Embedding_f(x_f) \in R^D
```

Voi ca batch:

```text
embeddings shape = [batch_size, num_fields, embedding_dim]
```

Ky hieu:

```math
E_{b,f,d}
```

Trong do:

- `b`: index cua sample trong batch.
- `f`: index cua feature.
- `d`: index cua chieu embedding.

Voi config Kaggle hien tai:

```yaml
model:
  embedding_dim: 16
```

nen:

```text
embeddings shape = [batch_size, 25, 16]
```

neu metadata co 25 fields.

## 2. KANFIN Tong Quan

Trong `src/models/kanfin.py`, forward gom 4 buoc:

```python
embeddings = self.embedding(x)
kan_logits, contributions = self.kan(embeddings)
fin_logits, attention_weights = self.fin(embeddings)
logits = kan_logits + fin_logits
```

Output dict:

```python
{
    "logits": logits,
    "kan_logits": kan_logits,
    "fin_logits": fin_logits,
    "attention_weights": attention_weights,
    "feature_contributions": contributions,
}
```

Y nghia:

- `logits`: output fused cua KANFIN.
- `kan_logits`: output rieng cua nhanh KAN.
- `fin_logits`: output rieng cua nhanh FIN.
- `attention_weights`: attention cua FIN branch.
- `feature_contributions`: contribution theo feature cua KAN branch.

Cong thuc tong:

```math
KANFIN(x) = KAN(E) + FIN(E)
```

voi:

```math
E = Embedding(x)
```

## 3. KAN Branch

`KANBranch` la nhanh additive theo feature:

```math
KAN(E) = b_{kan} + \sum_{f=1}^{F} c_f
```

Trong do:

- `F`: so feature.
- `c_f`: scalar contribution cua feature `f`.
- `b_kan`: global bias cua KAN branch.

### 3.1 KAN scalar function

KAN dung `SharedScalarKANLayer` lam mot ham phi tuyen mot bien:

```math
\phi(x) = base\_weight \cdot SiLU(x) + spline(x) + bias
```

Trong do:

- `SiLU(x)` la base function.
- `spline(x)` la spline learnable.
- `base_weight`, `spline_weight`, `bias` la tham so hoc duoc.

Config hien tai:

```yaml
kan:
  grid_size: 5
  degree: 2
  grid_min: -0.8
  grid_max: 0.8
  use_base: true
```

`grid_min/grid_max` la range embedding scalar truoc khi dua vao spline. Gia tri ngoai range bi clamp.

### 3.2 Spline degree

Trong code:

- `degree = 1`: piecewise-linear interpolation, giu behavior ban dau.
- `degree > 1`: dung open-uniform B-spline basis.

So basis cua moi scalar KAN function:

```math
num\_basis =
\begin{cases}
grid\_size, & degree = 1 \\
grid\_size + degree - 1, & degree > 1
\end{cases}
```

Voi:

```text
grid_size = 5
degree = 2
```

thi:

```text
num_basis = 6
```

Moi ham scalar KAN co:

```text
num_basis spline weights + 1 base_weight + 1 bias
```

neu `use_base=true`.

### 3.3 share_mode = field

Voi `share_mode: field`, moi feature co mot ham KAN scalar rieng:

```math
z_{b,f,d} = \phi_f(E_{b,f,d})
```

Nghia la:

```text
site_id     -> phi_site_id
device_ip   -> phi_device_ip
hour        -> phi_hour
```

Ben trong cung mot feature, tat ca cac chieu embedding dung chung ham `phi_f`:

```text
site_id dim_1  -> phi_site_id
site_id dim_2  -> phi_site_id
...
site_id dim_16 -> phi_site_id
```

Day la diem quan trong: code hien tai khong dung mot ham rieng cho tung chieu embedding. Neu lam nhu vay cong thuc se la:

```math
z_{b,f,d} = \phi_{f,d}(E_{b,f,d})
```

nhung do khong phai implementation hien tai.

### 3.4 Tu embedding vector ra scalar contribution

Moi feature embedding la vector:

```math
E_f = [e_{f,1}, e_{f,2}, ..., e_{f,D}]
```

KAN ap ham `phi_f` len tung scalar:

```math
z_{f,d} = \phi_f(e_{f,d})
```

Sau do gom cac chieu embedding bang weighted sum:

```math
s_f = \sum_{d=1}^{D} w_d z_{f,d}
```

Trong code:

```python
dim_weight
```

la vector weight theo chieu embedding. Luu y: `dim_weight` dung chung cho moi feature.

Sau do scale va bias theo feature:

```math
c_f = a_f s_f + \beta_f
```

Trong code:

```python
field_weight[f] = a_f
field_bias[f] = beta_f
```

Cong thuc day du cua KAN branch:

```math
KAN(E)
=
b_{kan}
+
\sum_{f=1}^{F}
\left[
  a_f
  \sum_{d=1}^{D}
  w_d \phi_f(E_{f,d})
  +
  \beta_f
\right]
```

Voi batch:

```math
KAN(E_b)
=
b_{kan}
+
\sum_{f=1}^{F}
\left[
  a_f
  \sum_{d=1}^{D}
  w_d \phi_f(E_{b,f,d})
  +
  \beta_f
\right]
```

`feature_contributions` chinh la:

```math
c_{b,f}
```

voi shape:

```text
[batch_size, num_fields]
```

### 3.5 share_mode = global

Neu doi thanh:

```yaml
kan:
  share_mode: global
```

thi tat ca feature va tat ca embedding dimension dung chung mot ham:

```math
z_{b,f,d} = \phi(E_{b,f,d})
```

Nhung contribution theo feature van ton tai:

```math
c_{b,f}
=
a_f
\sum_d w_d \phi(E_{b,f,d})
+
\beta_f
```

Khac biet:

- `field`: moi feature co ham rieng `phi_f`.
- `global`: moi feature dung chung mot ham `phi`.

`global` it tham so hon, regularize manh hon, nhung kem flexible va kem interpretability theo function cua tung feature.

## 4. FIN Branch

`FINBranch` trong `src/models/fin.py` hoc feature interaction bang multi-head self-attention.

Input:

```text
embeddings shape = [batch_size, num_fields, embedding_dim]
```

Moi attention layer:

```python
out, attention_weights = attn(out, out, out)
```

Day la self-attention giua cac feature embeddings:

```math
Attention(Q,K,V)
=
softmax\left(\frac{QK^T}{\sqrt{d}}\right)V
```

Trong code:

- `Q = out`
- `K = out`
- `V = out`
- `batch_first=True`

Voi `num_heads=4`, model hoc nhieu kieu interaction giua feature.

### 4.1 Residual, ReLU, LayerNorm

Moi FIN layer lam:

```python
residual = out
out, attention_weights = attn(out, out, out)
if use_residual:
    out = out + residual
out = norm(ReLU(out))
```

Cong thuc tom tat:

```math
H^{(l+1)}
=
LayerNorm(ReLU(Attention(H^{(l)}) + H^{(l)}))
```

neu `use_residual=true`.

### 4.2 FIN output

Sau cac attention layer, FIN flatten tensor:

```text
out shape = [batch_size, num_fields, embedding_dim]
flatten -> [batch_size, num_fields * embedding_dim]
```

Sau do qua MLP:

```python
MLP(num_fields * embedding_dim, [64, 32], output_dim=1)
```

Tao ra:

```text
fin_logits shape = [batch_size]
```

Cong thuc tong quat:

```math
FIN(E) = MLP(flatten(SelfAttentionStack(E)))
```

FIN khong phai additive theo feature. No hoc interaction giua cac feature.

## 5. Fusion cua KAN va FIN

`KANFIN` cong hai logits:

```math
logit_b = kan\_logit_b + fin\_logit_b
```

Sau do neu can probability:

```math
p_b = sigmoid(logit_b)
```

Y nghia:

- `KAN` hoc main effect/additive contribution theo tung feature.
- `FIN` hoc interaction giua feature bang attention.
- `KANFIN` ket hop ca hai:

```math
KANFIN(x)
=
b_{kan}
+
\sum_f c_f
+
FIN(E)
```

## 6. Relation voi NAFI

NAFI goc trong project:

```math
NAFI(x) = NAM(E) + FIN(E)
```

KANFIN thay:

```math
NAM(E)
```

bang:

```math
KAN(E)
```

Nen:

```math
KANFIN(x) = KAN(E) + FIN(E)
```

So sanh:

| Branch | NAFI | KANFIN |
|---|---|---|
| Additive branch | NAMBranch | KANBranch |
| Interaction branch | FINBranch | FINBranch |
| Output | NAM logits + FIN logits | KAN logits + FIN logits |
| Feature contribution | NAM contribution | KAN contribution |
| Interaction interpretability | attention heatmap | attention heatmap |

## 7. Interpretability

KANFIN co hai loai interpretability chinh.

### 7.1 KAN feature contribution

`KANBranch` tra ve:

```python
feature_contributions
```

Shape:

```text
[batch_size, num_fields]
```

Moi gia tri:

```math
c_{b,f}
```

la contribution cua feature `f` vao `kan_logits` cua sample `b`.

Neu:

```text
c_{b,f} > 0
```

feature do day KAN logit tang.

Neu:

```text
c_{b,f} < 0
```

feature do keo KAN logit giam.

Tong hop tren validation set:

```math
importance_f
=
\frac{mean(|c_f|)}
{\sum_j mean(|c_j|)}
```

Nen report them signed mean:

```math
direction_f = mean(c_f)
```

De biet feature thuong day logit len hay xuong.

### 7.2 KAN scalar function theo feature

Voi `share_mode: field`, moi feature co mot ham:

```math
\phi_f(x)
```

Co the ve:

```math
x \mapsto \phi_f(x)
```

Luu y quan trong: `x` la scalar cua embedding, khong phai raw category id.

Do do co the noi:

```text
feature site_id co KAN scalar response curve nhu the nao tren embedding scalar
```

nhung khong nen noi truc tiep:

```text
site_id category A co y nghia vat ly X
```

vi input da qua:

```text
category -> hash bucket -> embedding vector -> KAN
```

### 7.3 FIN attention heatmap

FIN tra ve:

```python
attention_weights
```

Voi multi-head attention:

```text
attention_weights shape roughly = [batch_size, num_heads, num_fields, num_fields]
```

Sau khi average qua batch/head:

```text
attention_matrix shape = [num_fields, num_fields]
```

O vi tri:

```math
attention_{i,j}
```

co the doc la feature `i` chu y den feature `j` trong nhanh interaction.

Can can than: attention khong phai causal explanation tuyet doi. No la signal ve interaction pattern, khong phai bang chung nhan qua.

## 8. Output va Metrics

Trong training/evaluation, model co the log:

```text
valid_auc
valid_logloss
valid_kan_auc
valid_kan_logloss
valid_fin_auc
valid_fin_logloss
valid_auc_gain_over_kan
valid_auc_gain_over_fin
```

Y nghia:

- `valid_auc`: metric cua output fused `KAN + FIN`.
- `valid_kan_auc`: metric cua nhanh KAN rieng.
- `valid_fin_auc`: metric cua nhanh FIN rieng.
- `valid_auc_gain_over_kan`: fused output tot hon KAN rieng bao nhieu.
- `valid_auc_gain_over_fin`: fused output tot hon FIN rieng bao nhieu.

Neu:

```text
valid_auc > valid_kan_auc va valid_auc > valid_fin_auc
```

thi hai nhanh dang bo sung thong tin cho nhau.

## 9. Parameter Count cua KAN Branch

Ky hieu:

- `F`: so feature.
- `D`: embedding dim.
- `G`: `num_basis` cua scalar KAN function.
- `P_phi`: tham so cua mot scalar KAN function.

Neu `use_base=true`:

```math
P_\phi = G + 2
```

gom:

- `G` spline weights.
- `1` base weight.
- `1` bias.

Voi `share_mode=field`:

```math
P_{KAN}
=
F P_\phi
+
D
+
2F
+
1
```

Trong do:

- `F P_phi`: moi feature mot scalar KAN function.
- `D`: `dim_weight`.
- `F`: `field_weight`.
- `F`: `field_bias`.
- `1`: global branch bias.

Voi config hien tai:

```text
F = 25
D = 16
grid_size = 5
degree = 2
num_basis = 6
P_phi = 6 + 2 = 8
```

Thi:

```math
P_{KAN}
=
25 * 8 + 16 + 50 + 1
=
267
```

Day chi la tham so cua KAN branch, khong tinh embedding tables va FIN branch.

Voi `share_mode=global`:

```math
P_{KAN}
=
P_\phi
+
D
+
2F
+
1
```

Theo config tren:

```math
P_{KAN}
=
8 + 16 + 50 + 1
=
75
```

## 10. Practical Commands

Train KANFIN tren parquet:

```bash
python scripts/train.py \
  --config config/kaggle_t4x2.yaml \
  --model kanfin \
  --epochs 3
```

Evaluate checkpoint:

```bash
python scripts/evaluate.py \
  --config config/kaggle_t4x2.yaml \
  --checkpoint outputs/checkpoints/best_model.pt \
  --model kanfin
```

Plot interpretability:

```bash
python scripts/plot_kanfi_interpretability.py \
  --config config/kaggle_t4x2.yaml \
  --checkpoint outputs/checkpoints/best_model.pt \
  --split valid \
  --max-batches 50 \
  --top-features 15
```

Outputs:

```text
outputs/figures/kanfi_interpretability/
  kan_feature_importance.png
  kan_feature_weights.png
  kan_branch_schematic.png
  kan_scalar_functions.png
  fin_attention_heatmap.png
  kan_feature_contributions.npy
  fin_attention_matrix.npy

outputs/metrics/
  valid_kanfin_kanfi_interpretability.json
```

## 11. Tom tat ngan

KANFIN gom:

```text
categorical ids
-> embedding
-> KAN additive branch
-> FIN interaction branch
-> sum logits
-> sigmoid for CTR probability
```

KAN branch:

```math
KAN(E)
=
b
+
\sum_f
\left[
  a_f
  \sum_d
  w_d \phi_f(E_{f,d})
  +
  \beta_f
\right]
```

FIN branch:

```math
FIN(E) = MLP(flatten(SelfAttentionStack(E)))
```

Final:

```math
KANFIN(x) = KAN(E) + FIN(E)
```

Interpretability:

- KAN: contribution theo feature va ham `phi_f` theo feature.
- FIN: attention heatmap giua cac feature.
- Output fused: predictive performance cua ca hai nhanh cong lai.
