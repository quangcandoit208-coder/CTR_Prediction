# Giai thich KAN branch trong KANFI va lien he voi code PyTorch

Tai lieu nay giai thich chi tiet cach nhanh KAN trong `KANFIN` duoc tinh trong code, cach doc contribution theo feature, contribution theo tung chieu embedding, va y nghia cua cac ham PyTorch duoc su dung.

Code chinh nam o:

- `src/models/kan.py`
- `src/models/kanfin.py`
- `src/models/embedding.py`
- `src/models/fin.py`

## 1. Ky hieu va tensor shape

Voi mot batch co kich thuoc \(B\), moi sample gom \(m\) feature fields:

\[
\mathbf{x} = (x_1, x_2, \ldots, x_m).
\]

Sau lop embedding, moi feature \(x_i\) duoc bien doi thanh vector:

\[
\mathbf{v}_i = (v_{i,1}, v_{i,2}, \ldots, v_{i,d}) \in \mathbb{R}^{d}.
\]

Toan bo embedding cua mot sample la:

\[
\mathbf{E}
=
[\mathbf{v}_1,\mathbf{v}_2,\ldots,\mathbf{v}_m]^T
\in \mathbb{R}^{m \times d}.
\]

Trong code, voi batch size \(B\):

```python
embeddings = self.embedding(x)
```

co shape:

```text
[batch, fields, dim] = [B, m, d]
```

Trong `KANFIN.forward`, tensor nay duoc dua song song vao hai nhanh:

```python
kan_logits, contributions = self.kan(embeddings)
fin_logits, attention_weights = self.fin(embeddings)
logits = kan_logits + fin_logits
```

Tuong ung voi:

\[
\operatorname{KANFI}(\mathbf{x})
=
\operatorname{KAN}(\mathbf{x})
+
\operatorname{FIN}(\mathbf{x}).
\]

## 2. Tao embedding trong `FeatureEmbedding`

File: `src/models/embedding.py`

Moi feature field co mot bang embedding rieng:

```python
self.embeddings = nn.ModuleList([nn.Embedding(dim, embedding_dim) for dim in field_dims])
```

Neu co \(m\) feature fields, `ModuleList` nay chua \(m\) object `nn.Embedding`.

Trong `forward`:

```python
outputs = []
for idx, embedding in enumerate(self.embeddings):
    field_x = torch.remainder(x[:, idx], embedding.num_embeddings)
    outputs.append(embedding(field_x))
return torch.stack(outputs, dim=1)
```

Y nghia:

- `x[:, idx]`: lay cot feature thu `idx` trong batch, shape `[B]`.
- `torch.remainder(...)`: dam bao chi so category nam trong khoang hop le cua embedding table.
- `embedding(field_x)`: tra ve embedding vector cua feature do, shape `[B, d]`.
- `outputs.append(...)`: gom embedding cua tung feature.
- `torch.stack(outputs, dim=1)`: ghep \(m\) tensor `[B, d]` thanh mot tensor `[B, m, d]`.

Ket qua la:

\[
\mathbf{E} \in \mathbb{R}^{B \times m \times d}.
\]

## 3. Ham scalar KAN \(\phi_i(t)\)

Ham \(\phi_i(t)\) duoc cai dat trong class `SharedScalarKANLayer` o `src/models/kan.py`.

Trong docstring, ham nay duoc mo ta gan dung nhu sau:

```python
phi(x) = base_weight * SiLU(x) + linear_spline(x) + bias
```

Dang toan hoc:

\[
\phi_i(t)
=
\gamma_i \operatorname{SiLU}(t)
+
\sum_{r=1}^{M} \alpha_{i,r} B_{r,p}(\bar{t})
+
\eta_i,
\]

voi:

\[
\bar{t} = \operatorname{clip}(t, g_{\min}, g_{\max}).
\]

Trong do:

- \(\phi_i\): ham scalar KAN cua feature \(i\).
- \(t\): mot scalar trong embedding.
- \(\gamma_i\): `base_weight`.
- \(\operatorname{SiLU}(t)\): base activation.
- \(B_{r,p}\): B-spline basis thu \(r\), bac \(p\).
- \(\alpha_{i,r}\): `spline_weight`.
- \(\eta_i\): bias trong `SharedScalarKANLayer`.
- \([g_{\min}, g_{\max}]\): grid range, vi du `[-0.8, 0.8]`.

### 3.1. Tham so cua \(\phi_i\)

Trong `SharedScalarKANLayer.__init__`:

```python
self.spline_weight = nn.Parameter(torch.empty(self.num_basis))
```

Day la cac he so spline:

\[
\alpha_{i,1}, \alpha_{i,2}, \ldots, \alpha_{i,M}.
\]

Neu `use_base=True`:

```python
self.base_weight = nn.Parameter(torch.ones(()))
```

Day la he so \(\gamma_i\) cua base activation.

Bias cua ham scalar:

```python
self.bias = nn.Parameter(torch.zeros(()))
```

Day la \(\eta_i\).

### 3.2. Grid va knot cua spline

```python
self.register_buffer("grid_min", torch.tensor(float(grid_min)))
self.register_buffer("grid_max", torch.tensor(float(grid_max)))
self.register_buffer("grid_step", torch.tensor(float(grid_max - grid_min) / float(grid_size - 1)))
self.register_buffer("knots", self._make_open_uniform_knots(...))
```

`register_buffer` dung de luu cac tensor khong phai tham so hoc duoc. Chunga van di theo model khi goi `.to(device)`, duoc luu trong checkpoint, nhung khong duoc optimizer cap nhat.

Trong truong hop nay:

- `grid_min`, `grid_max`: can trai/phai cua mien spline.
- `grid_step`: khoang cach giua hai diem grid lien tiep.
- `knots`: vector knot cua B-spline.

### 3.3. Forward cua \(\phi_i(t)\)

Trong `SharedScalarKANLayer.forward`:

```python
clipped = torch.minimum(torch.maximum(x, grid_min), grid_max)
```

Dong nay tuong ung voi:

\[
\bar{t}=\operatorname{clip}(t,g_{\min},g_{\max}).
\]

Y nghia PyTorch:

- `torch.maximum(x, grid_min)`: neu gia tri nho hon `grid_min` thi day len `grid_min`.
- `torch.minimum(..., grid_max)`: neu gia tri lon hon `grid_max` thi keo xuong `grid_max`.
- Ket qua la moi gia tri nam trong khoang spline.

Neu `degree == 1`, code dung noi suy tuyen tinh:

```python
position = (clipped - grid_min) / grid_step
lower_idx = torch.floor(position).to(torch.long).clamp(min=0, max=self.grid_size - 2)
upper_idx = lower_idx + 1
frac = (position - lower_idx.to(position.dtype)).clamp(0.0, 1.0)
lower = spline_weight[lower_idx]
upper = spline_weight[upper_idx]
out = lower * (1.0 - frac) + upper * frac
```

Y nghia:

- `position`: vi tri lien tuc cua scalar tren grid.
- `torch.floor(position)`: lay chi so grid ben trai.
- `lower_idx`, `upper_idx`: hai diem grid lan can.
- `frac`: ty le nam giua hai diem grid.
- `lower`, `upper`: trong so spline tai hai diem grid.
- `out`: gia tri noi suy tuyen tinh.

Cong thuc:

\[
\operatorname{spline}(t)
=
(1-\lambda)\alpha_{\mathrm{lower}}
+
\lambda\alpha_{\mathrm{upper}},
\]

voi \(\lambda=\texttt{frac}\).

Neu `degree > 1`, code dung B-spline basis:

```python
basis = self._bspline_basis(clipped.reshape(-1))
out = torch.matmul(basis, spline_weight).reshape_as(x)
```

Tuong ung voi:

\[
\operatorname{spline}(t)
=
\sum_{r=1}^{M} \alpha_{i,r} B_{r,p}(t).
\]

Y nghia PyTorch:

- `clipped.reshape(-1)`: trai phang input thanh vector 1 chieu de tinh basis.
- `_bspline_basis(...)`: tinh ma tran basis, shape `[num_values, num_basis]`.
- `torch.matmul(basis, spline_weight)`: nhan ma tran basis voi vector trong so spline.
- `reshape_as(x)`: dua output ve lai shape ban dau cua `x`.

Sau do cong base activation:

```python
if self.base_weight is not None:
    out = out + self.base_weight.to(dtype=x.dtype) * F.silu(x)
return out + self.bias.to(dtype=x.dtype)
```

Tuong ung voi:

\[
\phi_i(t)
=
\operatorname{spline}(t)
+
\gamma_i \operatorname{SiLU}(t)
+
\eta_i.
\]

`F.silu(x)` la ham:

\[
\operatorname{SiLU}(x)=x\sigma(x),
\]

trong do:

\[
\sigma(x)=\frac{1}{1+e^{-x}}.
\]

## 4. Field mode: share ham \(\phi_i\) trong mot feature

Trong `KANBranch.__init__`:

```python
num_scalar_functions = self.num_fields if share_mode == "field" else 1
self.scalar_kans = nn.ModuleList(
    [
        SharedScalarKANLayer(...)
        for _ in range(num_scalar_functions)
    ]
)
```

Neu `share_mode == "field"`:

\[
\text{so ham scalar KAN} = m.
\]

Tuc la moi feature field co mot ham rieng:

\[
\phi_1,\phi_2,\ldots,\phi_m.
\]

Nhung trong cung mot feature \(i\), moi chieu embedding deu dung chung ham \(\phi_i\):

\[
\phi_i(v_{i,1}),\phi_i(v_{i,2}),\ldots,\phi_i(v_{i,d}).
\]

Trong `KANBranch.forward`:

```python
transformed = torch.stack(
    [scalar_kan(embeddings[:, idx, :]) for idx, scalar_kan in enumerate(self.scalar_kans)],
    dim=1,
)
```

Day la doan code quan trong nhat de bieu dien share ham \(\phi\).

Voi feature `idx = i`:

```python
embeddings[:, idx, :]
```

co shape:

```text
[B, d]
```

Toan bo \(d\) scalar trong embedding cua feature \(i\) duoc dua qua cung object:

```python
scalar_kan(...)
```

Tuong ung voi:

\[
u_{i,j} = \phi_i(v_{i,j}), \qquad j=1,\ldots,d.
\]

Sau khi `torch.stack(..., dim=1)`, `transformed` co shape:

```text
[B, m, d]
```

Va:

\[
\texttt{transformed}_{b,i,j}
=
\phi_i(v_{b,i,j}).
\]

### So sanh voi global mode

Neu `share_mode == "global"`:

```python
transformed = self.scalar_kans[0](embeddings)
```

Luc nay toan bo scalar cua moi feature va moi chieu embedding dung chung mot ham:

\[
\phi(v_{i,j}).
\]

Tuc la:

\[
\phi_1=\phi_2=\cdots=\phi_m=\phi.
\]

Global mode tiet kiem tham so hon, nhung interpretability theo feature yeu hon vi shape cua ham khong con rieng cho tung feature.

## 5. Contribution cua tung chieu embedding \(p_{i,j}\)

Sau khi qua ham KAN scalar, ta co:

\[
u_{i,j} = \phi_i(v_{i,j}).
\]

Code tiep theo:

```python
dim_weight = self.dim_weight.to(dtype=transformed.dtype).view(1, 1, -1)
field_weight = self.field_weight.to(dtype=transformed.dtype).view(1, -1)
field_bias = self.field_bias.to(dtype=transformed.dtype).view(1, -1)
```

Trong do:

- `dim_weight` tuong ung \(\rho_j\).
- `field_weight` tuong ung \(a_i\).
- `field_bias` tuong ung \(\beta_i\).

`dim_weight` co shape:

```text
[1, 1, d]
```

No duoc broadcast len `[B, m, d]` khi nhan voi `transformed`.

Contribution cua chieu embedding thu \(j\) trong feature \(i\) co the viet la:

\[
p_{i,j}(\mathbf{x})
=
a_i \rho_j \phi_i(v_{i,j}).
\]

Trong code hien tai, tensor \(p_{i,j}\) chua duoc return truc tiep. Nhung co the tinh tu cac tensor trung gian:

```python
p_ij = transformed * dim_weight * field_weight.unsqueeze(-1)
```

Shape cua `p_ij`:

```text
[B, m, d]
```

Y nghia:

- `p_ij[:, i, j] > 0`: chieu embedding \(j\) cua feature \(i\) dang keo KAN logit tang len.
- `p_ij[:, i, j] < 0`: chieu embedding \(j\) cua feature \(i\) dang keo KAN logit giam xuong.

Luu y: trong tai lieu nay \(p_{i,j}\) la contribution cua embedding dimension. Khong nen nham voi \(p\) trong \(B_{r,p}\), la bac cua B-spline.

## 6. Contribution cua feature \(c_i(\mathbf{x})\)

Code:

```python
contributions = (transformed * dim_weight).sum(dim=2)
contributions = contributions * field_weight + field_bias
```

Dong dau:

```python
(transformed * dim_weight).sum(dim=2)
```

tuong ung voi:

\[
\sum_{j=1}^{d} \rho_j \phi_i(v_{i,j}).
\]

`sum(dim=2)` nghia la cong theo truc embedding dimension. Vi `transformed` co shape `[B, m, d]`, sau khi cong theo `dim=2`, ket qua co shape:

```text
[B, m]
```

Dong tiep theo:

```python
contributions = contributions * field_weight + field_bias
```

tuong ung voi:

\[
c_i(\mathbf{x})
=
a_i
\sum_{j=1}^{d}
\rho_j \phi_i(v_{i,j})
+
\beta_i.
\]

Do do:

\[
c_i(\mathbf{x}) \in \mathbb{R}.
\]

No la mot scalar cho moi feature trong moi sample. Voi batch, tensor `contributions` co shape:

```text
[B, m]
```

Y nghia:

- `contributions[b, i]` la contribution cua feature \(i\) vao KAN logit cua sample \(b\).
- Gia tri duong lam tang KAN logit.
- Gia tri am lam giam KAN logit.
- Tri tuyet doi lon hon thuong co anh huong manh hon.

Co the viet lai:

\[
c_i(\mathbf{x})
=
\sum_{j=1}^{d} p_{i,j}(\mathbf{x})
+
\beta_i.
\]

## 7. KAN logits

Sau khi co contribution cua tung feature, code tinh KAN logit:

```python
logits = contributions.sum(dim=1) + self.bias.to(dtype=transformed.dtype).squeeze(0)
```

`contributions` co shape `[B, m]`.

`sum(dim=1)` cong theo feature fields:

\[
\sum_{i=1}^{m} c_i(\mathbf{x}).
\]

`self.bias` tuong ung voi bias cua ca nhanh KAN:

\[
b_{\mathrm{KAN}}.
\]

Do do:

\[
\operatorname{KAN}(\mathbf{x})
=
b_{\mathrm{KAN}}
+
\sum_{i=1}^{m} c_i(\mathbf{x}).
\]

Hay viet day du:

\[
\operatorname{KAN}(\mathbf{x})
=
b_{\mathrm{KAN}}
+
\sum_{i=1}^{m}
\left(
a_i
\sum_{j=1}^{d}
\rho_j \phi_i(v_{i,j})
+
\beta_i
\right).
\]

Output `logits` co shape:

```text
[B]
```

Day la scalar logit cua KAN branch cho moi sample.

## 8. Fuse voi FIN trong KANFIN

File: `src/models/kanfin.py`

Trong `KANFIN.forward`:

```python
embeddings = self.embedding(x)
kan_logits, contributions = self.kan(embeddings)
fin_logits, attention_weights = self.fin(embeddings)
logits = kan_logits + fin_logits
```

Nghia la KAN va FIN dung chung embedding matrix \(\mathbf{E}\), sau do tao hai logit song song:

\[
\operatorname{KAN}(\mathbf{x})
\]

va:

\[
\operatorname{FIN}(\mathbf{x}).
\]

Sau do cong o muc logit:

\[
\operatorname{KANFI}(\mathbf{x})
=
\operatorname{KAN}(\mathbf{x})
+
\operatorname{FIN}(\mathbf{x}).
\]

Trong training, output nay nen di vao `BCEWithLogitsLoss`, tuc la loss nhan logit truc tiep. Khi evaluate/predict probability thi moi dung sigmoid:

\[
\widehat{y}
=
\sigma(\operatorname{KANFI}(\mathbf{x})).
\]

Dictionary output cua `KANFIN.forward`:

```python
return {
    "logits": logits,
    "kan_logits": kan_logits,
    "fin_logits": fin_logits,
    "attention_weights": attention_weights,
    "feature_contributions": contributions,
}
```

Trong do:

- `"logits"`: logit cuoi cua KANFI.
- `"kan_logits"`: logit cua nhanh KAN.
- `"fin_logits"`: logit cua nhanh FIN.
- `"attention_weights"`: attention map cua FIN.
- `"feature_contributions"`: \(c_i(\mathbf{x})\), shape `[B, m]`.

## 9. FIN branch tinh logits nhu the nao

File: `src/models/fin.py`

FIN branch bat dau tu cung tensor embedding:

```python
out = embeddings
```

Sau do qua nhieu lop multi-head attention:

```python
out, attention_weights = attn(out, out, out, need_weights=True, average_attn_weights=False)
```

O day `attn(out, out, out)` la self-attention:

- Query = `out`
- Key = `out`
- Value = `out`

Vi `batch_first=True`, input/output co shape:

```text
[B, m, d]
```

`attention_weights` khi `average_attn_weights=False` giu rieng tung head, thuong co shape:

```text
[B, num_heads, m, m]
```

Moi entry gan voi muc do feature nay attend den feature kia.

Neu `use_residual=True`:

```python
out = out + residual
```

Day la residual connection:

\[
\mathbf{E}^{\mathrm{Res}}
=
\widetilde{\mathbf{E}}
+
\mathbf{E}.
\]

Sau do:

```python
out = norm(self.activation(out))
```

gom:

- `nn.ReLU()`: giu phan duong, dua phan am ve 0.
- `nn.LayerNorm(embedding_dim)`: chuan hoa moi vector embedding theo chieu `d`.

Cuoi cung:

```python
logits = self.output(out.flatten(start_dim=1)).squeeze(-1)
```

`out.flatten(start_dim=1)` bien tensor `[B, m, d]` thanh `[B, m*d]`.

Sau do MLP:

```python
self.output = MLP(num_fields * embedding_dim, [64, 32], output_dim=1, dropout=dropout)
```

tra ve shape `[B, 1]`, va:

```python
squeeze(-1)
```

dua ve `[B]`.

## 10. Giai thich cac ham PyTorch quan trong

### `nn.Module`

Lop co so cho moi model/layer PyTorch. Khi class ke thua `nn.Module`, PyTorch co the:

- quan ly tham so hoc duoc;
- chuyen model sang GPU/CPU bang `.to(device)`;
- luu/load checkpoint bang `state_dict`;
- goi `forward` khi dung `model(input)`.

### `nn.Parameter`

Dung de khai bao tensor la tham so hoc duoc. Vi du:

```python
self.dim_weight = nn.Parameter(...)
self.field_weight = nn.Parameter(...)
self.field_bias = nn.Parameter(...)
self.bias = nn.Parameter(...)
```

Nhung tensor nay se duoc optimizer cap nhat trong qua trinh backpropagation.

### `nn.ModuleList`

Giong list Python, nhung PyTorch biet cac module ben trong la thanh phan cua model.

Vi du:

```python
self.scalar_kans = nn.ModuleList([...])
```

Neu dung list Python thuong, PyTorch co the khong dang ky tham so cua cac layer ben trong.

### `register_buffer`

Dung de luu tensor thuoc model nhung khong phai tham so hoc duoc.

Trong KAN:

```python
self.register_buffer("grid_min", ...)
self.register_buffer("grid_max", ...)
self.register_buffer("knots", ...)
```

Nhung gia tri nay can di theo device/checkpoint, nhung khong can optimizer cap nhat.

### `torch.stack`

Ghep nhieu tensor cung shape thanh mot tensor moi, tao them mot dimension moi.

Vi du trong embedding:

```python
torch.stack(outputs, dim=1)
```

Neu `outputs` gom \(m\) tensor shape `[B, d]`, ket qua la:

```text
[B, m, d]
```

Trong KAN field mode:

```python
torch.stack([...], dim=1)
```

ghep ket qua cua \(m\) ham \(\phi_i\) thanh tensor `[B, m, d]`.

### Tensor slicing `embeddings[:, idx, :]`

Lay toan bo batch, feature field thu `idx`, va toan bo embedding dimensions.

Neu `embeddings` co shape `[B, m, d]`, thi:

```python
embeddings[:, idx, :]
```

co shape:

```text
[B, d]
```

Day la ly do trong field mode, mot `scalar_kan` nhan ca vector embedding cua feature \(i\), va ap dung cung ham \(\phi_i\) len moi scalar trong vector do.

### `.to(dtype=...)`

Chuyen dtype cua tensor cho khop voi tensor khac.

Vi du:

```python
dim_weight = self.dim_weight.to(dtype=transformed.dtype)
```

Khi dung mixed precision tren GPU, `transformed` co the la `float16` hoac `bfloat16`. Chuyen dtype giup tranh loi hoac cast khong mong muon.

### `.view(...)`

Reshape tensor ma khong doi du lieu.

Vi du:

```python
dim_weight.view(1, 1, -1)
```

neu `dim_weight` ban dau co shape `[d]`, sau `view` se co shape:

```text
[1, 1, d]
```

Shape nay giup broadcast khi nhan voi `transformed` shape `[B, m, d]`.

### Broadcasting

Broadcasting la co che PyTorch tu dong mo rong dimension co size 1 khi thuc hien phep toan.

Vi du:

```python
transformed * dim_weight
```

voi:

```text
transformed: [B, m, d]
dim_weight: [1, 1, d]
```

PyTorch hieu la cung mot vector \(\rho\) duoc dung cho moi sample va moi feature.

### `.sum(dim=...)`

Cong theo mot truc cu the.

Trong KAN:

```python
(transformed * dim_weight).sum(dim=2)
```

cong theo embedding dimension:

\[
\sum_{j=1}^{d} \rho_j \phi_i(v_{i,j}).
\]

Ket qua tu `[B, m, d]` thanh `[B, m]`.

Sau do:

```python
contributions.sum(dim=1)
```

cong theo feature fields:

\[
\sum_{i=1}^{m} c_i(\mathbf{x}).
\]

Ket qua tu `[B, m]` thanh `[B]`.

### `.unsqueeze(-1)`

Them mot dimension co size 1.

Neu can tinh contribution tung chieu:

```python
p_ij = transformed * dim_weight * field_weight.unsqueeze(-1)
```

`field_weight` co shape `[1, m]`, sau `unsqueeze(-1)` thanh:

```text
[1, m, 1]
```

Nhu vay no co the broadcast voi `[B, m, d]`.

### `.squeeze(-1)`

Xoa dimension cuoi neu dimension do co size 1.

Trong FIN:

```python
logits = self.output(...).squeeze(-1)
```

MLP tra ve `[B, 1]`, sau `squeeze(-1)` thanh `[B]`.

### `.reshape(-1)` va `.reshape_as(x)`

Trong KAN spline degree > 1:

```python
basis = self._bspline_basis(clipped.reshape(-1))
out = torch.matmul(basis, spline_weight).reshape_as(x)
```

- `reshape(-1)`: trai phang tensor thanh vector 1 chieu.
- `reshape_as(x)`: dua output ve cung shape voi tensor `x`.

Lam vay giup tinh spline basis tren moi scalar roi tra ve dung shape ban dau.

### `torch.matmul`

Nhan ma tran/vector.

Trong KAN spline:

```python
out = torch.matmul(basis, spline_weight)
```

Neu:

```text
basis: [N, M]
spline_weight: [M]
```

ket qua la:

```text
[N]
```

Tuong ung voi:

\[
\sum_{r=1}^{M}\alpha_r B_r(t).
\]

### `torch.floor`

Lay phan nguyen duoi cua mot gia tri.

Trong linear spline:

```python
lower_idx = torch.floor(position)
```

Dung de tim diem grid ben trai cua scalar input.

### `.clamp(...)`

Gioi han gia tri trong mot khoang.

Vi du:

```python
lower_idx.clamp(min=0, max=self.grid_size - 2)
frac.clamp(0.0, 1.0)
```

Dung de tranh chi so vuot ngoai grid va dam bao ty le noi suy nam trong `[0, 1]`.

### `F.silu`

Ham activation SiLU:

\[
\operatorname{SiLU}(x)=x\sigma(x).
\]

Trong KAN, no la base activation duoc cong voi spline:

```python
out = out + base_weight * F.silu(x)
```

### `nn.Dropout`

Trong KAN:

```python
self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
transformed = self.dropout(transformed)
```

Neu `dropout > 0`, mot phan output sau \(\phi_i\) bi dat ve 0 trong training. Dieu nay giup regularization, nhung cung lam contribution trong train co nhieu nhieu hon. Khi model o eval mode, dropout tat.

### `nn.Identity`

Layer khong lam gi ca. Neu `dropout == 0`, code dung:

```python
nn.Identity()
```

de forward van goi duoc nhu mot layer binh thuong:

```python
transformed = self.dropout(transformed)
```

## 11. Dien giai khi plot ham \(\phi_i(t)\)

Khi plot ham:

\[
y=\phi_i(t),
\]

voi:

\[
t \in [g_{\min}, g_{\max}],
\]

ta dang xem response curve cua feature field \(i\) tren mot scalar embedding.

Y nghia:

- Truc hoanh \(t\): gia tri scalar trong embedding cua feature \(i\).
- Truc tung \(\phi_i(t)\): gia tri sau bien doi phi tuyen cua KAN.
- Neu \(\phi_i(t)\) tang khi \(t\) tang, scalar embedding o vung do co xu huong lam tang contribution truoc khi nhan voi \(\rho_j\) va \(a_i\).
- Neu \(\phi_i(t)\) am, no co the keo logit xuong, tuy con phu thuoc dau cua \(\rho_j\) va \(a_i\).

Can nho rang \(\phi_i(t)\) khong phai truc tiep la contribution cuoi cung cua feature. Contribution cuoi cung la:

\[
c_i(\mathbf{x})
=
a_i
\sum_{j=1}^{d}
\rho_j \phi_i(v_{i,j})
+
\beta_i.
\]

Vay plot \(\phi_i(t)\) cho biet cach KAN bien doi mot scalar embedding, con `feature_contributions` moi la dong gop feature vao KAN logit.

## 12. Ta co the interpret duoc gi?

Voi `share_mode="field"`, co the interpret theo ba muc:

### Muc 1: Ham scalar cua feature

\[
\phi_i(t)
\]

Cho biet feature \(i\) bien doi scalar embedding nhu the nao.

Day la thu co the plot thanh curve.

### Muc 2: Contribution cua tung chieu embedding

\[
p_{i,j}(\mathbf{x})
=
a_i\rho_j\phi_i(v_{i,j}).
\]

Cho biet chieu embedding \(j\) trong feature \(i\) dang dong gop bao nhieu vao KAN logit.

Code hien tai chua return truc tiep `p_ij`, nhung co the tinh them tu tensor trung gian.

### Muc 3: Contribution cua feature

\[
c_i(\mathbf{x})
=
\sum_{j=1}^{d}p_{i,j}(\mathbf{x})
+
\beta_i.
\]

Day la output dang duoc return trong:

```python
"feature_contributions": contributions
```

No cho biet feature \(i\) dong gop bao nhieu vao KAN logit cua tung sample.

## 13. Tom tat mapping cong thuc sang code

| Cong thuc | Y nghia | Code |
|---|---|---|
| \(\mathbf{v}_i\) | embedding vector cua feature \(i\) | `embeddings[:, idx, :]` |
| \(\phi_i(v_{i,j})\) | scalar KAN transform | `scalar_kan(embeddings[:, idx, :])` |
| \(\rho_j\) | trong so cua embedding dimension \(j\) | `self.dim_weight` |
| \(a_i\) | trong so field \(i\) | `self.field_weight` |
| \(\beta_i\) | bias field \(i\) | `self.field_bias` |
| \(p_{i,j}\) | contribution cua dimension \(j\) trong field \(i\) | `transformed * dim_weight * field_weight.unsqueeze(-1)` |
| \(c_i(\mathbf{x})\) | contribution cua feature \(i\) | `contributions` |
| \(b_{\mathrm{KAN}}\) | bias cua KAN branch | `self.bias` |
| \(\operatorname{KAN}(\mathbf{x})\) | KAN branch logit | `logits` trong `KANBranch.forward` |
| \(\operatorname{KANFI}(\mathbf{x})\) | final fused logit | `kan_logits + fin_logits` |

## 14. Ket luan

Trong code hien tai, KAN branch khong flatten moi scalar embedding roi gan moi scalar mot ham rieng. Thay vao do, voi `share_mode="field"`, moi feature field \(i\) co mot ham \(\phi_i\), va ham nay duoc share cho tat ca \(d\) scalar trong embedding vector cua feature do.

Do do, output cua nhanh KAN duoc tinh theo dang additive:

\[
\operatorname{KAN}(\mathbf{x})
=
b_{\mathrm{KAN}}
+
\sum_{i=1}^{m}
\left(
a_i
\sum_{j=1}^{d}
\rho_j\phi_i(v_{i,j})
+
\beta_i
\right).
\]

Thiet ke nay giu duoc interpretability theo feature thong qua \(c_i(\mathbf{x})\), dong thoi van cho phep plot ham \(\phi_i(t)\) de xem response curve cua tung feature field tren khong gian embedding scalar.
