# KAN và các biến thể

Tài liệu này tóm tắt Kolmogorov-Arnold Networks (KAN), nền toán phía sau, thuật ngữ thường gặp, các biến thể quan trọng, và cách suy nghĩ khi đưa KAN vào bài toán CTR dùng embedding categorical. Nội dung dựa trên các paper/repo trong phần tài liệu tham khảo ở cuối file, đặc biệt là KAN gốc, KAN 2.0, pykan, efficient-kan và các biến thể FastKAN, ReLU-KAN, Wav-KAN, Chebyshev KAN, rKAN, ConvKAN, GKAN.

## 1. Ý tưởng rất ngắn

MLP truyền thống học trọng số tuyến tính trên cạnh và đặt activation cố định ở node:

```math
x^{(\ell+1)} = \sigma\left(W^{(\ell)} x^{(\ell)} + b^{(\ell)}\right)
```

KAN đảo góc nhìn này: node chủ yếu chỉ cộng, còn mỗi cạnh học một hàm một biến:

```math
x^{(\ell+1)}_j = \sum_{i=1}^{n_\ell} \phi^{(\ell)}_{j,i}\left(x^{(\ell)}_i\right)
```

Trong đó `phi_{j,i}` không còn là một hệ số vô hướng như `W[j, i]`, mà là một hàm học được, thường được tham số hóa bằng spline hoặc một basis function khác. Vì vậy KAN hay được mô tả là "activation nằm trên edge", còn MLP là "activation nằm trên node".

Điểm hấp dẫn của KAN là:

- Có nền toán từ định lý biểu diễn Kolmogorov-Arnold.
- Mỗi cạnh là một hàm một biến nên có thể vẽ, prune, hoặc thay bằng biểu thức symbolic.
- Với bài toán hàm nhỏ, scientific computing, PDE, hoặc data fitting, KAN đôi khi đạt chất lượng tốt với ít node hơn MLP.
- Điểm yếu là chi phí tính toán/bộ nhớ lớn hơn MLP nếu triển khai ngây thơ, và lợi thế chưa chắc xuất hiện trên bài toán classification lớn, sparse categorical, NLP/CV quy mô lớn.

## 2. Định lý Kolmogorov-Arnold

Định lý Kolmogorov-Arnold representation theorem nói rằng mọi hàm liên tục nhiều biến trên miền compact, ví dụ:

```math
f: [0, 1]^n \rightarrow \mathbb{R}
```

có thể được biểu diễn bằng tổng của các hàm một biến và phép cộng:

```math
f(x_1, \ldots, x_n)
= \sum_{q=0}^{2n}
\Phi_q\left(
  \sum_{p=1}^{n} \phi_{q,p}(x_p)
\right)
```

Ý nghĩa:

- `f` là hàm nhiều biến cần học.
- `x_p` là biến đầu vào thứ `p`.
- `phi_{q,p}` và `Phi_q` là các hàm một biến.
- Tương tác nhiều biến được tạo ra bằng composition và tổng, không cần một activation nhiều biến trực tiếp.

Một hiểu nhầm hay gặp: định lý này không tự động nói KAN sẽ luôn tốt hơn MLP trong thực nghiệm. Định lý đảm bảo tồn tại biểu diễn, nhưng không nói biểu diễn đó dễ học, smooth, ít tham số, ổn định gradient, hay hiệu quả trên GPU. KAN là một cách biến cảm hứng từ định lý này thành kiến trúc neural network có thể train.

## 3. Từ MLP layer sang KAN layer

### 3.1 MLP layer

Với input `x in R^{n_in}`, output `y in R^{n_out}`:

```math
y_j = \sigma\left(\sum_{i=1}^{n_{in}} W_{j,i}x_i + b_j\right)
```

Trong MLP:

- Mỗi cạnh chỉ có một số `W_{j,i}`.
- Phi tuyến `sigma` dùng chung theo node, ví dụ ReLU, GELU, SiLU.
- Biểu diễn của layer là linear transform rồi activation cố định.

### 3.2 KAN layer

Trong KAN:

```math
y_j = \sum_{i=1}^{n_{in}} \phi_{j,i}(x_i)
```

Trong đó:

```math
\phi_{j,i}: \mathbb{R} \rightarrow \mathbb{R}
```

là một hàm học được trên edge từ input node `i` sang output node `j`.

Một KAN nhiều layer là composition của các KAN layer:

```math
\mathrm{KAN}(x) =
\Phi^{(L-1)} \circ \Phi^{(L-2)} \circ \cdots \circ \Phi^{(0)}(x)
```

với mỗi `Phi^(l)` là ma trận các hàm một biến:

```math
\Phi^{(\ell)} =
\begin{bmatrix}
\phi^{(\ell)}_{1,1}(\cdot) & \cdots & \phi^{(\ell)}_{1,n_\ell}(\cdot) \\
\vdots & \ddots & \vdots \\
\phi^{(\ell)}_{n_{\ell+1},1}(\cdot) & \cdots & \phi^{(\ell)}_{n_{\ell+1},n_\ell}(\cdot)
\end{bmatrix}
```

Đếm tham số sơ bộ:

- MLP layer: khoảng `n_in * n_out` trọng số.
- KAN layer: khoảng `n_in * n_out * P_phi`, với `P_phi` là số tham số để mô tả một hàm edge.

Vì mỗi cạnh chứa cả một hàm, KAN có thể rất biểu cảm nhưng cũng dễ phình tham số và compute nếu `n_in`, `n_out` lớn.

## 4. Spline KAN gốc

KAN gốc thường dùng B-spline để tham số hóa mỗi edge function. Một dạng thường gặp:

```math
\phi(x) = w_b b(x) + w_s \operatorname{spline}(x)
```

Trong đó:

```math
\operatorname{spline}(x) =
\sum_{r=1}^{G+k} c_r B_{r,k}(x)
```

Giải thích:

- `b(x)` là base function, thường là một activation trơn như SiLU.
- `w_b`, `w_s` là scale cho base và spline branch.
- `B_{r,k}(x)` là B-spline basis thứ `r`, bậc/order `k`.
- `c_r` là coefficient học được.
- `G` là grid size, tức số khoảng hoặc số điểm lưới dùng để dựng spline.
- `k` là spline order/degree tùy cách ký hiệu trong implementation.

### 4.1 B-spline là gì?

Spline là hàm ghép từ nhiều đa thức cục bộ. B-spline là một hệ basis để biểu diễn spline:

```math
s(x) = \sum_r c_r B_r(x)
```

Điểm quan trọng:

- Mỗi basis `B_r` chỉ khác 0 trên một vùng nhỏ, gọi là local support.
- Vì local support, đổi một coefficient thường chỉ ảnh hưởng một đoạn của hàm.
- Spline thường mượt hơn piecewise linear thuần túy, tùy bậc spline.

### 4.2 Grid, knot, spline order

Các thuật ngữ này thường làm người mới đọc KAN hơi khựng lại:

- `grid`: lưới các điểm chia miền input của một edge function.
- `knot`: điểm mốc của spline; các đoạn đa thức nối nhau tại knot.
- `grid_size`: số điểm/khoảng của lưới. Grid lớn thì hàm linh hoạt hơn nhưng nhiều tham số hơn.
- `spline_order`: bậc/order của spline. Order cao hơn thường mượt hơn nhưng tính toán phức tạp hơn.
- `grid update`: cập nhật grid theo phân phối input thực tế, giúp vùng dữ liệu dày được biểu diễn tốt hơn.
- `base function`: nhánh activation đơn giản như SiLU để giữ khả năng ngoại suy và ổn định training.
- `local support`: mỗi basis chỉ tác động gần một đoạn input, giúp diễn giải theo vùng.

### 4.3 Regularization, sparsity, pruning

KAN thường đi kèm interpretability workflow:

1. Train model.
2. Dùng regularization để khuyến khích một số edge function nhỏ đi.
3. Prune edge/node ít quan trọng.
4. Vẽ các hàm còn lại.
5. Nếu bài toán khoa học, thử symbolic regression để thay spline bằng công thức như `sin`, `exp`, `x^2`, `log`.

Regularization hay gặp:

- L1 trên độ lớn activation hoặc coefficient để làm edge thưa.
- Entropy/sparsity penalty để model dùng ít đường truyền hơn.
- Weight decay thông thường.

Pruning không làm KAN chính xác hơn một cách tự động; nó là cách đổi một phần accuracy lấy mô hình gọn và dễ đọc hơn.

## 5. Tại sao KAN có thể interpretable?

Vì mỗi edge là một hàm một biến, ta có thể vẽ:

```math
x_i \mapsto \phi_{j,i}(x_i)
```

Nếu edge này gần như tuyến tính, parabolic, sinusoidal, saturating, hoặc threshold-like, con người có thể nhìn ra cấu trúc. Trong MLP, mỗi trọng số chỉ là một số, còn phi tuyến nằm ở node và bị trộn qua ma trận; khó gán ý nghĩa cho từng cạnh.

Nhưng interpretability của KAN có điều kiện:

- Nếu input là biến vật lý rõ nghĩa, ví dụ thời gian, vị trí, nhiệt độ, vận tốc, thì edge function dễ diễn giải.
- Nếu input là embedding latent của categorical hash bucket, edge function nằm trên tọa độ embedding, không phải category gốc. Khi đó diễn giải yếu hơn nhiều.
- Nếu model quá rộng/sâu, số edge function lớn, việc vẽ mọi hàm không còn thực tế.

## 6. Chi phí tính toán và bộ nhớ

Vấn đề lớn của KAN gốc là mỗi edge có hàm riêng. Với batch `B`, input width `I`, output width `O`, nếu triển khai ngây thơ có thể phải tạo tensor trung gian:

```math
[B, O, I]
```

hoặc lớn hơn nếu expand thêm basis:

```math
[B, O, I, G]
```

Đây là lý do KAN có thể chậm hoặc tốn VRAM dù số node ít. `efficient-kan` cải tiến bằng cách tách việc tính basis và tổ hợp tuyến tính để đưa về dạng gần matrix multiplication hơn. Các biến thể như FastKAN, ReLU-KAN, Fourier KAN cũng chủ yếu cố giảm chi phí basis và tăng độ thân thiện với GPU.

## 7. Các biến thể theo basis function

KAN gốc là Spline KAN. Hầu hết biến thể sau đó thay `spline(x)` bằng một họ basis khác.

### 7.1 Spline KAN

Edge function:

```math
\phi(x) = w_b b(x) + w_s \sum_r c_r B_{r,k}(x)
```

Ưu điểm:

- Local support tốt cho diễn giải.
- Có thể biểu diễn hàm trơn theo từng đoạn.
- Hợp với function fitting, PDE nhỏ, symbolic discovery.

Nhược điểm:

- Tính B-spline và grid update phức tạp.
- GPU utilization thường kém hơn Linear/ReLU.
- Dễ phình bộ nhớ khi layer rộng.

### 7.2 Efficient KAN

Efficient KAN không hẳn là một kiến trúc mới, mà là cách triển khai lại Spline KAN để tiết kiệm memory. Ý tưởng là tính basis B-spline theo input rồi combine tuyến tính, thay vì expand input theo mọi output-edge trước.

Tư duy:

```math
B_r(x_i) \rightarrow \text{basis features}
```

sau đó dùng coefficient để tổ hợp thành output.

Ưu điểm:

- Giảm memory so với implementation gốc.
- Dễ tích hợp PyTorch hơn.

Nhược điểm:

- Một số regularization dựa trên activation tensor đầy đủ của KAN gốc không còn giữ nguyên.
- Đây là tối ưu implementation, không tự giải quyết vấn đề layer quá rộng.

### 7.3 FastKAN / RBF-KAN

FastKAN dựa trên quan sát rằng cubic B-spline có thể được xấp xỉ bằng Gaussian radial basis functions (RBF). Edge function có thể viết gần như:

```math
\phi(x) =
\sum_{m=1}^{M} a_m
\exp\left(-\gamma (x - \mu_m)^2\right)
```

Trong đó:

- `mu_m` là center của RBF.
- `gamma` hoặc `sigma` điều khiển độ rộng kernel.
- `a_m` là coefficient học được.

Ưu điểm:

- RBF dễ vectorize.
- Có local behavior tương tự spline.
- Thường nhanh hơn Spline KAN.

Nhược điểm:

- Cần chọn center và bandwidth.
- Nếu bandwidth không hợp, basis quá nhọn hoặc quá phẳng.
- Interpretability vẫn là function-level, không tự động thành symbolic formula.

### 7.4 ReLU-KAN

ReLU-KAN thay basis phức tạp bằng phép cộng ma trận, nhân từng phần tử và ReLU. Mục tiêu là giữ tinh thần KAN nhưng dùng primitive rất thân thiện với CUDA.

Ý tưởng tổng quát:

```math
\phi(x) = \sum_m a_m \rho_m(x)
```

với `rho_m` được tạo từ các tổ hợp ReLU và point-wise multiplication thay vì B-spline recursion.

Ưu điểm:

- Tốc độ tốt hơn KAN spline trong báo cáo của paper.
- Dễ triển khai bằng PyTorch/TensorFlow thông thường.
- Training có thể ổn định hơn khi tránh spline basis phức tạp.

Nhược điểm:

- Basis piecewise-linear/piecewise-polynomial có inductive bias khác spline trơn.
- Diễn giải hình dạng hàm có thể kém "mượt" hơn spline.

### 7.5 Chebyshev KAN

Chebyshev KAN dùng đa thức Chebyshev làm basis:

```math
\phi(x) = \sum_{r=0}^{d} c_r T_r(\tilde{x})
```

với `\tilde{x}` thường được chuẩn hóa vào `[-1, 1]`. Chebyshev polynomial thỏa recurrence:

```math
T_0(x)=1,\quad T_1(x)=x,\quad
T_{r+1}(x)=2xT_r(x)-T_{r-1}(x)
```

Ưu điểm:

- Tính toán bằng recurrence đơn giản.
- Mạnh cho hàm trơn và approximation trên khoảng hữu hạn.
- Orthogonality giúp giảm tương quan giữa basis hơn đa thức thường.

Nhược điểm:

- Basis có tính global: đổi coefficient có thể ảnh hưởng toàn miền.
- Degree cao dễ dao động nếu normalization hoặc regularization không tốt.
- Ít local interpretability hơn B-spline/RBF.

### 7.6 Wav-KAN

Wav-KAN dùng wavelet basis để bắt cả thành phần tần số thấp và cao:

```math
\phi(x) =
\sum_{s,t} c_{s,t}
\psi\left(\frac{x - t}{s}\right)
```

Trong đó:

- `psi` là mother wavelet.
- `s` là scale, điều khiển độ co giãn.
- `t` là translation, điều khiển vị trí.
- `c_{s,t}` là coefficient.

Ưu điểm:

- Hợp với tín hiệu có cấu trúc đa độ phân giải.
- Bắt pattern cục bộ và pattern toàn cục tốt hơn một basis đơn scale.
- Có ý nghĩa mạnh trong signal/time-series.

Nhược điểm:

- Nhiều lựa chọn wavelet/scale làm tăng hyperparameter.
- Có thể quá nặng cho tabular CTR nếu không có cấu trúc tín hiệu rõ.

### 7.7 Fourier KAN / KAF / FKAN

Fourier KAN dùng basis sin/cos hoặc Random Fourier Features:

```math
\phi(x) =
a_0 +
\sum_{m=1}^{M}
a_m \cos(m\omega x) +
b_m \sin(m\omega x)
```

Hoặc dạng random Fourier:

```math
\phi(x) =
\sum_{m=1}^{M}
a_m \cos(\omega_m x + \beta_m)
```

Ưu điểm:

- Mạnh với periodic pattern và high-frequency signal.
- Có thể phù hợp implicit neural representation, ảnh, audio, PDE.
- Một số biến thể spectral reparameterization cố giảm parameter complexity.

Nhược điểm:

- Basis global nên local interpretability yếu hơn spline.
- Dễ sinh oscillation nếu tần số cao không được kiểm soát.
- Với categorical embedding, periodic inductive bias không luôn hợp lý.

### 7.8 rKAN

rKAN dùng rational functions, ví dụ Pade approximation hoặc rational Jacobi functions:

```math
\phi(x) =
\frac{P_m(x)}{Q_n(x)}
=
\frac{\sum_{i=0}^{m} a_i x^i}
     {1 + \sum_{j=1}^{n} b_j x^j}
```

Ưu điểm:

- Hợp với hàm có asymptote, saturation, decay, hoặc biến thiên global.
- Có khả năng xấp xỉ tốt hơn polynomial thuần trong một số hàm khó.

Nhược điểm:

- Cần tránh mẫu số gần 0.
- Training có thể nhạy hơn vì hàm rational có singularity.
- Ít thân thiện hơn ReLU/RBF về ổn định số học.

## 8. Các biến thể theo kiến trúc

### 8.1 MultKAN / KAN 2.0

KAN gốc chủ yếu dùng phép cộng ở node. KAN 2.0 giới thiệu MultKAN, thêm multiplication nodes để mô hình hóa cấu trúc dạng nhân:

```math
z = u \cdot v
```

Điều này quan trọng trong khoa học vì nhiều định luật có cấu trúc tích, ví dụ:

```math
E = \frac{1}{2}mv^2
```

hoặc các biểu thức Lagrangian, conserved quantities, constitutive laws.

Ưu điểm:

- Dễ khớp biểu thức symbolic có tích/chia/lũy thừa hơn KAN chỉ cộng.
- Hỗ trợ scientific discovery tốt hơn.

Nhược điểm:

- Multiplication làm optimization và range của activation nhạy hơn.
- Cần kiểm soát scale/normalization.

KAN 2.0 cũng nhấn mạnh các công cụ:

- `kanpiler`: biên dịch công thức symbolic thành KAN topology.
- `tree converter`: chuyển KAN hoặc neural network thành tree graph dễ diễn giải.
- workflow hai chiều: dùng tri thức khoa học để thiết kế KAN, và dùng KAN để gợi ý tri thức khoa học.

### 8.2 ConvKAN / Convolutional KAN

ConvKAN đưa edge function học được vào convolution. CNN thường dùng kernel tuyến tính:

```math
y[p] =
\sum_{\Delta \in \mathcal{K}}
W[\Delta]\,x[p+\Delta]
```

ConvKAN thay hệ số tuyến tính bằng hàm phi:

```math
y[p] =
\sum_{\Delta \in \mathcal{K}}
\phi_{\Delta}(x[p+\Delta])
```

Ưu điểm:

- Giữ inductive bias local receptive field của CNN.
- Mỗi vị trí kernel có nonlinear function riêng.
- Có thể giảm tham số trong một số setup vision nhỏ.

Nhược điểm:

- Chậm hơn convolution chuẩn vốn đã được tối ưu cực mạnh.
- Với ảnh lớn, overhead edge function là vấn đề nghiêm túc.

### 8.3 GKAN

GKAN đưa KAN vào graph neural network. GNN message passing thường có dạng:

```math
h_v^{(l+1)}
=
\operatorname{AGG}_{u \in \mathcal{N}(v)}
M(h_v^{(l)}, h_u^{(l)}, e_{u,v})
```

GKAN dùng spline/KAN function trên edge/message để tăng khả năng diễn giải:

```math
m_{u \rightarrow v}
=
\Phi_{\text{KAN}}(h_u, h_v, e_{u,v})
```

Ưu điểm:

- Hợp dữ liệu network-like: node classification, link prediction, graph classification.
- Có thể xem edge/message function để giải thích.

Nhược điểm:

- Graph batch vốn đã khó tối ưu; KAN làm chi phí tăng thêm.
- Interpretability phụ thuộc feature node/edge có ý nghĩa hay không.

### 8.4 KAN Autoencoder

KAN Autoencoder thay MLP encoder/decoder bằng KAN layer:

```math
z = E_{\text{KAN}}(x), \quad \hat{x}=D_{\text{KAN}}(z)
```

Ưu điểm:

- Hữu ích nếu muốn latent representation có edge function dễ xem hơn.
- Có thể thử cho anomaly detection hoặc compression nhỏ.

Nhược điểm:

- Với dữ liệu lớn/ảnh phức tạp, autoencoder thường cần convolution/attention; KAN thuần có thể không đủ hiệu quả.

### 8.5 KAN Attention / KArAt

Một hướng mới là dùng learnable activation kiểu KAN trong attention/token interaction. Attention chuẩn:

```math
\operatorname{Attention}(Q,K,V)
=
\operatorname{softmax}\left(\frac{QK^\top}{\sqrt{d}}\right)V
```

KArAt đặt câu hỏi liệu token interaction có thể học qua các basis kiểu KAN như Fourier, wavelet, spline, rational. Paper này cũng nêu rõ vấn đề memory explosion và dùng low-rank approximation để giảm chi phí.

Ưu điểm:

- Có thể học attention kernel linh hoạt hơn softmax cố định.
- Có thể chọn basis theo dữ liệu.

Nhược điểm:

- Attention đã là phần tốn memory; thêm KAN activation dễ bùng VRAM.
- Cần approximation low-rank hoặc modularization để thực tế.

## 9. Bảng so sánh nhanh

| Biến thể | Basis/ý tưởng | Mạnh ở đâu | Rủi ro chính |
|---|---|---|---|
| Spline KAN | B-spline trên edge | Function fitting, interpretability, PDE nhỏ | Chậm, tốn memory khi rộng |
| Efficient KAN | Reformulate spline computation | Giảm memory implementation | Không đổi bản chất phình edge |
| FastKAN/RBF-KAN | Gaussian RBF xấp xỉ spline | Nhanh hơn, local basis | Nhạy bandwidth/center |
| ReLU-KAN | ReLU + pointwise multiplication | GPU-friendly, đơn giản | Basis kém mượt hơn spline |
| Chebyshev KAN | Chebyshev polynomial | Hàm trơn, scientific approximation | Global basis, có thể dao động |
| Wav-KAN | Wavelet basis | Signal/time-series, multi-resolution | Nhiều hyperparameter |
| Fourier KAN/KAF/FKAN | Sin/cos, RFF, spectral basis | Periodic/high-frequency, INR, audio/ảnh | Global oscillation, local interpretability yếu |
| rKAN | Rational functions | Asymptote, decay, symbolic-like functions | Singularities, ổn định số học |
| MultKAN | Thêm multiplication nodes | Scientific discovery, biểu thức có tích | Scale/optimization nhạy |
| ConvKAN | KAN trong convolution | Vision nhỏ, local spatial pattern | Chậm hơn conv chuẩn |
| GKAN | KAN trong message passing | Graph tasks, edge/message interpretability | Chi phí graph + KAN |
| KArAt | KAN-like learnable attention | Vision Transformer/token interaction | Memory explosion |

## 10. Liên hệ với CTR và embedding categorical

Trong Avazu CTR, input gốc là categorical/hash bucket:

```text
category -> stable hash bucket -> embedding vector -> model
```

Nếu đưa KAN sau embedding, KAN không học trực tiếp:

```math
f(\text{category})
```

mà học:

```math
f(e_1, e_2, \ldots, e_D)
```

với `e_d` là tọa độ embedding latent. Vì vậy:

- KAN vẫn có thể tăng năng lực nonlinear của branch additive.
- Interpretability theo "category nào làm CTR tăng" không tự nhiên.
- Hash collision làm diễn giải category-level càng khó.
- Với CTR quy mô lớn, biến thể nhỏ/gọn đáng thử hơn KAN full per-edge.

### 10.1 NAM hiện tại và KAN shared-scalar

NAM branch hiện tại trong project có dạng:

```math
\operatorname{NAM}(E)
=
\sum_{f=1}^{F}
g_f(E_f)
```

Trong đó:

- `F` là số field.
- `E_f in R^D` là embedding của field `f`.
- `g_f` là MLP riêng cho field `f`.

Nếu dùng KAN full theo từng field, ta có thể viết:

```math
\operatorname{KANAdd}(E)
=
\sum_{f=1}^{F}
g^{KAN}_f(E_f)
```

Nhưng cách này vẫn có nhiều hàm vector-to-scalar riêng. Với input là embedding cho từng feature, thiết kế cân bằng hơn là: mỗi feature `f` có một hàm scalar KAN riêng `phi_f`, và mọi scalar trong embedding của feature đó dùng chung `phi_f`.

```math
z_{b,f,d} = \phi_{f,\theta}(E_{b,f,d})
```

Trong đó cùng một `phi_f` được share qua các chiều embedding `d` của feature `f`, nhưng feature khác có hàm khác. Sau đó gom theo embedding dimension:

```math
c_{b,f}
=
a_f \sum_{d=1}^{D} w_d z_{b,f,d} + \beta_f
```

và logit additive:

```math
\operatorname{logit}^{KAN}_b
=
\sum_{f=1}^{F} c_{b,f} + b
```

Thiết kế này có tham số xấp xỉ:

```math
P \approx F P_\phi + D + 2F + 1
```

Nếu `phi` là spline nhỏ với `G` coefficient:

```math
P_\phi \approx G + 2
```

Trong implementation hiện tại của project, `kan.degree` điều khiển bậc spline. `degree=1` giữ behavior cũ là piecewise-linear interpolation; `degree=3` dùng cubic B-spline basis. Khi `degree > 1`, số basis của mỗi hàm scalar là xấp xỉ:

```math
G_{\text{basis}} = grid\_size + degree - 1
```

Trong khi NAM per-field MLP có tham số khoảng:

```math
P_{NAM}
\approx
F \cdot \left(Dh_1 + h_1h_2 + h_2\right)
```

Với `F=25`, `D=16`, `h_1=32`, `h_2=16`, NAM branch lên tới hàng chục nghìn tham số; per-feature shared-scalar KAN thường chỉ vài trăm tham số tùy `G`.

Một biến thể còn gọn hơn là global sharing:

```math
z_{b,f,d} = \phi_\theta(E_{b,f,d})
```

tức mọi feature và mọi chiều embedding cùng dùng một hàm `phi`. Cách này ít tham số nhất, nhưng interpretability theo feature yếu hơn vì các feature không có shape function riêng.

### 10.2 Khi nào nên dùng KAN trong CTR?

Nên thử:

- Muốn branch additive nhỏ hơn NAM.
- Muốn regularize mạnh phần nonlinear additive, để FIN/attention lo interaction.
- Muốn so sánh A/B giữa `NAFI` và `KANFIN`.
- Muốn inspect shape của hàm shared scalar trên embedding distribution.

Không nên kỳ vọng quá mức:

- KAN không tự giải quyết sparse categorical.
- KAN không thay thế embedding table.
- KAN không đảm bảo AUC tăng.
- Interpretability sẽ thấp hơn KAN trên biến vật lý/numeric có nghĩa.

## 11. Checklist chọn biến thể

Nếu mục tiêu là memory-safe CTR trên Kaggle:

1. Bắt đầu với shared-scalar Spline KAN hoặc RBF/FastKAN nhỏ.
2. Giữ `grid_size` nhỏ, ví dụ `3`, `5`, hoặc `7`.
3. Không dùng KAN layer full `F*D -> hidden` rộng ngay từ đầu.
4. So parameter count của branch additive với NAM.
5. Log riêng `kan_logits`, `fin_logits`, AUC/logloss của từng branch nếu có.
6. Dùng mixed precision cẩn thận; spline/RBF cần tránh dtype overflow/underflow.
7. Chạy debug/small trước full Avazu.

Nếu mục tiêu là scientific discovery:

1. Dùng Spline KAN hoặc MultKAN.
2. Normalize input về khoảng ổn định.
3. Train nhỏ, prune, refine grid.
4. Vẽ edge functions.
5. Dùng symbolic regression hoặc pykan symbolic tools.

Nếu mục tiêu là signal/time-series:

1. Thử Wav-KAN nếu có multi-resolution pattern.
2. Thử Fourier KAN nếu có periodic/high-frequency pattern.
3. Kiểm soát số basis để tránh overfit.

Nếu mục tiêu là vision/graph:

1. Thử ConvKAN/GKAN ở benchmark nhỏ trước.
2. So wall-clock, VRAM, throughput, không chỉ accuracy.
3. Với model lớn, cân nhắc chỉ thay MLP head/block nhỏ bằng KAN.

## 12. Các hạn chế cần nhớ

KAN là một hướng rất hay, nhưng chưa phải "MLP killer" tổng quát.

- Trên bài toán classification phức tạp, có báo cáo cho thấy KAN không vượt MLP và tốn tài nguyên phần cứng hơn.
- MLP/ReLU/GELU đã được tối ưu cực sâu trên GPU; KAN basis function khó cạnh tranh throughput.
- KAN dễ đẹp trên toy/scientific functions nhỏ, nhưng chưa chắc mạnh trên dữ liệu nhiễu, sparse, high-cardinality.
- Số edge function tăng theo `n_in * n_out`; layer rộng có thể rất đắt.
- Interpretability chỉ tốt khi input dimension có ý nghĩa.
- Với embedding latent, shape của edge function giải thích "tọa độ embedding", không trực tiếp giải thích feature gốc.

## 13. Thuật ngữ nhanh

- `KAN`: Kolmogorov-Arnold Network, mạng có learnable univariate functions trên edges.
- `KAT`: Kolmogorov-Arnold representation theorem.
- `Edge function`: hàm một biến nằm trên cạnh giữa hai node.
- `Node`: trong KAN thường là nơi cộng các edge outputs.
- `Basis function`: họ hàm cơ sở để biểu diễn edge function, ví dụ B-spline, RBF, Fourier, wavelet.
- `Coefficient`: hệ số học được của basis.
- `B-spline`: basis spline có local support.
- `Knot`: điểm mốc chia miền input của spline.
- `Grid`: lưới knot/interval của spline.
- `Grid size`: độ phân giải của lưới.
- `Spline order`: bậc/order của spline.
- `Local support`: basis chỉ khác 0 trên một vùng nhỏ.
- `RBF`: radial basis function, thường là Gaussian quanh một center.
- `Wavelet`: basis có scale và translation, hợp multi-resolution.
- `Chebyshev polynomial`: đa thức trực giao trên `[-1,1]`, tính bằng recurrence.
- `Fourier basis`: sin/cos basis, hợp tín hiệu tuần hoàn/tần số.
- `Rational function`: tỉ số hai đa thức.
- `Pruning`: bỏ edge/node ít quan trọng.
- `Symbolic regression`: tìm công thức symbolic gần với hàm học được.
- `MultKAN`: KAN có multiplication nodes.
- `KANFIN`: trong ngữ cảnh project này, nhánh KAN additive cộng với nhánh FIN interaction.

## 14. Tài liệu tham khảo

- Liu et al., "KAN: Kolmogorov-Arnold Networks", arXiv:2404.19756: https://arxiv.org/abs/2404.19756
- Liu et al., "KAN 2.0: Kolmogorov-Arnold Networks Meet Science", arXiv:2408.10205: https://arxiv.org/abs/2408.10205
- `pykan` official repository: https://github.com/KindXiaoming/pykan
- `efficient-kan` repository: https://github.com/Blealtan/efficient-kan
- Somvanshi et al., "A Survey on Kolmogorov-Arnold Network", arXiv:2411.06078: https://arxiv.org/abs/2411.06078
- Li, "Kolmogorov-Arnold Networks are Radial Basis Function Networks", arXiv:2405.06721: https://arxiv.org/abs/2405.06721
- Qiu et al., "ReLU-KAN: New Kolmogorov-Arnold Networks that Only Need Matrix Addition, Dot Multiplication, and ReLU", arXiv:2406.02075: https://arxiv.org/abs/2406.02075
- Bozorgasl and Chen, "Wav-KAN: Wavelet Kolmogorov-Arnold Networks", arXiv:2405.12832: https://arxiv.org/abs/2405.12832
- Sidharth et al., "Chebyshev Polynomial-Based Kolmogorov-Arnold Networks", arXiv:2405.07200: https://arxiv.org/abs/2405.07200
- Aghaei, "rKAN: Rational Kolmogorov-Arnold Networks", arXiv:2406.14495: https://arxiv.org/abs/2406.14495
- Mehrabian et al., "Implicit Neural Representations with Fourier Kolmogorov-Arnold Networks", arXiv:2409.09323: https://arxiv.org/abs/2409.09323
- Zhang et al., "Kolmogorov-Arnold Fourier Networks", arXiv:2502.06018: https://arxiv.org/abs/2502.06018
- Bodner et al., "Convolutional Kolmogorov-Arnold Networks", arXiv:2406.13155: https://arxiv.org/abs/2406.13155
- De Carlo et al., "Kolmogorov-Arnold Graph Neural Networks", arXiv:2406.18354: https://arxiv.org/abs/2406.18354
- Tran et al., "Exploring the Limitations of Kolmogorov-Arnold Networks in Classification", arXiv:2407.17790: https://arxiv.org/abs/2407.17790
- Schoots et al., "Relating Piecewise Linear Kolmogorov Arnold Networks to ReLU Networks", arXiv:2503.01702: https://arxiv.org/abs/2503.01702
- Maity et al., "Kolmogorov-Arnold Attention: Is Learnable Attention Better For Vision Transformers?", arXiv:2503.10632: https://arxiv.org/abs/2503.10632
