# Exploratory Data Analysis for the Avazu CTR Dataset

Tai lieu nay trinh bay ket qua EDA tren file raw CSV truoc khi ap dung hashing tricks. Ket qua duoc tong hop tu:

- `/Users/w4ng/Downloads/raw_eda_report.md`
- `/Users/w4ng/Downloads/raw_eda_summary.json`
- `/Users/w4ng/Downloads/raw_column_summary.csv`
- `/Users/w4ng/Downloads/raw_top_values.csv`
- `/Users/w4ng/Downloads/raw_time_ctr.csv`

## 1. Dataset overview

Avazu la bo du lieu click-through rate prediction cho quang cao truc tuyen. Moi dong du lieu bieu dien mot ad impression, voi target `click` cho biet nguoi dung co click vao quang cao hay khong.

Bo du lieu gom cac nhom feature chinh:

- `hour`: thoi diem impression, duoc ma hoa theo dang `YYMMDDHH`.
- `site_*`: thong tin inventory phia website, gom `site_id`, `site_domain`, `site_category`.
- `app_*`: thong tin inventory phia app, gom `app_id`, `app_domain`, `app_category`.
- `device_*`: thong tin thiet bi/nguoi dung, gom `device_id`, `device_ip`, `device_model`, `device_type`, `device_conn_type`.
- `C1`, `C14`-`C21`: cac categorical feature duoc an danh.
- `banner_pos`, `C15`, `C16`: cac bien lien quan den vi tri/kich thuoc hien thi quang cao.

Ket qua scan:

| Metric | Value |
|---|---:|
| Rows scanned | 32,343,175 |
| Chunks scanned | 130 |
| Target column | `click` |
| Click count | 5,492,054 |
| Click rate | 0.169806 |
| Non-click rate | 0.830194 |

CTR trung binh cua bo du lieu la:

\[
\operatorname{CTR}
=
\frac{\#click}{\#impression}
=
16.98\%.
\]

Day la mot bai toan binary classification bi lech nhan: so mau non-click lon hon click khoang 4.9 lan. Vi vay, khi train model nen uu tien cac metric nhu AUC va logloss thay vi chi nhin accuracy.

## 2. Data quality note

Ket qua EDA nay duoc tinh bang streaming CSV chunks, phu hop voi rang buoc khong load toan bo Avazu train file vao RAM.

Tuy nhien, ban report hien tai co mot diem can chu y: mot so feature trong `raw_eda_summary.json` co `rows_seen = 64,686,350`, gap doi `rows_scanned = 32,343,175`. Nguyen nhan la script EDA ban dau ghep `AVAZU_COLUMNS + feature_cols`, trong khi nhieu feature xuat hien trong ca hai danh sach. Do do, cac cot goc nhu `C1`, `site_id`, `app_id`, `device_ip`, `hour` da bi update hai lan.

Anh huong:

- `count` va `frequency` trong bang top raw values cua cac feature goc bi phong dai.
- `rows_seen` cua cac feature goc bi gap doi.
- `missing_rate` van dung vi ca missing count va rows_seen cung bi gap doi.
- `ctr` cua top values gan nhu van dung vi click sum va count cung bi gap doi.
- `day`, `weekday`, `hour_of_day` trong bang time CTR khong bi anh huong boi loi top-value loop.

Trong repo, script `scripts/eda_raw_csv.py` da duoc sua de deduplicate danh sach cot truoc khi scan. Nen chay lai EDA de lay lai count/frequency chinh xac neu can dua bang so lieu vao paper.

Ngoai ra, mot so cot co `approx_distinct = 4095`. Vi KMV sketch size la 4096, nhung gia tri sat nguong nay nen duoc doc nhu mot uoc luong/lower-bound can than, khong nen khang dinh day la so unique chinh xac.

## 3. Missing values

Tat ca cac cot trong report co `missing_rate = 0.0`. Dieu nay cho thay file raw CSV khong co missing value theo nghia `NA`/empty string trong qua trinh scan.

Tuy nhien, voi Avazu, nhieu gia tri "unknown" hoac "not available" duoc ma hoa thanh categorical value binh thuong. Vi du:

- `device_id = a99f214a` xuat hien rat pho bien, thuong duoc hieu nhu mot ID mac dinh/unknown.
- `C20 = -1` la mot gia tri dac biet, khong nen coi nhu numeric missing theo cach thong thuong.

Do do, trong preprocessing nen giu cac gia tri nay nhu category rieng va dua vao hashing/embedding, thay vi impute bang mean hoac drop row.

## 4. Cardinality analysis

Avazu la bo du lieu gan nhu hoan toan categorical va co cardinality rat lech giua cac feature.

Nhom cardinality rat cao:

| Feature | Approx distinct |
|---|---:|
| `device_ip` | 6,070,657 |
| `device_id` | 2,327,754 |
| `site_id` | about 4,095+ |
| `site_domain` | about 4,095+ |
| `app_id` | about 4,095+ |
| `device_model` | about 4,095+ |

Nhom cardinality trung binh:

| Feature | Approx distinct |
|---|---:|
| `C14` | 2,614 |
| `app_domain` | 529 |
| `C17` | 435 |
| `hour` | 240 |
| `C20` | 172 |
| `C19` | 68 |
| `C21` | 60 |

Nhom cardinality thap:

| Feature | Approx distinct |
|---|---:|
| `C1` | 7 |
| `banner_pos` | 7 |
| `device_type` | 5 |
| `device_conn_type` | 4 |
| `C15` | 8 |
| `C16` | 9 |
| `C18` | 4 |
| `hour_of_day` | 24 |
| `weekday` | 7 |

He qua quan trong la khong nen one-hot encode truc tiep toan bo raw category. Cac feature nhu `device_ip` va `device_id` co hang trieu gia tri, nen hashing trick va embedding la lua chon hop ly de giu bo nho an toan tren Kaggle.

## 5. Target distribution

CTR trung binh:

\[
\operatorname{CTR}=0.169806.
\]

So click:

\[
5,492,054.
\]

So non-click:

\[
32,343,175 - 5,492,054 = 26,851,121.
\]

Ty le non-click/click:

\[
\frac{26,851,121}{5,492,054}
\approx 4.89.
\]

Day la muc imbalance quen thuoc trong CTR prediction. Model phai hoc xac suat click nho, nen logits va calibration rat quan trong. Trong code train, viec dung `BCEWithLogitsLoss` la phu hop vi loss nhan logit truc tiep va on dinh so hoc hon so voi viec sigmoid truoc roi moi tinh BCE.

## 6. Temporal CTR analysis

Feature `hour` trong Avazu ma hoa ca ngay va gio. Tu do script tach them:

- `day`
- `hour_of_day`
- `weekday`

### 6.1. CTR theo ngay

| Day | Count | CTR |
|---:|---:|---:|
| 21 | 3,298,847 | 0.1743 |
| 22 | 4,269,564 | 0.1570 |
| 23 | 3,097,245 | 0.1821 |
| 24 | 2,668,034 | 0.1749 |
| 25 | 2,689,983 | 0.1824 |
| 26 | 3,067,173 | 0.1830 |
| 27 | 2,579,915 | 0.1816 |
| 28 | 4,230,033 | 0.1522 |
| 29 | 3,067,157 | 0.1565 |
| 30 | 3,375,224 | 0.1693 |

CTR thay doi ro theo ngay. Cac ngay 23, 25, 26, 27 co CTR cao quanh 18.16-18.30%, trong khi ngay 22, 28, 29 thap hon, chi khoang 15.2-15.7%.

Dieu nay cho thay co temporal drift trong du lieu. Khi validation, nen uu tien split theo thoi gian thay vi random split, vi random split co the lam ro ri pattern thoi gian va danh gia lac quan hon thuc te.

### 6.2. CTR theo gio trong ngay

Gio co CTR cao nhat:

| Hour of day | Count | CTR |
|---:|---:|---:|
| 1 | 787,410 | 0.1856 |
| 15 | 1,665,236 | 0.1814 |
| 0 | 676,031 | 0.1802 |
| 16 | 1,640,126 | 0.1794 |
| 7 | 1,486,090 | 0.1792 |

Gio co CTR thap nhat:

| Hour of day | Count | CTR |
|---:|---:|---:|
| 4 | 1,530,066 | 0.1596 |
| 9 | 1,820,988 | 0.1601 |
| 20 | 895,656 | 0.1603 |
| 21 | 794,496 | 0.1606 |
| 10 | 1,720,384 | 0.1613 |

CTR co tinh chu ky theo gio. Vi vay, `hour_of_day` nen duoc giu nhu mot categorical/time feature rieng. Voi model embedding, `hour_of_day` co the duoc hash/embedding nhu categorical feature. Neu lam feature engineering bo sung, co the can nhac them cyclical encoding, nhung voi he thong hien tai, categorical hashing la du va Kaggle-safe.

## 7. Feature-level observations

### 7.1. Site features

Mot so site co CTR rat khac nhau:

| Feature | Value | CTR |
|---|---|---:|
| `site_id` | `85f751fd` | 0.1188 |
| `site_id` | `1fbe01fe` | 0.2057 |
| `site_id` | `e151e245` | 0.2963 |
| `site_id` | `5b08c53b` | 0.4671 |
| `site_domain` | `c4e18dd6` | 0.1227 |
| `site_domain` | `7687a86e` | 0.4597 |
| `site_category` | `50e219e0` | 0.1286 |
| `site_category` | `3e814130` | 0.2828 |

Dieu nay cho thay inventory phia site la nguon tin hieu manh. Khong chi category, ma ca ID/domain cu the deu co lien he voi xac suat click.

### 7.2. App features

Mot so app value cung co CTR rat khac nhau:

| Feature | Value | CTR |
|---|---|---:|
| `app_id` | `ecad2386` | 0.1986 |
| `app_id` | `92f5800b` | 0.0194 |
| `app_id` | `9c13b419` | 0.3051 |
| `app_domain` | `7801e8d9` | 0.1949 |
| `app_domain` | `ae637522` | 0.0239 |
| `app_category` | `07d7df22` | 0.1992 |
| `app_category` | `0f2161f8` | 0.1081 |
| `app_category` | `f95efa07` | 0.2476 |

App-side features co ca gia tri rat pho bien va gia tri co CTR rat thap/cao. Day la ly do cac mo hinh embedding nhu DeepFM, AutoInt, FIN, KANFI phu hop hon logistic regression don gian voi feature thu cong.

### 7.3. Device features

`device_ip` va `device_id` co cardinality rat cao. Dac biet, `device_id = a99f214a` xuat hien rat nhieu va co CTR gan muc trung binh:

| Feature | Value | CTR |
|---|---|---:|
| `device_id` | `a99f214a` | 0.1742 |
| `device_id` | `0f7c61dc` | 0.7573 |
| `device_id` | `c357dbff` | 0.6349 |
| `device_model` | `8a4875bd` | 0.1382 |
| `device_model` | `1f0bc64f` | 0.2246 |
| `device_conn_type` | `0` | 0.1811 |
| `device_conn_type` | `3` | 0.0441 |

Voi device-level features, can than voi rare categories. Mot vai `device_id` co CTR rat cao nhung count co the nho, nen khong nen dien giai nhu causal signal. Hashing va regularization giup model khong overfit qua manh vao nhung ID hiem.

### 7.4. Banner and anonymous categorical features

Mot so feature an danh va kich thuoc banner co CTR chenh lech manh:

| Feature | Value | CTR |
|---|---|---:|
| `banner_pos` | `0` | 0.1643 |
| `banner_pos` | `1` | 0.1837 |
| `banner_pos` | `7` | 0.3216 |
| `C15` | `320` | 0.1586 |
| `C15` | `300` | 0.3592 |
| `C16` | `50` | 0.1583 |
| `C16` | `250` | 0.4213 |
| `C14` | `4687` | 0.2493 |
| `C14` | `21189` | 0.0200 |
| `C14` | `21191` | 0.0197 |
| `C21` | `23` | 0.2129 |
| `C21` | `71` | 0.0283 |

`C15` va `C16` thuong duoc xem nhu cac bien lien quan den kich thuoc ad slot. Su khac biet CTR giua cac kich thuoc cho thay ad layout/format co tac dong lon den kha nang click.

## 8. Modeling implications

### 8.1. Hashing tricks la can thiet

Do `device_ip` va `device_id` co hang trieu gia tri, one-hot encoding truc tiep se khong an toan ve bo nho. Hashing trick giup anh xa category ve khong gian huu han, sau do dung embedding lookup de hoc representation.

Can dung hashing on dinh, khong dung Python `hash()`, vi Python hash co random seed theo process. Trong project, nen tiep tuc dung stable hashing tu `src/features/hashing.py`.

### 8.2. Categorical embeddings phu hop voi Avazu

Avazu khong phai bai toan co nhieu bien numeric lien tuc. Phan lon feature la categorical ID. Vi vay, pipeline:

\[
\text{raw category}
\rightarrow
\text{hash bucket}
\rightarrow
\text{embedding}
\rightarrow
\text{CTR model}
\]

la hop ly.

### 8.3. FIN branch co co so tu EDA

Nhieu tin hieu CTR nam o tuong tac giua cac nhom feature:

- site va app inventory;
- device va app;
- banner position va kich thuoc ad;
- time va inventory.

Do do, FIN branch voi self-attention co y nghia vi no hoc feature interactions thay vi chi cong doc lap tung feature.

Attention heatmap cua FIN co the duoc dung de quan sat feature nao attend den feature nao. Tuy nhien, attention la interaction pattern, khong phai causal importance.

### 8.4. KAN branch co the dien giai theo feature contribution

Trong KANFI hien tai, KAN branch tinh:

\[
\operatorname{KAN}(\mathbf{x})
=
b_{\mathrm{KAN}}
+
\sum_i c_i(\mathbf{x}).
\]

Voi:

\[
c_i(\mathbf{x})
=
a_i
\sum_j
\rho_j\phi_i(v_{i,j})
+
\beta_i.
\]

Do do, sau khi train co the xem:

- `feature_contributions`: dong gop \(c_i(\mathbf{x})\) cua tung feature vao KAN logit.
- `mean(|c_i(\mathbf{x})|)`: feature importance theo KAN branch.
- `mean(c_i(\mathbf{x}))`: xu huong signed contribution cua feature.
- plot \(\phi_i(t)\): response curve cua scalar embedding trong feature \(i\).

Can nho rang \(\phi_i(t)\) duoc hoc tren embedding scalar, khong phai raw category value. Vi vay, interpretation cua KAN branch la interpretation trong learned embedding space.

### 8.5. Validation nen theo thoi gian

Do CTR thay doi theo ngay va gio, temporal split quan trong. Neu train/valid split random, model co the thay pattern cua cung ngay/gio trong ca train va valid, lam metric lac quan. Mot split theo ngay se gan voi thuc te hon, vi CTR system thuong duoc deploy tren du lieu tuong lai.

## 9. Main conclusions

1. Dataset co 32.34M impressions duoc scan va CTR trung binh 16.98%, the hien bai toan binary classification bi lech nhan.

2. Khong co missing value theo nghia empty/NA, nhung co cac category dac biet nhu `device_id=a99f214a` va `C20=-1`, nen can giu chung nhu categorical values.

3. Cardinality rat cao o `device_ip`, `device_id`, `site_id`, `site_domain`, `app_id`, `device_model`, nen hashing trick va embedding la bat buoc de dam bao memory safety.

4. CTR thay doi manh theo ngay va gio, cho thay temporal drift. Validation split theo thoi gian la can thiet.

5. Nhieu feature co CTR chenh lech lon giua cac raw value, dac biet la site/app/device, `banner_pos`, `C15`, `C16`, `C14`, `C21`. Dieu nay ung ho viec dung mo hinh embedding va interaction-aware models.

6. Voi KANFI, FIN branch phu hop de hoc feature interactions, trong khi KAN branch cho phep doc contribution theo feature thong qua \(c_i(\mathbf{x})\).

7. Can chay lai EDA sau khi script da duoc fix double-count truoc khi dua cac bang count/frequency vao bao cao chinh thuc. Cac ket luan ve CTR, missing rate, temporal trend va cardinality nhin chung van co gia tri.
