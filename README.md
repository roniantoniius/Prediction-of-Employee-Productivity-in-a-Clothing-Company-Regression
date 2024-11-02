![image](https://github.com/user-attachments/assets/91033874-7525-4a77-9f52-83461137d015)# Laporan Proyek Machine Learning - Nama Anda

## Domain Proyek
Pada industri garment atau clothing, efisiensi dan produktivitas karyawan menjadi aspek penting dalam menjaga kualitas dan memenuhi permintaan pasar yang fluktuatif. Sebagian besar aktivitas di perusahaan garment masih dilakukan secara manual, membuat pemantauan kinerja dan pemberian insentif menjadi proses yang cukup rumit. Pendekatan tradisional ini dapat menyulitkan manajemen dalam mendeteksi penurunan kinerja secara dini serta mengidentifikasi tim-tim dengan performa baik yang layak mendapat insentif. Di sinilah model prediksi produktivitas dapat berperan untuk mempermudah pengambilan keputusan.

Penerapan machine learning dalam memprediksi produktivitas aktual karyawan memungkinkan perusahaan untuk meningkatkan efisiensi operasional, mempercepat proses pengambilan keputusan, dan memberikan intervensi atau reward yang tepat waktu. Riset menunjukkan bahwa model prediksi yang akurat dapat membantu perusahaan garment mencapai target produktivitas serta mengurangi biaya yang muncul akibat keterlambatan atau penurunan kinerja (Sabuj et al, 2022). Model ini dapat membantu perusahaan untuk tetap kompetitif di tengah tekanan industri yang membutuhkan respon cepat dan kinerja yang konsisten.

Sabuj, H., Nuha, N., Gomes, P., Lameesa, A., & Alam, M. (2022). Interpretable Garment Workers’ Productivity Prediction in Bangladesh Using Machine Learning Algorithms and Explainable AI. 2022 25th International Conference on Computer and Information Technology (ICCIT), 236-241. https://doi.org/10.1109/ICCIT57492.2022.10054863.

## Business Understanding

Masalah utama yang dihadapi perusahaan garmen terkait produktivitas karyawan dan solusi yang ingin dicapai adalah:

### Problem Statements
- Bagaimana cara memprediksi produktivitas karyawan secara efektif agar perusahaan dapat melakukan intervensi atau pemberian insentif dengan tepat waktu?
- Bagaimana perusahaan dapat mengidentifikasi faktor-faktor yang berdampak pada produktivitas karyawan untuk mengoptimalkan alokasi tugas?

### Goals
- Mengembangkan model prediksi produktivitas karyawan yang dapat memperkirakan produktivitas secara akurat berdasarkan data historis, sehingga intervensi dan pemberian insentif dapat dilakukan secara efektif.
- Mengidentifikasi faktor-faktor yang berpengaruh terhadap produktivitas, seperti target harian, idle time, lembur, dan insentif, untuk membantu pengelolaan sumber daya manusia yang lebih efisien.


### Solution Statements
- Model prediksi akan menggunakan algoritma Regresi seperti Random Forest dan Cat Boost, dengan pemilihan model terbaik berdasarkan performa metrik evaluasi (MAE, MSE, RMSE, dan SMAPE).
- Melakukan pengoptimalan model dengan analisis faktor eksternal dan membandingkan hasil produktivitas antara tim yang memiliki target tinggi dengan tim yang cenderung sering mengalami perubahan desain. Model yang paling sesuai akan dievaluasi untuk memastikan ketepatan prediksi dalam skenario yang berbeda.

------------------------------------------------------------------------------------
## Data Understanding
Dataset yang digunakan dalam proyek ini berisi data produktivitas karyawan di sebuah perusahaan garment. Dataset ini diambil dari sumber terpercaya, yang dapat diakses melalui tautan berikut: [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/597/productivity+prediction+of+garment+employees). Dataset ini mencakup total 1197 entri dan 15 variabel yang memuat berbagai informasi mengenai kinerja dan kondisi operasional dari karyawan garment.

Berikut ini adalah daftar variabel dalam dataset beserta deskripsinya:

| Nama Variabel           | Deskripsi                                                                                     |
|-------------------------|-----------------------------------------------------------------------------------------------|
| `date`                  | Tanggal data diambil                                                                          |
| `quarter`               | Kuartal bisnis ketika data diambil (1–5)                                                      |
| `department`            | Departemen tempat karyawan bekerja (e.g., `sweing`, `finishing`)                             |
| `day`                   | Hari dalam minggu data dikumpulkan (e.g., `Saturday`, `Sunday`)                              |
| `team`                  | ID unik untuk setiap tim karyawan                                                             |
| `targeted_productivity` | Target produktivitas yang ditetapkan untuk setiap tim per hari                                |
| `smv`                   | Waktu standar (standard minute value) yang dialokasikan untuk tugas-tugas                     |
| `wip`                   | Work-in-progress atau pekerjaan yang sedang berjalan                                         |
| `over_time`             | Total waktu lembur yang dilakukan (dalam menit)                                               |
| `incentive`             | Insentif tambahan dalam bentuk bonus                                                          |
| `idle_time`             | Waktu idle atau waktu tidak produktif                                                         |
| `idle_men`              | Jumlah pekerja yang idle (tidak aktif bekerja)                                                |
| `no_of_style_change`    | Jumlah perubahan gaya atau desain pada hari tertentu                                          |
| `no_of_workers`         | Total jumlah karyawan dalam tim                                                               |
| `actual_productivity`   | Produktivitas aktual atau produktivitas yang berhasil dicapai (variabel target)               |

Dataset ini mengandung missing values pada kolom `wip` sebanyak 506 entri atau sekitar 50% dari total data. Karena proporsi missing value yang tinggi, kolom ini tidak dapat dihapus atau diisi dengan nilai nol; oleh karena itu, nilai yang hilang akan diisi menggunakan mean (rata-rata) sebagai pendekatan imputasi.


### Exploratory Data Analysis (EDA)
Langkah pertama dalam analisis data adalah menangani beberapa variabel yang menunjukkan outlier, yaitu `target_productivity`, `incentive`, `wip`, dan `over_time`. Teknik yang digunakan adalah Interquartile Range (IQR) untuk mendeteksi dan menangani outlier agar tidak mempengaruhi performa model.
![image](https://github.com/user-attachments/assets/475e860c-517a-4a7b-9233-ed5be3532ea5)

![image](https://github.com/user-attachments/assets/60df1993-362a-4ed8-ba36-efeb231ace1a)

#### Distribusi Data Variabel Kontinu
Beberapa hasil observasi dari EDA yang dilakukan pada variabel-variabel kontinu adalah sebagai berikut:
- **`targeted_productivity`**: Memiliki distribusi yang padat di sekitar 0.80, menunjukkan banyak karyawan mencapai target produktivitas pada nilai ini.
- **`smv`**: Waktu standar kerja memperlihatkan distribusi padat pada rentang 0–10, menunjukkan bahwa variasi produktivitas aktual lebih sering terjadi di rentang waktu standar yang rendah.
- **`wip`**: Distribusi `wip` menunjukkan persebaran seragam baik pada nilai rendah maupun tinggi. Kenaikan `wip` terkadang berkorelasi positif dengan produktivitas, meskipun tidak konsisten.
- **`over_time`**: Menunjukkan bahwa sedikit waktu lembur berkorelasi dengan peningkatan produktivitas.
- **`incentive`**: Insentif yang tinggi cenderung meningkatkan produktivitas, meskipun pada rentang insentif 0 terlihat variasi produktivitas yang padat.
- **`idle_men`**: Produktivitas aktual lebih rendah ketika jumlah karyawan idle meningkat.
- **`no_of_workers`**: Variasi produktivitas maksimal terlihat pada berbagai rentang jumlah pekerja, dengan distribusi bimodal.
- **`idle_time`**: Distribusi produktivitas aktual lebih padat ketika `idle_time` bernilai rendah.
- **`no_of_style_change`**: Variabel ini mungkin bersifat kategorikal karena tampak tidak menunjukkan tren tertentu terhadap produktivitas aktual.

![image](https://github.com/user-attachments/assets/28d41ab7-664e-4751-ac2c-d61f3a9f1bb5)


#### Analisis Variabel Kategorikal
- **`Quarter`**: Produktivitas karyawan terbaik terjadi pada Quarter 5, menunjukkan adanya pengaruh musiman atau strategi manajemen khusus.
- **`Department`**: Departemen `finishing` memiliki produktivitas sedikit lebih tinggi dibandingkan departemen `sweing`.
- **`day`**: Produktivitas relatif stabil dan seragam pada semua hari kerja.

![image](https://github.com/user-attachments/assets/e42c9909-21be-47a5-bc75-c9cc34f6c13d)

#### Distribusi Persebaran Data
Distribusi beberapa variabel menunjukkan bentuk yang unik, seperti:
- `smv` dan `incentive` memiliki sedikit skew ke kanan.
- `wip` menunjukkan distribusi dengan kurtosis.
- `over_time` dan `no_of_workers` cenderung menunjukkan distribusi normal atau bimodal.

![image](https://github.com/user-attachments/assets/807e3f79-e153-48b8-9b6b-0898a9eb11f9)


#### Fitur yang Dipilih untuk Model
Berdasarkan EDA yang dilakukan, beberapa variabel yang akan dimasukkan ke dalam model meliputi:
- `targeted_productivity`
- `smv`
- `wip`
- `over_time`
- `incentive`
- `no_of_workers`
- `no_of_style_change`

Dengan memahami persebaran data, missing values, dan potensi outlier, kita dapat memastikan model yang dibangun akan lebih robust dan akurat dalam memprediksi produktivitas karyawan pada industri garment.


## Data Preparation
### Feature Engineering
1. **Encoding Variabel Kategorikal**  
Variabel-variabel kategorikal seperti `day` dan `department` dikonversi ke bentuk numerik dengan metode **One Hot Encoding**. Hal ini dilakukan agar algoritma regresi dapat memanfaatkan informasi tersebut tanpa mengasumsikan adanya hubungan urutan di antara kategori. Sementara itu, variabel `quarter` di-encode menggunakan **OrdinalEncoder** karena memiliki urutan tertentu yang bisa relevan dalam analisis.

2. **Scaling**  
Distribusi variabel numerik diperiksa terlebih dahulu untuk menentukan metode scaling yang paling sesuai, yang diperlukan untuk menjaga stabilitas dan performa model regresi:
   - **Min-Max Scaling** diterapkan pada variabel `smv`, `over_time`, dan `no_of_workers` karena distribusi mereka cukup mendekati normal atau bimodal, yang cocok untuk pendekatan ini.
   - **Robust Scaling** digunakan untuk `wip`, yang memiliki outliers dan kurtosis tinggi sehingga skala robust dapat menjaga informasi inti dari data tanpa terlalu terpengaruh oleh outliers.
   - **Log Transformation** diterapkan pada `incentive` dan `idle_time` yang mengalami skewness, membantu mengurangi skewness dan menghasilkan distribusi yang lebih mendekati normal.
  
![image](https://github.com/user-attachments/assets/1218059c-c31e-415a-874e-66bcad5a8c27)


### Feature Selection
Proses pemilihan fitur dilakukan untuk meningkatkan performa model dengan mengidentifikasi fitur-fitur yang paling relevan terhadap prediksi produktivitas karyawan (`actual_productivity`). Berdasarkan analisis korelasi dengan heatmap, ditemukan bahwa variabel `targeted_productivity` dan `incentive` memiliki korelasi positif terhadap target. Untuk mencari fitur terbaik secara lebih akurat, **Recursive Feature Elimination (RFE)** diterapkan.
![image](https://github.com/user-attachments/assets/4e6ea115-6de6-43dd-9dc0-9c44595a968a)


- **RFE** berfungsi mengidentifikasi dan mempertahankan fitur-fitur yang paling berkontribusi melalui proses iteratif yang terus-menerus mengeliminasi fitur dengan kontribusi terendah. Kami menggunakan RFE dengan model regresi linear sebagai estimator untuk mendapatkan 10 fitur dengan peringkat tertinggi. Fitur-fitur terpilih ini kemudian akan digunakan pada model utama, dengan tujuan meningkatkan akurasi prediksi dan efisiensi proses.

![image](https://github.com/user-attachments/assets/8b8b639e-8f20-4845-9cbb-793eb2103038)


### Handling Imbalance
Distribusi variabel target `actual_productivity` menunjukkan ketidakseimbangan, di mana 86% data bernilai di atas 0.5. Untuk menangani hal ini, **Synthetic Minority Over-sampling Technique for Regression (SMOTER)** diterapkan, sebuah teknik oversampling yang bertujuan meningkatkan representasi dari nilai-nilai yang kurang umum dalam variabel target. SMOTER memperkirakan sampel sintetis pada kelas-kelas minoritas untuk mendapatkan data yang lebih seimbang, memungkinkan model menangkap variasi yang lebih luas dalam data produktivitas aktual.
![image](https://github.com/user-attachments/assets/b826a34e-9201-49af-819e-542c1829e308)

Handling Imbalance Case Result:
![image](https://github.com/user-attachments/assets/fceaeda2-ce67-4774-81d8-dc8d320fdd63)

## Modeling
Pertama-tama data dipecah menjadi data latih (X_train, y_train) dan data uji (X_test, y_test) dengan ukuran (757, 13) untuk train set dan (253, 13) untuk test set.

Berikut adalah penjelasan dari setiap algoritma yang digunakan, termasuk parameter yang digunakan, kelebihan, dan kekurangan masing-masing model:

### Algoritma dan Evaluasi Model
1. **Linear Regression**
   - **Parameter**: Menggunakan default parameter dari `LinearRegression()`.
   - **Kelebihan**: Algoritma sederhana dan cepat dalam komputasi, cocok untuk data dengan hubungan linear.
   - **Kekurangan**: Pada proyek ini, performa model stabil namun tidak memberikan hasil yang terlalu tinggi dalam hal akurasi. Model tidak terlalu bagus untuk menangkap hubungan kompleks antar fitur.
   
2. **K-Nearest Neighbors (KNN) Regression**
   - **Parameter**: `n_neighbors=30`, `algorithm="auto"`
   - **Kelebihan**: Algoritma ini fleksibel untuk data non-linear dan relatif mudah diimplementasikan.
   - **Kekurangan**: Meskipun performa cukup stabil, hasil akurasinya kurang tinggi pada dataset ini dan algoritma ini sensitif terhadap skala data.

3. **Decision Tree Regression**
   - **Parameter**: `max_depth=10`, `criterion="squared_error"`
   - **Kelebihan**: Baik dalam mempelajari pola data dan cenderung memberikan hasil akurasi yang baik pada train set.
   - **Kekurangan**: Model ini cenderung overfitting, karena performa menurun ketika diterapkan pada data test. Model kurang generalisasi pada data yang tidak terlihat saat pelatihan.

4. **Random Forest Regression**
   - **Parameter**: `max_depth=15`, `criterion="squared_error"`, `n_estimators=38`
   - **Kelebihan**: Model ini memiliki performa yang sangat baik pada train set dan umumnya lebih stabil daripada decision tree, dengan kemampuan mengurangi overfitting melalui ensemble learning.
   - **Kekurangan**: Meski model ini menunjukkan performa sangat baik pada data latih, performa menurun pada data uji, yang menunjukkan adanya potensi overfitting. Model juga lebih memakan waktu komputasi dibandingkan model lain.

5. **Support Vector Machine (SVM) RBF Regression**
   - **Parameter**: `kernel='rbf'`
   - **Kelebihan**: SVM dengan kernel RBF dapat memodelkan data non-linear dengan baik.
   - **Kekurangan**: Performa model stabil, namun hasil prediksi kurang akurat untuk proyek ini, sehingga tidak menjadi pilihan utama.

6. **Multilayer Perceptron (MLP) Regression**
   - **Parameter**: `hidden_layer_sizes=(100,)`, `activation='relu'`, `solver='adam'`, `max_iter=500`
   - **Kelebihan**: Kemampuan menangani data non-linear dan dapat diadaptasi dengan konfigurasi lebih dalam.
   - **Kekurangan**: Meskipun model menunjukkan performa stabil pada data latih, model ini masih mengalami sedikit overfitting dan memerlukan tuning lebih lanjut.

7. **XGBoost Regression**
   - **Parameter**: Menggunakan default dari `xgb.XGBRegressor()`.
   - **Kelebihan**: Algoritma ini cepat, efisien, dan memiliki performa tinggi untuk menangkap pola data. Model ini memiliki performa yang sangat baik pada data latih.
   - **Kekurangan**: Performa menurun sedikit pada data test, namun masih tergolong baik dan cocok untuk ditingkatkan dengan hyperparameter tuning.

8. **LightGBM Regression**
   - **Parameter**: Menggunakan default dari `lgb.LGBMRegressor()`.
   - **Kelebihan**: Cepat dan efisien untuk data besar, menghasilkan prediksi akurat pada data latih.
   - **Kekurangan**: Performanya sedikit menurun pada data uji, namun tidak overfitting berlebihan dan masih bisa dioptimalkan lebih lanjut.

9. **CatBoost Regression**
   - **Parameter**: Menggunakan default dari `CatBoostRegressor(verbose=0)`.
   - **Kelebihan**: Mudah menangani data dengan kategori dan memberikan hasil yang konsisten dengan performa yang baik.
   - **Kekurangan**: Model ini memiliki performa yang sedikit lebih baik dibandingkan LightGBM, namun performa pada data uji masih dapat ditingkatkan melalui tuning parameter.

### Memilih Model Terbaik
Berdasarkan evaluasi nilai MAE, MSE, RMSE, dan sMAPE, kami mengidentifikasi tiga model dengan performa terbaik untuk prediksi produktivitas karyawan, yaitu:
1. **Random Forest Regression**
2. **CatBoost Regression**
3. **XGBoost Regression**

### Hyperparameter Tuning terhadap Top 3 Model Terbaik
Proses tuning dilakukan pada ketiga model terbaik dengan menggunakan `RandomizedSearchCV` untuk mengoptimalkan performa. Berikut hasil tuning:

1. **CatBoost Regression**:
   - Best Parameter: `{'depth': 4, 'iterations': 190, 'l2_leaf_reg': 5, 'learning_rate': 0.199}`
   
2. **Random Forest Regression**:
   - Best Parameter: `{'max_depth': 10, 'max_features': 'sqrt', 'min_samples_leaf': 1, 'min_samples_split': 5, 'n_estimators': 164}`
   
3. **XGBoost Regression**:
   - Best Parameter: `{'colsample_bytree': 0.4, 'gamma': 0, 'learning_rate': 0.06, 'max_depth': 4, 'min_child_weight': 5, 'n_estimators': 163}`

### Model Terbaik
Model **Cat Boost** setelah diterapkan Hyperparameter Tuning menunjukkan performa terbaik dalam memprediksi produktivitas karyawan dengan nilai evaluasi sebagai berikut:
- MAE: 0.0944
- MSE: 0.0197
- RMSE: 0.1402
- sMAPE: 17.2017%

Model ini dipilih sebagai model final karena hasilnya yang stabil pada train dan test set, serta menunjukkan performa **good fit** tanpa tanda overfitting. Model ini diharapkan memberikan prediksi produktivitas yang akurat untuk mendukung pengambilan keputusan di industri garment.

![image](https://github.com/user-attachments/assets/565fac4c-876c-45f1-a308-7032cfbe98c9)

![image](https://github.com/user-attachments/assets/dc82b457-a27e-4c53-913f-814f52323fcb)


## Evaluation
### Penjelasan Metrik Evaluasi
![image](https://github.com/user-attachments/assets/d1636897-5ddb-4255-bce3-f2277883627a)

- **MAE (Mean Absolute Error)**: MAE mengukur rata-rata dari selisih absolut antara nilai prediksi dan nilai aktual, menunjukkan seberapa besar rata-rata kesalahan prediksi yang dibuat oleh model. Formula MAE adalah:

  \[
  \text{MAE} = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y_i}|
  \]

- **MSE (Mean Squared Error)**: MSE menghitung rata-rata dari kuadrat kesalahan, memberikan penekanan lebih pada kesalahan yang lebih besar. Formula MSE adalah:

  \[
  \text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y_i})^2
  \]

- **RMSE (Root Mean Squared Error)**: RMSE merupakan akar dari MSE, yang menjaga unit kesalahan konsisten dengan data. RMSE memberi bobot lebih pada kesalahan besar, sehingga dapat mengidentifikasi ketidakakuratan prediksi yang lebih signifikan. Formula RMSE adalah:

  \[
  \text{RMSE} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y_i})^2}
  \]

- **SMAPE (Symmetric Mean Absolute Percentage Error)**: SMAPE mengukur persentase kesalahan yang dinormalisasi dengan mempertimbangkan kedua nilai aktual dan prediksi. Ini adalah metrik yang cocok untuk melihat performa model dari segi kesalahan relatif. Formula SMAPE adalah:

  \[
  \text{SMAPE} = \frac{100\%}{n} \sum_{i=1}^{n} \frac{|y_i - \hat{y_i}|}{(|y_i| + |\hat{y_i}|) / 2}
  \]

  Interpretasi dari SMAPE:
  - **SMAPE = 0%**: Model prediksi sempurna tanpa kesalahan.
  - **SMAPE > 0%**: Menunjukkan adanya kesalahan dalam prediksi, dengan nilai yang lebih rendah menunjukkan performa yang lebih baik.
![image](https://github.com/user-attachments/assets/34cbc1ea-5744-4fb8-8cd4-baac7a0ccac0)

### Hasil Proyek Berdasarkan Metrik Evaluasi

Dari beberapa algoritma yang diujikan, yaitu **Random Forest Regressor**, **Cat Boost**, dan **XG Boost**, berikut adalah hasil evaluasi berdasarkan data uji setelah proses hyperparameter tuning:

1. **Cat Boost Regressor**
   - MAE: 0.0944
   - MSE: 0.0197
   - RMSE: 0.1402
   - SMAPE: 17.2017%

Model Cat Boost menunjukkan performa terbaik dengan nilai MAE, MSE, dan RMSE yang rendah serta SMAPE di bawah 20%. Nilai SMAPE ini menunjukkan bahwa kesalahan prediksi masih dalam batas yang cukup baik, dengan tingkat kesalahan relatif rendah.

2. **Random Forest Regressor**
   - MAE: 0.0977
   - MSE: 0.0206
   - RMSE: 0.1434
   - SMAPE: 17.6437%

Random Forest memiliki performa yang baik, meski sedikit tertinggal dari Cat Boost dalam hal ketepatan prediksi. SMAPE menunjukkan model ini masih dalam rentang kesalahan yang cukup baik, namun tidak sebaik Cat Boost.

3. **XG Boost Regressor**
   - MAE: 0.1168
   - MSE: 0.0284
   - RMSE: 0.1685
   - SMAPE: 22.2079%

XG Boost menunjukkan performa yang sedikit lebih rendah dibandingkan dengan Cat Boost dan Random Forest. Nilai SMAPE yang lebih tinggi menunjukkan bahwa tingkat kesalahan relatif dari prediksi lebih besar, dan model ini mungkin kurang akurat dalam beberapa kasus.

### Kesimpulan Evaluasi
Berdasarkan nilai evaluasi dari keempat metrik, **Cat Boost Regressor** dipilih sebagai model terbaik untuk memprediksi produktivitas karyawan. Model Cat Boost yang telah di-tuning dapat memprediksi produktivitas karyawan dengan tingkat kesalahan yang cukup rendah, membuatnya layak untuk digunakan sebagai dasar dalam pengambilan keputusan terkait produktivitas di industri garmen atau perusahaan clothing.

### Rekomendasi Hasil Analisis:

1. Tim karyawan yang memiliki target harian yang tinggi cenderung mencapai produktivitas yang lebih tinggi. Tim dengan target harian yang ditetapkan 10% lebih tinggi dari rata-rata (70%), berhasil mencapai target produktivitas yang bagus. Selain itu, jumlah revisi desain yang lebih rendah berkorelasi positif dengan produktivitas yang lebih tinggi. Data menunjukkan bahwa tim dengan perubahan desain minimal hanya memiliki produktivitas 10% lebih tinggi dibandingkan tim yang sering melakukan revisi.
![image](https://github.com/user-attachments/assets/06b3839b-6240-40e6-b3c1-a1164d2e0a70)
![image](https://github.com/user-attachments/assets/2d96dbf4-2b3d-4aa5-ab86-3a26000afeda)


2. Data menunjukkan bahwa produktivitas menurun dengan meningkatnya lembur. Tim yang bekerja lebih dari 0.6 mengalami penurunan produktivitas sebesar 20% dan akan terus menurun. Ini menunjukkan bahwa kelelahan dan kelebihan beban kerja mengurangi efisiensi buruh. Selain itu, tim yang menghabiskan waktu lebih dari 0.4 per hari pada satu tugas mengalami penurunan produktivitas sebesar 20%. Namun, Jumlah karyawan dalam tim yang optimal juga memainkan peran penting. Tim dengan jumlah pekerja yang sedang pekerja menunjukkan produktivitas yang stabil.
![image](https://github.com/user-attachments/assets/0e047df4-f14a-4ab7-8a3e-9abf9f4e2671)
![image](https://github.com/user-attachments/assets/b923c46e-0fcb-4211-90dd-3599e177e904)
![image](https://github.com/user-attachments/assets/821ba6cc-2d46-4ac8-bb9f-2e61b143fbd2)

3. Insentif yang tinggi terbukti efektif dalam meningkatkan performa produktivitas tim. Tim yang menerima bonus kinerja menunjukkan peningkatan produktivitas yang signifikan. Selain itu, menjaga idle time tetap rendah juga krusial. Data menunjukkan bahwa tim karyawan dengan tanpa idle time memiliki produktivitas yang 26% lebih tinggi dibandingkan tim dengan idle time. Dengan mengurangi waktu tidak produktif, karyawan dapat fokus lebih baik pada tugas mereka.
![image](https://github.com/user-attachments/assets/de543e84-439f-4595-bf5e-160c05b0d524)
![image](https://github.com/user-attachments/assets/7ad6f753-986b-4125-88e8-194822632647)
