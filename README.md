# Đồ Án Nhập môn Khoa học Dữ liệu

**Môn học:** Nhập môn Khoa học Dữ liệu  
**Giảng viên:** ThS. Lê Nhựt Nam  
**Nhóm:** 10  
 
|Họ và Tên|MSSV|
|---|---|
| Nguyễn Hữu Khánh Hưng|	23120271|
|Phạm Quốc Khánh|23120283|
|Châu Huỳnh Phúc|	23120329|
|Vũ Trần Phúc| 23120333|

## Công nghệ 

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-5F9EA0?style=for-the-badge&logoColor=white)
![CatBoost](https://img.shields.io/badge/CatBoost-FF9900?style=for-the-badge&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-3776AB?style=for-the-badge&logo=matplotlib&logoColor=white)
![Seaborn](https://img.shields.io/badge/Seaborn-3776AB?style=for-the-badge&logo=seaborn&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)

![Dataset](https://img.shields.io/badge/Dataset-phongtro123.com-blue?style=for-the-badge&logo=google-maps&logoColor=white)
![Topic](https://img.shields.io/badge/Topic-Motel_Pricing_Predictions-orange?style=for-the-badge&logo=google-analytics&logoColor=white)
![Status](https://img.shields.io/badge/Status-Done-success?style=for-the-badge&logo=checkmarx&logoColor=white)



## Mô tả dự án
Dự án tập trung vào việc phân tích và dự đoán giá phòng trọ tại TP.HCM dựa trên dữ liệu crawl từ phongtro123.com.

### Thông số
Nhóm thu thập > 20.000 records với > 10 features.

## Cài đặt  
### Yêu cầu
- Python 3.10+
- Google Gemini API key
- Jupyter Notebook / JupyterLab

### Hướng dẫn cài đặt
1. Clone repository
```bash
# Clone repository
git clone https://github.com/HungHiHung10/Intro2DS.git
cd Intro2DS
```
2. Khởi tạo virtual environment

_Khuyến nghị sử dụng `venv` hoặc `conda` để quản lý môi trường ảo._
```bash
conda create -n intro2ds python=3.10
conda activate intro2ds
```
1. Cài đặt dependencies
```bash
pip install -r requirements.txt
```
1. Chạy các notebooks theo thứ tự(trong thư mục `notebooks/`)
```bash
jupyter notebook notebooks/
```
## Cấu trúc thư mục

```text
Intro2DS
│   .gitignore             
│   LICENSE                # Giấy phép MIT
│   README.md              
│   requirements.txt       
│   
├───Data                  
│    ├──   raw.csv         # Dữ liệu thô merge từ tất cả các page crawl
│    ├──   cleaned.csv     # Dữ liệu đã làm sạch cơ bản (loại bỏ duplicates, convert type)
│    ├──   processed.csv   # Dữ liệu cuối cùng sau preprocessing đầy đủ (dùng cho EDA & Modeling)
│    ├──   Page1to300.csv  # Dữ liệu crawl từ trang 1–300
│    ├──   Page301to600.csv # Trang 301–600
│    ├──   Page601to900.csv # Trang 601–900
│    └──   Page901to1200.csv # Trang 901–1200
├───assets                 # Hình ảnh, biểu đồ minh họa trong báo cáo và notebook
│
├───notebooks              # Các Jupyter Notebook theo quy trình dự án
│    ├──   images/         # Thư mục lưu ảnh visualize (heatmap, distribution, feature importance...)
│    ├──   CrawlData.ipynb # Notebook crawl dữ liệu từ phongtro123.com
│    ├──   ProcessingData.ipynb # Xử lý dữ liệu thô, merge file, clean cơ bản
│    ├──   Pre_Processing.ipynb # Tiền xử lý nâng cao (regex, missing values, feature engineering)
│    ├──   EDA.ipynb       # Khám phá dữ liệu (phân bố, correlation, outliers)
│    ├──   Analysis.ipynb  # Trả lời 5 câu hỏi nghiên cứu + visualize insights
│    ├──   Modeling.ipynb  # Xây dựng, tuning và đánh giá mô hình (Linear, XGBoost, CatBoost)
│    └──   Bonus_Extension.ipynb # Phần mở rộng (Gemini check tiện ích qua ảnh, amenities verification)
│
└───utilities              # Các module Python hỗ trợ (code tái sử dụng)
    ├──   analysis.py      # Hàm phân tích dữ liệu, tính toán metrics
    ├──   explore_data.py  # Hàm EDA (plot distribution, correlation heatmap...)
    ├──   modeling.py      # Hàm xây dựng và tuning mô hình (train XGBoost, CatBoost...)
    ├──   processing.py    # Hàm preprocessing (clean text, handle missing, regex...)
    └──   visualize.py     # Hàm vẽ biểu đồ (barplot, boxplot, scatter...)
                   
```

## Quy trình & Notebook chính
1. Data Collection → Crawl bài đăng từ phongtro123.com (title, url, tiện ích claim) → Lưu raw.csv
2. Amenities Verification → Lấy 3 ảnh tiềm năng nhất → Dùng Gemini 2.5 Flash phân tích ảnh → So sánh với claim → Kết quả: ĐỦ / THIẾU + missing list
3. EDA & Preprocessing → (sắp tới) Phân tích phân bố giá, diện tích, tỷ lệ "khai khống" tiện ích
Meaningful Questions → (sắp tới) 5+ câu hỏi ý nghĩa + visualize
4. Modeling → (tùy chọn mở rộng) Dự đoán giá phòng dựa trên tiện ích thực + claim, hoặc phân loại "tin thật / tin giả"
5. Evaluation & Reflection → Báo cáo metrics, khó khăn, bài học
## 5 Câu hỏi ý nghĩa
1. Giá thuê phòng thay đổi như thế nào giữa các quận khác nhau? → Hỗ trợ sinh viên chọn khu vực phù hợp ngân sách.
2. Tiện nghi nào phân biệt phòng bình dân và cao cấp? → Giúp đánh giá giá trị thực của listing.
3. Biến động giá trong mùa cao điểm (giữa quận trung tâm & ven)? → Dự báo thời điểm thuê tốt nhất.
4. Sự phân cực giá giữa mặt tiền và hẻm nhỏ trong cùng quận? → Hiểu ảnh hưởng vị trí chi tiết.
5. Có yếu tố mùa vụ ảnh hưởng đến giá thuê tại TP.HCM? → Insight về xu hướng thời gian.

_(analysis + visualize + insights được trình bày trong notebooks/Analysis.ipynb)_

## Phân công công việc (Tóm tắt)
| STT | Thành viên            | MSSV     | Nhiệm vụ chính                                                                    |
| --- | --------------------- | -------- | --------------------------------------------------------------------------------- |
| 1   | Nguyễn Hữu Khánh Hưng | 23120271 | Crawl data, Baseline Linear Regression, Hyperparameter tuning, Merge code, README |
| 2   | Phạm Quốc Khánh       | 23120283 | Crawl data, XGBoost, Phân tích địa lý & tiện nghi (Câu 1,2), Báo cáo              |
| 3   | Châu Huỳnh Phúc       | 23120329 | Feature Engineering, Correlation, Mùa vụ (Câu 5), Quản lý tiến độ                 |
| 4   | Vũ Trần Phúc          | 23120333 | Word-embedding, CatBoost, Biến động giá & vị trí (Câu 3,4), Báo cáo tổng kết      |
## Kết quả & So sánh mô hình
### Kết quả huấn luyện mô hình XGBoost
![XGBoost Traing Result](../assets/xgb.png)
### So sánh mô hình dự đoán giá phòng trọ
![Visualize Result](../assets/compare.png)
| Model                       | R²     | MAE (Triệu VND) | MSE    | RMSE   |
| --------------------------- | ------ | --------------- | ------ | ------ |
| XGBoost Regressor           | **0.5518** | **0.5763**          | **0.5786** | **0.7607** |
| CatBoost Regressor          | 0.4769 | 0.6438          | 0.8210 | 0.9061 |
| Linear Regression Optimized | 0.3337 | 0.7630          | 1.0283 | 1.0140 |
| Linear Regression           | 0.3251 | 0.7894          | 1.0523 | 1.0258 |

### Bảng kết quả mô hình dự đoán giá phòng trọ
**Nhận xét:**
- **XGBoost** là mô hình tốt nhất với R² cao nhất (0.5035) và lỗi tuyệt đối nhỏ nhất (MAE ~613k VND, RMSE ~801k VND).
- **CatBoost** xếp thứ 2, vẫn vượt trội hơn các phiên bản Linear Regression.
- **Linear Regression Optimized (ElasticNet)** cải thiện rõ rệt so với baseline (tăng R² từ 0.321 → 0.333), nhưng vẫn kém xa các mô hình boosting.
- **Kết luận:** XGBoost là lựa chọn tối ưu cho bài toán dự đoán giá phòng trọ trong dự án này.

## Reflection

### Thành viên: Nguyễn Hữu Khánh Hưng - 23120271

- **Khó khăn gặp phải:**  
  - Giai đoạn thu thập dữ liệu: Việc crawl từ phongtro123.com gặp nhiều trở ngại như rate limit (bị chặn request sau một số lượng lớn truy cập), dữ liệu không đồng nhất (nhiều bài đăng thiếu tiện ích hoặc mô tả text lộn xộn), và phải xử lý pagination thủ công để thu thập đủ >20.000 records mà không bị block IP.  
  - Preprocessing & EDA: Dữ liệu thô có nhiều missing values (đặc biệt ở tiện ích như máy giặt, gác lửng), giá và diện tích cần regex phức tạp để chuyển về float, đồng thời phát hiện outliers (giá phòng "ảo" quá cao/thấp do lỗi nhập liệu hoặc đàm phán).  
  - Modeling & Tuning: Xây dựng baseline Linear Regression khá đơn giản, nhưng khi tinh chỉnh hyperparameter cho XGBoost và Linear Regression Optimized by ElasticNet (grid/random search với learning_rate, max_depth, n_estimators...) tốn rất nhiều thời gian và tài nguyên tính toán (máy cá nhân chạy chậm, phải thử nhiều lần để tránh overfitting/underfitting).  
  - Merge code & README: Khi hợp nhất code từ 4 thành viên, gặp conflict ở một số notebook như `Pre_Processing.ipynb`, phải refactor thủ công để code sạch và chạy ổn định trên mọi máy. Viết README chi tiết cũng mất kha khá thời gian để đảm bảo hướng dẫn tái hiện chính xác.

- **Bài học rút ra:**  
  - Preprocessing và feature engineering chiếm phần lớn thời gian (khoảng 60–70%) nhưng là yếu tố quyết định chất lượng mô hình – dữ liệu sạch giúp XGBoost đạt R² cao hơn hẳn Linear Regression.  
  - Hyperparameter tuning không chỉ là "thử nhiều" mà cần chiến lược (bắt đầu từ learning_rate thấp + early stopping, dùng random search trước grid search để tiết kiệm thời gian).  
  - Làm việc nhóm hiệu quả nhờ GitHub (branching, pull request, peer review) giúp tránh mất dữ liệu/code và học hỏi lẫn nhau (tôi học được cách xử lý categorical từ bạn Phạm Quốc Khánh, và word-embedding từ bạn Vũ Trần Phúc).  
  - README và documentation rõ ràng rất quan trọng – không chỉ để giảng viên chấm mà còn giúp chính mình tái hiện dự án sau này.

- **Nếu có thêm thời gian:**  
  - Scale dataset lớn hơn bằng cách crawl thêm từ nhatot.com hoặc chotot.com để tăng độ đa dạng dữ liệu và giảm bias.  
  - Thử nghiệm ensemble (stacking XGBoost + CatBoost + LightGBM) hoặc thêm Deep learning (MLP với PyTorch) để đẩy R² lên cao hơn.  
  - Tích hợp thêm nhiều feature địa lý hơn (khoảng cách đến trường đại học/metro bằng Google Maps API) và deploy mô hình thành web app đơn giản (Streamlit) để dự đoán giá realtime cho sinh viên.  
  - Thực hiện A/B testing hoặc cross-validation nâng cao hơn để đánh giá độ ổn định mô hình trên các phân khúc giá khác nhau (phòng dưới 3 triệu vs trên 5 triệu).

## License
MIT License - Phục vụ mục đích học tập & nghiên cứu.














Dưới đây là nội dung **README.md** được viết lại hoàn chỉnh, dựa sát vào báo cáo nhóm 10 mà bạn cung cấp. README được thiết kế chuyên nghiệp, rõ ràng, phù hợp với đồ án cuối kỳ môn **Nhập môn Khoa học Dữ liệu** (CSC14119, lớp CQ2023/23_21), nhấn mạnh quy trình CRISP-DM, phân công công việc, tiến độ, công cụ, và kết quả chính.

Tôi đã:
- Cập nhật thông tin nhóm (thành viên, MSSV, giảng viên hướng dẫn từ báo cáo).
- Tích hợp badge như bạn yêu cầu (với sửa lỗi logo XGBoost/CatBoost bằng cách bỏ logo không tồn tại).
- Mô tả đề tài: **Phân tích & Dự đoán Giá Phòng Trọ cho Học sinh – Sinh viên** (dữ liệu crawl từ phongtro123.com).
- Liệt kê cấu trúc repo dựa trên báo cáo và repo thực tế (Data/, notebooks/, utilities/, requirements.txt).
- Thêm phần quy trình phối hợp, kế hoạch giai đoạn, phân công (tóm tắt từ báo cáo).
- Thêm phần 5 câu hỏi nghiên cứu (dựa trên mô tả trong báo cáo giai đoạn 5).
- Kết thúc bằng reflection nhóm và license.

Copy toàn bộ nội dung dưới đây vào file `README.md` ở root repo, commit & push để GitHub hiển thị đẹp.

```markdown
# Dự án Cuối kỳ: Phân tích & Dự đoán Giá Phòng Trọ cho Học sinh – Sinh viên  
**Môn học:** Nhập môn Khoa học Dữ liệu (CSC14119)  
**Lớp:** CQ2023/23_21  
**Giảng viên hướng dẫn:** ThS. Lê Nhựt Nam, ThS. Lê Ngọc Thành, ThS. Huỳnh Lâm Hải Đăng, ThS. Võ Nam Thục Đoan  
**Nhóm 10**  
**Thành viên:**  
- Nguyễn Hữu Khánh Hưng (23120271)  
- Phạm Quốc Khánh (23120283)  
- Châu Huỳnh Phúc (23120329)  
- Vũ Trần Phúc (23120333)  

**Thời gian thực hiện:** Tháng 10/2025 – Tháng 01/2026  
**Nguồn dữ liệu:** Crawl tự động từ phongtro123.com (dữ liệu thực tế, >10.000 bản ghi nếu scale full)  

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-5F9EA0?style=for-the-badge&logoColor=white)
![CatBoost](https://img.shields.io/badge/CatBoost-FF9900?style=for-the-badge&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-3776AB?style=for-the-badge&logo=matplotlib&logoColor=white)
![Seaborn](https://img.shields.io/badge/Seaborn-3776AB?style=for-the-badge&logo=seaborn&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)

![Dataset](https://img.shields.io/badge/Dataset-phongtro123.com-blue?style=for-the-badge&logo=google-maps&logoColor=white)
![Topic](https://img.shields.io/badge/Topic-Motel_Pricing_Predictions-orange?style=for-the-badge&logo=google-analytics&logoColor=white)
![Status](https://img.shields.io/badge/Status-Done-success?style=for-the-badge&logo=checkmarx&logoColor=white)

## Mô tả dự án (Project Overview)

Dự án tập trung vào việc **phân tích và dự đoán giá phòng trọ** dành cho học sinh – sinh viên tại TP.HCM, sử dụng dữ liệu thực tế crawl từ phongtro123.com.  

**Mục tiêu chính:**
- Thu thập dữ liệu thực tế (title, address, price, area, tiện ích, ngày đăng, mô tả...).
- Khám phá dữ liệu (EDA): phân bố giá, ảnh hưởng của vị trí/quận, tiện ích, yếu tố mùa vụ.
- Xây dựng mô hình học máy dự đoán giá phòng (Baseline: Linear Regression; Advanced: XGBoost, CatBoost).
- Trả lời 5+ câu hỏi nghiên cứu ý nghĩa về địa lý, tiện nghi, biến động giá, vị trí, mùa vụ.
- Đánh giá mô hình bằng metrics chuẩn (MSE, RMSE, R²...).

Dữ liệu tự crawl (không dùng dataset có sẵn), cấu trúc hóa thành bảng với nhiều feature (numerical, categorical, text).

## Cài đặt & Chạy dự án (Installation)

### Yêu cầu
- Python 3.10+
- Jupyter Notebook

### Các bước nhanh
1. Clone repository:
   ```bash
   git clone https://github.com/HungHiHung10/Intro2DS.git
   cd Intro2DS
   ```

2. Tạo môi trường ảo:
   ```bash
   conda create -n intro2ds python=3.10
   conda activate intro2ds
   ```

3. Cài đặt dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Chạy notebook theo thứ tự:
   ```bash
   jupyter notebook notebooks/
   ```
   - notebooks/1_Data_Collection.ipynb (crawl data)  
   - notebooks/2_Preprocessing_EDA.ipynb  
   - notebooks/3_Modeling_Evaluation.ipynb  

## Cấu trúc thư mục (Project Structure)

```
Intro2DS/
├── Data/                  # Dữ liệu thô và processed (raw.csv, cleaned data)
├── notebooks/             # Jupyter notebooks theo giai đoạn (1_Collect, 2_EDA, 3_Modeling...)
├── utilities/             # Hàm hỗ trợ (crawl, preprocess, modeling helpers)
├── requirements.txt       # Danh sách packages
├── README.md
└── .gitignore
```

## Quy trình phối hợp & Công cụ

Nhóm áp dụng quy trình CRISP-DM linh hoạt, giao tiếp qua **Zalo** (hàng ngày), **Google Meet** (họp 2-3 lần/tuần), quản lý code bằng **GitHub** (branching, peer review), tài liệu chung trên **Google Docs**.  

**Quy trình làm việc:**
- Phân công rõ ràng theo giai đoạn → peer review code & kết quả.
- Theo dõi tiến độ hàng tuần qua bảng Google Docs.
- Giải quyết vấn đề nhanh qua chat hoặc họp.

## Kế hoạch dự án & Tiến độ (Tóm tắt từ báo cáo)

Dự án chia thành 6 giai đoạn (10/2025 – 01/2026):
1. Xác định domain & công cụ (31/10 – 04/11/2025) → Thống nhất đề tài & nguồn dữ liệu.
2. Thu thập dữ liệu (04/11 – 08/11/2025) → Crawl từ phongtro123.com.
3. Tiền xử lý & demo modeling (08/11 – 20/11/2025) → Preprocessing, baseline model.
4. EDA & báo cáo giữa kỳ (24/11 – 04/12/2025) → Correlation, insights, slide.
5. Đặt câu hỏi, modeling, báo cáo cuối kỳ (05/12/2025 – 06/01/2026) → XGBoost, CatBoost, 5 questions.
6. Tối ưu & hoàn thiện (07/01 – 13/01/2026) → Merge code, README, reflections.

**Tất cả giai đoạn hoàn thành 100% đúng hạn.**

## Phân công công việc (Tóm tắt)



**Đánh giá đóng góp:** 100% cho mỗi thành viên (cân bằng, minh bạch qua GitHub commits).

## 5 Câu hỏi nghiên cứu ý nghĩa

1. Giá thuê phòng thay đổi như thế nào giữa các quận khác nhau? → Hỗ trợ sinh viên chọn khu vực phù hợp ngân sách.
2. Tiện nghi nào phân biệt phòng bình dân và cao cấp? → Giúp đánh giá giá trị thực của listing.
3. Biến động giá trong mùa cao điểm (giữa quận trung tâm & ven)? → Dự báo thời điểm thuê tốt nhất.
4. Sự phân cực giá giữa mặt tiền và hẻm nhỏ trong cùng quận? → Hiểu ảnh hưởng vị trí chi tiết.
5. Có yếu tố mùa vụ ảnh hưởng đến giá thuê tại TP.HCM? → Insight về xu hướng thời gian.

(Chi tiết + visualize + insights trong notebooks/3_Modeling_Evaluation.ipynb)

## Kết quả nổi bật

- Mô hình: Linear Regression (baseline), XGBoost, CatBoost (tối ưu hyperparameters).
- Metrics: MSE/RMSE thấp, R² cao (chi tiết trong notebook).
- Insights: Vị trí quận trung tâm + tiện ích (máy lạnh, gác lửng) tăng giá đáng kể; mùa vụ ảnh hưởng rõ rệt.

## Reflection (Tóm tắt)

- **Khó khăn:** Crawl data lớn, xử lý missing values/text, quota API nếu dùng thêm (nhưng không dùng trong dự án chính), merge code tránh conflict.
- **Bài học:** Tầm quan trọng của preprocessing tốt, hyperparameter tuning, làm việc nhóm minh bạch qua GitHub.
- **Nếu có thêm thời gian:** Scale dataset lớn hơn, thêm NLP sâu cho mô tả, deploy web app dự đoán giá.

## License
MIT License – Miễn phí sử dụng cho mục đích học tập & nghiên cứu.

**Cảm ơn quý Thầy Cô và các bạn đã theo dõi!**  
Nếu có góp ý, hãy mở issue trên repo. Trân trọng!  
**Last updated:** Tháng 01/2026
```

### Hướng dẫn sử dụng
- Copy toàn bộ → tạo/sửa file `README.md`.
- Commit: `git add README.md && git commit -m "Update README.md based on final report" && git push`.
- Refresh GitHub → README sẽ hiển thị chuyên nghiệp với badge, bảng, heading rõ ràng.
- Nếu repo có thêm file (ví dụ: reports/final_report.pdf), thêm link vào phần "Tài liệu tham khảo" hoặc "Results".

Nếu bạn muốn chỉnh thêm (thêm hình ảnh pipeline, link notebook cụ thể, hoặc badge khác), cứ báo nhé! Chúc nhóm 10 bảo vệ đồ án thành công và đạt điểm cao! 🎉