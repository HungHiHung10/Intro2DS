# PHÂN TÍCH GIÁ PHÒNG TRỌ TP.HCM (MOTEL PRICING ANALYSIS)

## **Mục lục**

- [PHÂN TÍCH GIÁ PHÒNG TRỌ TP.HCM (MOTEL PRICING ANALYSIS)](#phân-tích-giá-phòng-trọ-tphcm-motel-pricing-analysis)
  - [**Mục lục**](#mục-lục)
  - [**1. Tổng Quan Dự Án & Thông Tin Nhóm**](#1-tổng-quan-dự-án--thông-tin-nhóm)
    - [Tổng quan](#tổng-quan)
    - [Thông tin nhóm](#thông-tin-nhóm)
  - [**2. Nguồn Dữ Liệu & Mô Tả**](#2-nguồn-dữ-liệu--mô-tả)
  - [**3. Danh Sách Câu Hỏi Nghiên Cứu**](#3-danh-sách-câu-hỏi-nghiên-cứu)
  - [**4. Tóm Tắt Kết Quả Chính**](#4-tóm-tắt-kết-quả-chính)
  - [**5. Cấu Trúc Thư Mục**](#5-cấu-trúc-thư-mục)
  - [**6. Hướng Dẫn Cài Đặt & Chạy**](#6-hướng-dẫn-cài-đặt--chạy)
    - [6.1. Yêu cầu hệ thống](#61-yêu-cầu-hệ-thống)
    - [6.2. Tạo môi trường & cài đặt thư viện](#62-tạo-môi-trường--cài-đặt-thư-viện)
    - [6.3. Chạy Notebook đúng thứ tự](#63-chạy-notebook-đúng-thứ-tự)
  - [**7. Danh Sách Thư Viện**](#7-danh-sách-thư-viện)
    - [7.1. Xử lý dữ liệu & khoa học dữ liệu](#71-xử-lý-dữ-liệu--khoa-học-dữ-liệu)
    - [7.2. Trực quan hoá dữ liệu](#72-trực-quan-hoá-dữ-liệu)
    - [7.3. Machine Learning](#73-machine-learning)
    - [7.4. Hỗ trợ Notebook & AI mở rộng](#74-hỗ-trợ-notebook--ai-mở-rộng)
  - [**8. License**](#8-license)

---

## **1. Tổng Quan Dự Án & Thông Tin Nhóm**

### Tổng quan

Dự án này là đồ án cuối kỳ môn **Nhập môn Khoa học Dữ liệu (CSC17104)**. Nhóm thực hiện quy trình khoa học dữ liệu hoàn chỉnh nhằm phân tích và dự đoán giá thuê phòng trọ tại TP.HCM. Dữ liệu được thu thập từ website phongtro123.com, qua đó khám phá các yếu tố ảnh hưởng đến giá thuê và xây dựng mô hình Machine Learning để dự đoán giá dựa trên diện tích, vị trí và các tiện nghi đi kèm. Ngoài ra nhóm còn thực hiện thêm phần mở rộng "Kiểm tra tính xác thực của bài đăng" giúp người thuê tránh bị hụt hẩn hay mất tiền oan.

### Thông tin nhóm

**Nhóm:** 10

**Giảng viên hướng dẫn:** Thầy Lê Nhựt Nam, Cô Võ Nam Thục Đoan

| STT | MSSV | Họ và Tên | Phụ trách | Đóng góp |
|:---:|:----:|:---|:---|:---:|
| 1 | 23120271 | Nguyễn Hữu Khánh Hưng | Crawl data, Baseline Linear Regression, Hyperparameter tuning, Merge code & README. | 100% |
| 2 | 23120283 | Phạm Quốc Khánh | Crawl data, XGBoost, Phân tích địa lý & tiện nghi (Câu 1, 2), Báo cáo. | 100% |
| 3 | 23120329 | Châu Huỳnh Phúc | Crawl data, Processing & Feature Engineering, Correlation & Insight, Mùa vụ (Câu 5), Quản lý tiến độ. | 100% |
| 4 | 23120333 | Vũ Trần Phúc | Crawl data, CatBoost, Biến động giá & vị trí (Câu 3, 4), Báo cáo tổng kết. | 100% |

---

## **2. Nguồn Dữ Liệu & Mô Tả**

- **Nguồn dữ liệu:** https://phongtro123.com
- **Mô tả:** Bộ dữ liệu chứa thông tin chi tiết về các tin đăng cho thuê phòng trọ tại khu vực TP.HCM.
- **Kích thước:** 24,122 dòng và 17 cột (Dung lượng 4.3 MB).
- **Các đặc trưng chính (Features):**
  - `title`: Tiêu đề bài đăng.
  - `address`: Địa chỉ chi tiết.
  - `price`: Giá thuê (Biến mục tiêu - Target variable).
  - `area`: Diện tích phòng trọ.
  - `noithat`, `gac`, `kebep`, `maylanh`, `maygiat`, `tulanh`, `thangmay`: Các tiện nghi đi kèm.
  - `chungchu`, `giotu`, `hamxe`: Các quy định và cơ sở hạ tầng.
  - `ngaydang`, `thongtinmota`, `url`: Thông tin bổ trợ.

---

## **3. Danh Sách Câu Hỏi Nghiên Cứu**

Nhóm đã đặt ra 5 câu hỏi để khai thác dữ liệu:

1. **Câu hỏi 1 & 2:** Vị trí địa lý và các tiện nghi (nội thất, máy lạnh, thang máy...) ảnh hưởng như thế nào đến giá thuê phòng trọ? Đâu là những tiện nghi tác động mạnh nhất đến phân khúc giá cao?
2. **Câu hỏi 3 & 4:** Sự biến động giá thuê diễn ra như thế nào giữa các quận huyện tại TP.HCM? Có sự phân hóa rõ rệt về giá trung bình giữa khu vực trung tâm và vùng ven không?
3. **Câu hỏi 5:** Yếu tố mùa vụ (thời điểm đăng bài) có ảnh hưởng đến mức giá thuê hoặc số lượng tin đăng không? Liệu có "mùa cao điểm" thuê trọ tại TP.HCM?

---

## **4. Tóm Tắt Kết Quả Chính**

Dưới đây là những insight quan trọng rút ra từ quá trình phân tích:

- **Về mô hình dự đoán:** Các mô hình Boosting (XGBoost, CatBoost) cho kết quả vượt trội so với Linear Regression truyền thống.
- **Về yếu tố ảnh hưởng:** Vị trí (quận/huyện) và các tiện ích như Máy lạnh, Thang máy là những yếu tố dẫn dắt giá thuê quan trọng.
- **Kết quả Mô hình (Model Performance):**

| Model | R² | MAE (Triệu VND) | RMSE |
| :--- | :---: | :---: | :---: |
| **XGBoost Regressor** | **0.5518** | **0.5763** | **0.7607** |
| CatBoost Regressor | 0.4769 | 0.6438 | 0.9061 |
| Linear Regression Optimized | 0.3337 | 0.7630 | 1.0140 |

---

## **5. Cấu Trúc Thư Mục**

Dự án được tổ chức thành các thư mục và module như sau:

```text
Intro2DS/
├── Data/                    # Chứa dữ liệu các giai đoạn
│   ├── raw.csv              # Dữ liệu thô tổng hợp
│   ├── cleaned.csv          # Dữ liệu sạch cơ bản
│   ├── processed.csv        # Dữ liệu cuối cùng cho Modeling
│   └── Page...csv           # Dữ liệu thô theo từng trang crawl
│
├── notebooks/               # Các Jupyter Notebook theo quy trình
│   ├── CrawlData.ipynb      # Thu thập dữ liệu từ phongtro123
│   ├── ProcessingData.ipynb # Sơ chế và làm sạch cơ bản
│   ├── Pre_Processing.ipynb # Tiền xử lý nâng cao & Feature Engineering
│   ├── EDA.ipynb            # Khám phá dữ liệu & Biểu đồ phân phối
│   ├── Analysis.ipynb       # Trả lời câu hỏi nghiên cứu
│   ├── Modeling.ipynb       # Xây dựng và đánh giá mô hình
│   └── Bonus_Extension.ipynb # Mở rộng: Kiểm chứng tiện ích bằng Gemini AI
│
├── utilities/               # Các module Python hỗ trợ
│   ├── analysis.py          # Hàm phân tích, tính toán metrics
│   ├── explore_data.py      # Hàm EDA
│   ├── modeling.py          # Hàm xây dựng model
│   ├── processing.py        # Hàm xử lý text, regex
│   └── visualize.py         # Hàm vẽ biểu đồ
│
├── README.md                # File hướng dẫn này
└── requirements.txt         # Danh sách thư viện
```

## **6. Hướng Dẫn Cài Đặt & Chạy**

### 6.1. Yêu cầu hệ thống

* **Python:** 3.10+
* **Jupyter Notebook / JupyterLab:** để chạy các file `.ipynb`
* **API Key:** Google Gemini API key (dùng cho phần `Bonus_Extension.ipynb`) - Đã có sẳn trong notebook (Chú ý đến Quota)

---

### 6.2. Tạo môi trường & cài đặt thư viện

1. **Clone repository:**
```bash
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

### 6.3. Chạy Notebook đúng thứ tự

Để đảm bảo luồng dữ liệu và các file trung gian được tạo đầy đủ, vui lòng thực hiện chạy các notebook trong thư mục `notebooks/` theo đúng trình tự sau:

**1. Giai đoạn Thu thập**
* **CrawlData.ipynb**: Thực hiện crawl dữ liệu từ website phongtro123.com.

**2. Giai đoạn Tiền xử lý & Khám phá**
* **Pre_Processing.ipynb**: Chuyển đổi kiểu dữ liệu, sử dụng Regex để trích xuất thuộc tính từ mô tả và xử lý các giá trị thiếu.
* **EDA.ipynb**: Thực hiện thống kê mô tả, vẽ Heatmap tương quan và xử lý các giá trị ngoại lai (Outliers).

**3. Giai đoạn Phân tích & Mô hình hóa**
* **ProcessingData.ipynb**: Xử lí chuẩn hóa dữ liệu, xử lí outliers & missing value. Đồng thời thực hiện Feature Engineering.
* **Analysis.ipynb**: Trực quan hóa các insight quan trọng để trả lời 5 câu hỏi nghiên cứu chính của dự án.
* **Modeling.ipynb**: Chia tập dữ liệu, huấn luyện các mô hình Machine Learning (XGBoost, CatBoost, Linear Regression) và đánh giá hiệu suất.
* **Bonus_Extension.ipynb**: Phần mở rộng thêm - Sử dụng mô hình Gemini 2.5 Flash để đối chiếu tính xác thực của tiện ích qua hình ảnh.

---

## **7. Danh Sách Thư Viện**

Dự án sử dụng các thư viện Python chuyên dụng sau:

### 7.1. Xử lý dữ liệu & khoa học dữ liệu
* `numpy`
* `pandas`
* `scipy`

### 7.2. Trực quan hoá dữ liệu
* `matplotlib`
* `seaborn`

### 7.3. Machine Learning
* `scikit-learn`
* `xgboost`
* `catboost`

### 7.4. Hỗ trợ Notebook & AI mở rộng
* `jupyter`
* `google-generativeai` (Sử dụng cho Gemini API)
* `selenium` / `beautifulsoup4` (Sử dụng cho quá trình Crawl data)

---

## **8. License**

Dự án này được phát hành dưới giấy phép **MIT License** - Phục vụ mục đích học tập và nghiên cứu.