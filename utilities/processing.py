import pandas as pd
import numpy as np
import unicodedata
import re
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
import os

# Khởi tạo công cụ định vị và cấu hình giới hạn tần suất để tuân thủ chính sách API
geolocator = Nominatim(user_agent="hcm_rent_price")
geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1)

# Bộ nhớ đệm lưu trữ kết quả để tránh truy vấn lặp lại cùng một địa chỉ
geo_cache = {}

def key_case_insensitive(s):
    """Chuẩn hóa chuỗi làm khóa so sánh không phân biệt hoa thường và Unicode"""
    if pd.isna(s):
        return None
    # Đưa về chuẩn NFC để đồng nhất các ký tự tiếng Việt có dấu
    return unicodedata.normalize("NFC", str(s)).lower().strip()

def normalize_vn_title(s):
    """Làm sạch tiêu đề tiếng Việt và định dạng viết hoa chữ cái đầu mỗi từ"""
    if pd.isna(s):
        return np.nan
    s = unicodedata.normalize("NFC", str(s).strip())
    # Thay thế nhiều khoảng trắng liên tiếp bằng một khoảng trắng đơn
    s = re.sub(r"\s+", " ", s)
    s = s.lower()
    # Tách từ và viết hoa chữ cái đầu để chuẩn hóa hiển thị
    s = " ".join(w.capitalize() for w in s.split(" "))
    return s

def audit_case_variants(df, cols, top_n=10, show=True):
    """Phát hiện các giá trị trùng lặp nội dung nhưng khác nhau về cách viết hoa thường"""
    results = {}

    for c in cols:
        tmp = df[[c]].dropna().copy()
        # Tạo cột trung gian đã chuẩn hóa để làm khóa định danh
        tmp["_key"] = tmp[c].apply(key_case_insensitive)

        # Đếm số lượng biến thể viết khác nhau cho cùng một nội dung gốc
        dup = (
            tmp.groupby("_key")[c]
            .nunique()
            .reset_index(name="n_variant")
            .query("n_variant > 1")
            .sort_values("n_variant", ascending=False)
        )

        results[c] = dup

        print(f"\n{c}: số nhóm khác nhau chỉ do hoa/thường = {len(dup)}")

        if show and len(dup) > 0:
            # Hiển thị danh sách các cách viết khác nhau để kiểm tra thủ công
            display(
                tmp[tmp["_key"].isin(dup["_key"])]
                .groupby("_key")[c]
                .unique()
                .head(top_n)
            )

    return results

def clean_text(s):
    """Chuẩn hóa định dạng Unicode NFC và loại bỏ khoảng trắng dư thừa trong văn bản"""
    if pd.isna(s):
        return s
    s = unicodedata.normalize("NFC", s)
    # Loại bỏ các ký tự xuống dòng hoặc tab gây nhiễu
    s = re.sub(r"\s+", " ", s)
    return s.strip()

def is_nfc(text):
    """Kiểm tra xem chuỗi văn bản đã tuân thủ chuẩn Unicode NFC hay chưa"""
    return text == unicodedata.normalize("NFC", text)

def build_area_nominatim(address):
    """Xây dựng cấu trúc địa chỉ đầy đủ để tối ưu hóa kết quả tìm kiếm trên Nominatim"""
    if pd.isna(address):
        return None

    area = str(address).strip()
    area = unicodedata.normalize("NFC", area)

    # Chuyển đổi các ký tự phân cách không chuẩn về dạng phẩy để API dễ nhận diện
    area = area.replace(" - ", ", ").replace("-", ", ")
    area = re.sub(r"\s*,\s*", ", ", area)
    area = re.sub(r"\s+", " ", area).strip()

    # Xử lý trường hợp đặc biệt cho Thành phố Thủ Đức để tăng độ chính xác địa lý
    area = re.sub(r"^Thủ Đức\b", "Thành phố Thủ Đức", area)

    # Bổ sung thông tin tỉnh thành và quốc gia để thu hẹp phạm vi tìm kiếm
    if "Thành phố Hồ Chí Minh" not in area:
        area = f"{area}, Thành phố Hồ Chí Minh"

    if "Việt Nam" not in area:
        area = f"{area}, Việt Nam"

    return area

def setup_geocoder():
    """Cấu hình bộ định vị địa lý với các tham số thử lại và thời gian chờ an toàn"""
    geolocator = Nominatim(user_agent="my_rental_analysis_app")
    # Thiết lập cơ chế tự động thử lại khi gặp lỗi mạng hoặc quá tải (Rate Limit)
    return RateLimiter(
        geolocator.geocode,
        min_delay_seconds=1,
        max_retries=3,
        error_wait_seconds=3,
        swallow_exceptions=True
    )

def get_lat_lon(addr, geocode_func):
    """Truy vấn tọa độ địa lý (vĩ độ, kinh độ) từ một chuỗi địa chỉ văn bản"""
    try:
        # Giới hạn thời gian chờ để tránh treo tiến trình khi API phản hồi chậm
        loc = geocode_func(addr, timeout=15)
        return (loc.latitude, loc.longitude) if loc else (np.nan, np.nan)
    except Exception:
        # Trả về NaN nếu có lỗi phát sinh để không làm dừng luồng xử lý chính
        return (np.nan, np.nan)

def load_cache(path="geocode_cache.csv"):
    """Tải dữ liệu tọa độ đã được truy vấn trước đó từ tệp lưu trữ cục bộ"""
    if os.path.exists(path):
        cache_df = pd.read_csv(path)
        cache_df["query"] = cache_df["query"].astype(str)
        # Loại bỏ các bản ghi trùng lặp trong cache, giữ lại kết quả mới nhất
        cache_df = cache_df.drop_duplicates(subset=["query"], keep="last")
        # Chuyển đổi DataFrame thành từ điển để truy xuất với độ phức tạp O(1)
        cache = dict(zip(cache_df["query"], zip(cache_df["lat"], cache_df["lon"])))
        return cache
    return {}

def save_cache(cache, path="geocode_cache.csv"):
    """Lưu trữ kết quả truy vấn tọa độ mới vào tệp cục bộ để tái sử dụng"""
    # Chuyển đổi từ điển ngược lại thành DataFrame để lưu trữ bền vững
    cache_df = pd.DataFrame(
        [(q, v[0], v[1]) for q, v in cache.items()],
        columns=["query", "lat", "lon"]
    )
    cache_df.to_csv(path, index=False)

def haversine_km(lat1, lon1, lat2, lon2):
    """Tính toán khoảng cách đường chim bay giữa hai điểm trên trái đất theo km"""
    R = 6371.0 # Bán kính trung bình của Trái đất tính bằng km
    # Chuyển đổi đơn vị độ sang radian để tính toán lượng giác
    lat1 = np.radians(lat1.astype(float))
    lon1 = np.radians(lon1.astype(float))
    lat2 = np.radians(float(lat2))
    lon2 = np.radians(float(lon2))

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    # Áp dụng công thức Haversine để tính khoảng cách trên mặt cầu
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c

def _clean_str_series(s: pd.Series) -> pd.Series:
    """Hàm bổ trợ để làm sạch khoảng trắng cho toàn bộ một cột dữ liệu dạng chuỗi"""
    return s.astype(str).str.strip()

# Biểu thức chính quy để nhận diện tên Quận/Huyện dựa trên định dạng địa chỉ TP.HCM
DISTRICT_PATTERN = re.compile(
    r"^\s*(Quận\s+[^\-–—]+|Huyện\s+[^\-–—]+|Thủ\s*Đức)\s*[-–—]\s*Hồ\s*Chí\s*Minh\s*$",
    flags=re.UNICODE
)

def extract_district_from_address(addr):
    """Trích xuất tên Quận hoặc Huyện từ chuỗi địa chỉ cung cấp"""
    if not isinstance(addr, str):
        return None

    addr = addr.strip()
    # Tìm kiếm phần khớp với mô hình Quận/Huyện ở đầu chuỗi địa chỉ
    m = DISTRICT_PATTERN.match(addr)
    return m.group(1).strip() if m else None

def add_district_median_price(
    df: pd.DataFrame,
    price_col: str = "price",
    district_col: str = "district",
    out_col: str = "district_median_price",
):
    """Tính toán và gán giá trị giá thuê trung vị theo từng đơn vị Quận/Huyện"""
    df = df.copy()
    # Chuyển đổi giá về dạng số để đảm bảo tính toán thống kê chính xác
    df[price_col] = pd.to_numeric(df[price_col], errors="coerce")
    df[district_col] = _clean_str_series(df[district_col])

    # Tính toán giá trị trung vị (median) thay vì trung bình để tránh ảnh hưởng của outliers
    med = df.groupby(district_col)[price_col].median()
    # Ánh xạ kết quả trung vị ngược lại cho từng bản ghi dựa trên tên quận
    df[out_col] = df[district_col].map(med)
    return df

def add_hot_street_by_relative_price(
    df: pd.DataFrame,
    price_col: str = "price",
    district_col: str = "district",
    street_col: str = "street_name",
    district_median_col: str = "district_median_price",
    min_ads_street: int = 15,
    premium_ratio: float = 1.20,
    street_median_col: str = "street_median_price",
    rel_col: str = "street_relative_price",
    hot_col: str = "is_hot_street",
):
    """Xác định các tuyến đường có mức giá cao vượt trội so với mặt bằng chung của Quận"""
    df = df.copy()
    df[price_col] = pd.to_numeric(df[price_col], errors="coerce")
    
    # Bước 1: Tính số lượng tin đăng để đảm bảo tính đại diện thống kê của tuyến đường
    street_counts = (
        df.groupby([district_col, street_col])
          .size()
          .rename("street_ads_count")
    )
    df = df.merge(street_counts.reset_index(), on=[district_col, street_col], how="left")

    # Bước 2: Chỉ tính giá trung vị cho các đường có đủ số lượng tin tối thiểu
    valid_mask = df["street_ads_count"] >= min_ads_street
    street_median = (
        df.loc[valid_mask]
          .groupby([district_col, street_col])[price_col]
          .median()
    )
    df[street_median_col] = (
        df.set_index([district_col, street_col]).index.map(street_median)
    )

    # Bước 3: Tính tỷ lệ chênh lệch giá giữa đường và mặt bằng chung của Quận
    df[rel_col] = df[street_median_col] / df[district_median_col]

    # Bước 4: Đánh dấu 1 (Hot) nếu tỷ lệ chênh lệch vượt ngưỡng (mặc định 20%)
    df[hot_col] = (
        (df[rel_col] >= premium_ratio) &
        df[rel_col].notna()
    ).astype(int)

    return df

def iqr_cap(series, k=1.5):
    """Giới hạn các giá trị ngoại lai dựa trên phương pháp khoảng phân vị IQR"""
    # Xác định ngưỡng phân vị 25% và 75%
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    # Tính toán ranh giới chấp nhận được (ngưỡng hàng rào)
    lower = q1 - k * iqr
    upper = q3 + k * iqr
    # Thay thế các giá trị nằm ngoài ranh giới bằng giá trị biên (Capping)
    return series.clip(lower, upper)