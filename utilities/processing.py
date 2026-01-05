import pandas as pd
import numpy as np
import unicodedata
import re
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
import pandas as pd
import numpy as np
import os
import unicodedata

geolocator = Nominatim(user_agent="hcm_rent_price")
geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1)

geo_cache = {}

def key_case_insensitive(s):
    if pd.isna(s):
        return None
    return unicodedata.normalize("NFC", str(s)).lower().strip()

def normalize_vn_title(s):
    if pd.isna(s):
        return np.nan
    s = unicodedata.normalize("NFC", str(s).strip())
    s = re.sub(r"\s+", " ", s)
    s = s.lower()
    s = " ".join(w.capitalize() for w in s.split(" "))
    return s

def audit_case_variants(df, cols, top_n=10, show=True):
    """
    Kiểm tra các giá trị khác nhau chỉ do hoa/thường cho các cột trong `cols`.

    Parameters
    ----------
    df : pd.DataFrame
    cols : list[str]
        Danh sách cột cần kiểm tra
    top_n : int
        Số nhóm hiển thị (mỗi cột)
    show : bool
        Có display kết quả chi tiết hay không

    Returns
    -------
    result : dict
        {col: DataFrame(key, n_variant)}
    """
    results = {}

    for c in cols:
        tmp = df[[c]].dropna().copy()
        tmp["_key"] = tmp[c].apply(key_case_insensitive)

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
            display(
                tmp[tmp["_key"].isin(dup["_key"])]
                .groupby("_key")[c]
                .unique()
                .head(top_n)
            )

    return results

def clean_text(s):
    if pd.isna(s):
        return s
    s = unicodedata.normalize("NFC", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()

def is_nfc(text):
    return text == unicodedata.normalize("NFC", text)

def build_area_nominatim(address):
    if pd.isna(address):
        return None

    area = str(address).strip()
    area = unicodedata.normalize("NFC", area)

    # Chuẩn hoá dấu phân cách
    # "Quận 8 - Hồ Chí Minh" hoặc "Quận 8-..." (nếu còn sót)
    area = area.replace(" - ", ", ").replace("-", ", ")
    area = re.sub(r"\s*,\s*", ", ", area)
    area = re.sub(r"\s+", " ", area).strip()

    # Chuẩn hoá Thủ Đức
    area = re.sub(r"^Thủ Đức\b", "Thành phố Thủ Đức", area)

    if "Thành phố Hồ Chí Minh" not in area:
        area = f"{area}, Thành phố Hồ Chí Minh"

    # Thêm quốc gia
    if "Việt Nam" not in area:
        area = f"{area}, Việt Nam"

    return area

# Geocoder utils
def setup_geocoder():
    geolocator = Nominatim(user_agent="my_rental_analysis_app")
    return RateLimiter(
        geolocator.geocode,
        min_delay_seconds=1,
        max_retries=3,
        error_wait_seconds=3,
        swallow_exceptions=True
    )

def get_lat_lon(addr, geocode_func):
    try:
        loc = geocode_func(addr, timeout=15)
        return (loc.latitude, loc.longitude) if loc else (np.nan, np.nan)
    except Exception:
        return (np.nan, np.nan)

# Cache 
def load_cache(path="geocode_cache.csv"):
    if os.path.exists(path):
        cache_df = pd.read_csv(path)
        cache_df["query"] = cache_df["query"].astype(str)
        cache_df = cache_df.drop_duplicates(subset=["query"], keep="last")
        cache = dict(zip(cache_df["query"], zip(cache_df["lat"], cache_df["lon"])))
        return cache
    return {}

def save_cache(cache, path="geocode_cache.csv"):
    cache_df = pd.DataFrame(
        [(q, v[0], v[1]) for q, v in cache.items()],
        columns=["query", "lat", "lon"]
    )
    cache_df.to_csv(path, index=False)

# Caculate distance
def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0  # km
    lat1 = np.radians(lat1.astype(float))
    lon1 = np.radians(lon1.astype(float))
    lat2 = np.radians(float(lat2))
    lon2 = np.radians(float(lon2))

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c

def _clean_str_series(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip()

DISTRICT_PATTERN = re.compile(
    r"^\s*(Quận\s+[^\-–—]+|Huyện\s+[^\-–—]+|Thủ\s*Đức)\s*[-–—]\s*Hồ\s*Chí\s*Minh\s*$",
    flags=re.UNICODE
)

def extract_district_from_address(addr):
    if not isinstance(addr, str):
        return None

    addr = addr.strip()
    m = DISTRICT_PATTERN.match(addr)
    return m.group(1).strip() if m else None

def add_district_median_price(
    df: pd.DataFrame,
    price_col: str = "price",
    district_col: str = "district",
    out_col: str = "district_median_price",
) -> pd.DataFrame:
    df = df.copy()
    df[price_col] = pd.to_numeric(df[price_col], errors="coerce")
    df[district_col] = _clean_str_series(df[district_col])

    med = df.groupby(district_col)[price_col].median()
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
) -> pd.DataFrame:
    df = df.copy()

    df[price_col] = pd.to_numeric(df[price_col], errors="coerce")
    df[district_col] = df[district_col].astype(str).str.strip()
    df[street_col] = df[street_col].astype(str).str.strip()

    if district_median_col not in df.columns:
        raise ValueError(
            f"Thiếu cột {district_median_col}. Hãy tạo median theo quận trước."
        )

    # Đếm số tin theo (quận, đường)
    street_counts = (
        df.groupby([district_col, street_col])
          .size()
          .rename("street_ads_count")
    )

    df = df.merge(
        street_counts.reset_index(),
        on=[district_col, street_col],
        how="left"
    )

    # Chỉ giữ các (quận, đường) đủ số mẫu
    valid_mask = df["street_ads_count"] >= min_ads_street

    # Median giá theo (quận, đường)
    street_median = (
        df.loc[valid_mask]
          .groupby([district_col, street_col])[price_col]
          .median()
    )

    df[street_median_col] = (
        df.set_index([district_col, street_col]).index.map(street_median)
    )

    # So sánh với median quận
    df[rel_col] = df[street_median_col] / df[district_median_col]

    # Hot nếu đắt hơn quận >= premium_ratio
    df[hot_col] = (
        (df[rel_col] >= premium_ratio) &
        df[rel_col].notna()
    ).astype(int)

    return df

def iqr_cap(series, k=1.5):
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - k * iqr
    upper = q3 + k * iqr
    return series.clip(lower, upper)

