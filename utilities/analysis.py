import pandas as pd
import re
def classify_season(month):
    if month in [8, 9, 10]:
        return 'Cao điểm (Nhập học)'
    elif month in [2, 3]:
        return 'Cao điểm (Sau Tết)'
    else:
        return 'Bình thường'
def classify_zone(address):
    address_str = str(address)
    central_districts = ['Quận 1', 'Quận 3', 'Quận 5', 'Quận 10', 'Bình Thạnh', 'Phú Nhuận', 'Quận 2', 'Quận 4']
    for d in central_districts:
        pattern = re.compile(rf"{d}\b", re.IGNORECASE) 
        if pattern.search(address_str):
            return 'Trung tâm'
                
    return 'Ngoại thành'
def classify_street_type(row):
    address = str(row['location']).lower() if pd.notnull(row['location']) else ''
    title = str(row['description']).lower() if pd.notnull(row['description']) else ''
    full_text = address + " " + title
    
    alley_keywords = ['hẻm', 'ngõ', 'ngách', 'hẽm', 'gần mặt', 'sau lưng', 'cách mặt']
    if any(kw in full_text for kw in alley_keywords):
        return 'Hẻm (Alley)'

    if pd.notnull(row['location']):
        location_str = str(row['location']).lower()
        
        date_streets = ['3/2', '30/4', '19/5', '1/5', '2/9', '23/9', '26/3', '23/10']
        matches = re.findall(r'\d+\/\d+', location_str)
        
        for m in matches:
            if m not in date_streets:
                return 'Hẻm (Alley)'

    main_street_keywords = ['mặt tiền', 'mặt phố', 'mt đường', 'phố', 'đường chính']
    if any(kw in full_text for kw in main_street_keywords):
        return 'Mặt tiền (Main Street)'

    return 'Khác (Unknown)'
def clean_district_hcm(addr):
    if not isinstance(addr, str): return None
    addr = addr.lower()
    
    mapping = {
        'quận 1': ['quận 1', 'q1', 'q.1'],
        'quận 2': ['quận 2', 'q2', 'q.2'],
        'quận 3': ['quận 3', 'q3', 'q.3'],
        'quận 4': ['quận 4', 'q4', 'q.4'],
        'quận 5': ['quận 5', 'q5', 'q.5'],
        'quận 6': ['quận 6', 'q6', 'q.6'],
        'quận 7': ['quận 7', 'q7', 'q.7'],
        'quận 8': ['quận 8', 'q8', 'q.8'],
        'quận 9': ['quận 9', 'q9', 'q.9'],
        'quận 10': ['quận 10', 'q10', 'q.10'],
        'quận 11': ['quận 11', 'q11', 'q.11'],
        'quận 12': ['quận 12', 'q12', 'q.12'],
        'bình thạnh': ['bình thạnh'],
        'tân bình': ['tân bình'],
        'tân phú': ['tân phú'],
        'phú nhuận': ['phú nhuận'],
        'gò vấp': ['gò vấp'],
        'bình tân': ['bình tân'],
        'thủ đức': ['thủ đức'],
        'bình chánh': ['bình chánh'],
        'nhà bè': ['nhà bè'],
        'hóc môn': ['hóc môn'],
        'củ chi': ['củ chi'],
        'cần giờ': ['cần giờ']
    }
    
    for standard, variations in mapping.items():
        for var in variations:
            if re.search(r'\b' + re.escape(var) + r'\b', addr):
                return standard.title()
    return None