import os
import pandas as pd
import re

def parse_duration_to_minutes(duration_str):
    """
    '1 hours 5 minutes'와 같은 문자열을 분(minute) 단위 정수로 변환합니다.
    """
    if pd.isna(duration_str) or not isinstance(duration_str, str):
        return 0
    
    hours = 0
    minutes = 0
    
    # 정규식으로 시간과 분 추출
    h_match = re.search(r'(\d+)\s*hours?', duration_str)
    m_match = re.search(r'(\d+)\s*minutes?', duration_str)
    
    if h_match:
        hours = int(h_match.group(1))
    if m_match:
        minutes = int(m_match.group(1))
        
    return hours * 60 + minutes

def analyze_csv_columns(root_folder):
    # 1. 분석 대상 컬럼 (소문자로 정규화)
    # 여기에 포함된 컬럼만 분석합니다.
    target_cols = {
        'room type', 'price', 'minimum nights', 'review rate number', 'house_rules',
        'maximum occupancy', 'city', 'cuisines', 'average cost', 'aggregate rating',
        'deptime', 'arrtime', 'actualelapsedtime', 'flightdate', 'origincityname',
        'destcityname', 'distance', 'latitude', 'longitude', 'origin', 'destination',
        'duration'
    }

    # 2. 제외할 컬럼 (혹시 target_cols에 포함되었더라도 제외하고 싶을 때 사용)
    # name, address, website 등은 이미 target_cols에 없으므로 자동 제외됨.
    exclude_cols = {'name', 'website', 'phone', 'address'}

    print(f"Scanning folder: {root_folder}...\n")

    for root, dirs, files in os.walk(root_folder):
        for file in files:
            if file.endswith(".csv"):
                file_path = os.path.join(root, file)
                print(f"{'='*60}")
                print(f"File: {file}")
                print(f"{'='*60}")
                
                try:
                    df = pd.read_csv(file_path)
                except Exception as e:
                    print(f"Error reading {file}: {e}")
                    continue

                for col in df.columns:
                    col_lower = col.lower().strip()
                    
                    # 분석 대상인지 확인
                    if col_lower not in target_cols:
                        continue
                    if col_lower in exclude_cols:
                        continue
                        
                    col_data = df[col].dropna() # NaN 제외
                    
                    print(f"\n[{col}]")
                    
                    # 데이터가 없는 경우
                    if len(col_data) == 0:
                        print("  (Empty)")
                        continue

                    # 1. 숫자형 데이터 (범위 및 평균)
                    if pd.api.types.is_numeric_dtype(col_data):
                        # distance 컬럼인데 문자열로 되어 있는 경우(예: "1,234 km")는 아래 문자열 처리 로직으로 넘어감
                        # 순수 숫자형인 경우만 여기서 처리
                        print(f"  Type: Numeric")
                        print(f"  Range: {col_data.min()} ~ {col_data.max()}")
                        print(f"  Mean: {col_data.mean():.2f}")
                        
                    # 2. 시간/기간 관련 (heuristic: 컬럼명에 time, date, duration, elapsed 포함)
                    elif any(k in col_lower for k in ['time', 'date', 'duration', 'elapsed']):
                        sample = str(col_data.iloc[0])
                        
                        # "1 hours 30 mins" 같은 Duration 문자열인 경우
                        if 'hour' in sample or 'min' in sample:
                            # 파싱해서 분 단위로 변환 후 min/max 계산
                            minutes_series = col_data.apply(parse_duration_to_minutes)
                            min_idx = minutes_series.idxmin()
                            max_idx = minutes_series.idxmax()
                            
                            print(f"  Type: Duration (String)")
                            print(f"  Range: '{col_data[min_idx]}' ({minutes_series.min()}m) ~ '{col_data[max_idx]}' ({minutes_series.max()}m)")
                        
                        # "10:30", "2022-01-01" 같은 일반 날짜/시간 문자열
                        else:
                            try:
                                sorted_vals = sorted(col_data.unique())
                                print(f"  Type: Date/Time (String)")
                                print(f"  Range: {sorted_vals[0]} ~ {sorted_vals[-1]}")
                            except:
                                print(f"  Type: Date/Time (Unsortable)")
                                print(f"  Unique Count: {col_data.nunique()}")

                    # 3. 그 외 (범주형/문자열) - Unique Values 출력
                    else:
                        # distance 처럼 'km'가 붙어 문자열이 된 숫자 데이터인지 확인
                        sample = str(col_data.iloc[0])
                        if 'km' in sample.lower() and any(c.isdigit() for c in sample):
                             # 숫자만 추출해서 범위 계산 시도
                            try:
                                nums = col_data.astype(str).str.replace(',', '').str.extract(r'(\d+\.?\d*)')[0].astype(float)
                                print(f"  Type: Numeric (with unit)")
                                print(f"  Range: {nums.min()} ~ {nums.max()}")
                                continue
                            except:
                                pass # 실패하면 아래 범주형으로 처리

                        unique_vals = col_data.unique()
                        n_unique = len(unique_vals)
                        
                        print(f"  Type: Categorical/String")
                        print(f"  Unique Count: {n_unique}")
                        
                        # 값이 적으면 다 보여주고, 많으면 상위 5개만
                        if n_unique <= 15:
                            print(f"  Values: {list(unique_vals)}")
                        else:
                            top_counts = col_data.value_counts().head(5)
                            print(f"  Top 5 Frequent:")
                            for val, count in top_counts.items():
                                print(f"    - {val}: {count}")

# 실행
# 실제 경로로 수정해서 사용하세요.
target_path = 'benchmarks/travelplanner/official/database'
analyze_csv_columns(target_path)