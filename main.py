import flet as ft
import pandas as pd
import re
import datetime
import time
from datetime import timedelta
from collections import Counter
from functools import lru_cache
import numpy as np

# ==============================================================================
# 1. LOGIC CỐT LÕI (GIỮ NGUYÊN TỪ CODE GỐC)
# ==============================================================================

# Regex & Sets
RE_NUMS = re.compile(r'\d+')
RE_CLEAN_SCORE = re.compile(r'[^A-Z0-9]')
RE_ISO_DATE = re.compile(r'(20\d{2})[\.\-/](\d{1,2})[\.\-/](\d{1,2})')
RE_SLASH_DATE = re.compile(r'(\d{1,2})[\.\-/](\d{1,2})')
BAD_KEYWORDS = frozenset(['N', 'NGHI', 'SX', 'XIT', 'MISS', 'TRUOT', 'NGHỈ', 'LỖI'])

# --- HELPER FUNCTIONS ---
@lru_cache(maxsize=10000)
def get_nums(s):
    if pd.isna(s): return []
    s_str = str(s).strip()
    if not s_str: return []
    s_upper = s_str.upper()
    if any(kw in s_upper for kw in BAD_KEYWORDS): return []
    raw_nums = RE_NUMS.findall(s_upper)
    return [n.zfill(2) for n in raw_nums if len(n) <= 2]

@lru_cache(maxsize=1000)
def get_col_score(col_name, mapping_tuple):
    clean = RE_CLEAN_SCORE.sub('', str(col_name).upper().replace(' ', ''))
    mapping = dict(mapping_tuple)
    if 'M10' in clean: return mapping.get('M10', 0)
    for key, score in mapping.items():
        if key in clean:
            if key == 'M1' and 'M10' in clean: continue
            if key == 'M0' and 'M10' in clean: continue
            return score
    return 0

def parse_date_smart(col_str, f_m, f_y):
    s = str(col_str).strip().upper()
    s = s.replace('NGAY', '').replace('NGÀY', '').strip()
    match_iso = RE_ISO_DATE.search(s)
    if match_iso:
        y, p1, p2 = int(match_iso.group(1)), int(match_iso.group(2)), int(match_iso.group(3))
        if p1 != f_m and p2 == f_m: return datetime.date(y, p2, p1)
        return datetime.date(y, p1, p2)
    match_slash = RE_SLASH_DATE.search(s)
    if match_slash:
        d, m = int(match_slash.group(1)), int(match_slash.group(2))
        if m < 1 or m > 12 or d < 1 or d > 31: return None
        curr_y = f_y
        if m == 12 and f_m == 1: curr_y -= 1
        elif m == 1 and f_m == 12: curr_y += 1
        try: return datetime.date(curr_y, m, d)
        except: return None
    return None

def extract_meta_from_filename(filename):
    clean_name = filename.upper().replace(".CSV", "").replace(".XLSX", "")
    clean_name = re.sub(r'\s*-\s*', '-', clean_name) 
    y_match = re.search(r'202[0-9]', clean_name)
    y_global = int(y_match.group(0)) if y_match else datetime.datetime.now().year
    m_match = re.search(r'(?:THANG|THÁNG|T)[^0-9]*(\d{1,2})', clean_name)
    m_global = int(m_match.group(1)) if m_match else 12
    full_date_match = re.search(r'(\d{1,2})[\.\-](\d{1,2})(?:[\.\-]20\d{2})?', clean_name)
    if full_date_match:
        try:
            d = int(full_date_match.group(1))
            m = int(full_date_match.group(2))
            y = int(full_date_match.group(3)) if full_date_match.lastindex >= 3 else y_global
            if m == 12 and m_global == 1: y -= 1 
            elif m == 1 and m_global == 12: y += 1
            return m, y, datetime.date(y, m, d)
        except: pass
    return m_global, y_global, None

def find_header_row(df_preview):
    keywords = ["STT", "MEMBER", "THÀNH VIÊN", "TV TOP", "DANH SÁCH", "HỌ VÀ TÊN", "NICK"]
    for idx, row in df_preview.head(30).iterrows():
        row_str = str(row.values).upper()
        if any(k in row_str for k in keywords):
            return idx
    return 3

def fast_get_top_nums(df, p_map_dict, s_map_dict, top_n, min_v, inverse):
    cols_in_scope = sorted(list(set(p_map_dict.keys()) | set(s_map_dict.keys())))
    valid_cols = [c for c in cols_in_scope if c in df.columns]
    if not valid_cols or df.empty: return []
    sub_df = df[valid_cols].copy()
    melted = sub_df.melt(ignore_index=False, var_name='Col', value_name='Val')
    melted = melted.dropna(subset=['Val'])
    bad_pattern = r'N|NGHI|SX|XIT|MISS|TRUOT|NGHỈ|LỖI'
    mask_valid = ~melted['Val'].astype(str).str.upper().str.contains(bad_pattern, regex=True)
    melted = melted[mask_valid]
    if melted.empty: return []
    s_nums = melted['Val'].astype(str).str.findall(r'\d+')
    exploded = melted.assign(Num=s_nums).explode('Num')
    exploded = exploded.dropna(subset=['Num'])
    exploded['Num'] = exploded['Num'].str.strip().str.zfill(2)
    exploded = exploded[exploded['Num'].str.len() <= 2]
    exploded['P'] = exploded['Col'].map(p_map_dict).fillna(0)
    exploded['S'] = exploded['Col'].map(s_map_dict).fillna(0)
    stats = exploded.groupby('Num')[['P', 'S']].sum()
    votes = exploded.reset_index().groupby('Num')['index'].nunique()
    stats['V'] = votes
    stats = stats[stats['V'] >= min_v]
    if stats.empty: return []
    stats = stats.reset_index()
    stats['Num_Int'] = stats['Num'].astype(int)
    if inverse:
        stats = stats.sort_values(by=['P', 'S', 'Num_Int'], ascending=[False, False, True])
    else:
        stats = stats.sort_values(by=['P', 'V', 'Num_Int'], ascending=[False, False, True])
    return stats['Num'].head(int(top_n)).tolist()

def calculate_v24_logic_only(target_date, rolling_window, _cache, _kq_db, limits_config, min_votes, score_std, score_mod, use_inverse, manual_groups=None, max_trim=None):
    if target_date not in _cache: return None
    curr_data = _cache[target_date]
    df = curr_data['df']
    real_cols = df.columns
    p_map_dict = {}; s_map_dict = {}
    score_std_tuple = tuple(score_std.items()); score_mod_tuple = tuple(score_mod.items())
    for col in real_cols:
        s_p = get_col_score(col, score_std_tuple)
        if s_p > 0: p_map_dict[col] = s_p
        s_s = get_col_score(col, score_mod_tuple)
        if s_s > 0: s_map_dict[col] = s_s
    
    prev_date = target_date - timedelta(days=1)
    if prev_date not in _cache:
        for i in range(2, 4):
            if (target_date - timedelta(days=i)) in _cache:
                prev_date = target_date - timedelta(days=i); break
                
    col_hist_used = curr_data['hist_map'].get(prev_date)
    if not col_hist_used and prev_date in _cache:
        col_hist_used = _cache[prev_date]['hist_map'].get(prev_date)
    if not col_hist_used: return None

    groups = [f"{i}x" for i in range(10)]
    stats_std = {g: {'wins': 0, 'ranks': []} for g in groups}
    stats_mod = {g: {'wins': 0} for g in groups}
    
    if not manual_groups:
        past_dates = []
        check_d = target_date - timedelta(days=1)
        while len(past_dates) < rolling_window:
            if check_d in _cache and check_d in _kq_db: past_dates.append(check_d)
            check_d -= timedelta(days=1)
            if (target_date - check_d).days > 40: break
        
        for d in past_dates:
            d_df = _cache[d]['df']
            kq = _kq_db[d]
            d_p_map = {}; d_s_map = {}
            for col in d_df.columns:
                s_p = get_col_score(col, score_std_tuple)
                if s_p > 0: d_p_map[col] = s_p
                s_s = get_col_score(col, score_mod_tuple)
                if s_s > 0: d_s_map[col] = s_s
            
            d_hist_col = None
            sorted_dates = sorted([k for k in _cache[d]['hist_map'].keys() if k < d], reverse=True)
            if sorted_dates: d_hist_col = _cache[d]['hist_map'][sorted_dates[0]]
            if not d_hist_col: continue
            
            try:
                hist_series_d = d_df[d_hist_col].astype(str).str.upper().replace('S', '6', regex=False)
                hist_series_d = hist_series_d.str.replace(r'[^0-9X]', '', regex=True)
            except: continue
            
            for g in groups:
                mask = hist_series_d == g.upper()
                mems = d_df[mask]
                if mems.empty: stats_std[g]['ranks'].append(999); continue
                
                top80_std = fast_get_top_nums(mems, d_p_map, d_s_map, 80, min_votes, use_inverse)
                if kq in top80_std:
                    stats_std[g]['wins'] += 1
                    stats_std[g]['ranks'].append(top80_std.index(kq) + 1)
                else: stats_std[g]['ranks'].append(999)
                
                top86_mod = fast_get_top_nums(mems, d_s_map, d_p_map, int(limits_config['mod']), min_votes, use_inverse)
                if kq in top86_mod: stats_mod[g]['wins'] += 1
                
    final_std = []
    for g, inf in stats_std.items(): 
        final_std.append((g, -inf['wins'], sum(inf['ranks']), sorted(inf['ranks'])))
    final_std.sort(key=lambda x: (x[1], x[2], x[3], x[0])) 
    top6_std = [x[0] for x in final_std[:6]]
    best_mod_grp = sorted(stats_mod.keys(), key=lambda g: (-stats_mod[g]['wins'], g))[0]

    hist_series = df[col_hist_used].astype(str).str.upper().replace('S', '6', regex=False)
    hist_series = hist_series.str.replace(r'[^0-9X]', '', regex=True)

    def get_final_pool(group_list, limit_dict, p_map, s_map):
        pool = []
        for g in group_list:
            mask = hist_series == g.upper()
            valid_mems = df[mask]
            lim = limit_dict.get(g, limit_dict.get('default', 80))
            res = fast_get_top_nums(valid_mems, p_map, s_map, int(lim), min_votes, use_inverse)
            pool.extend(res)
        return pool

    limits_std = {
        top6_std[0]: limits_config['l12'], top6_std[1]: limits_config['l12'], 
        top6_std[2]: limits_config['l34'], top6_std[3]: limits_config['l34'], 
        top6_std[4]: limits_config['l56'], top6_std[5]: limits_config['l56']
    }
    
    g_set1 = [top6_std[0], top6_std[5], top6_std[3]]
    pool1 = get_final_pool(g_set1, limits_std, p_map_dict, s_map_dict)
    s1 = {n for n, c in Counter(pool1).items() if c >= 2} 
    
    g_set2 = [top6_std[1], top6_std[4], top6_std[2]]
    pool2 = get_final_pool(g_set2, limits_std, p_map_dict, s_map_dict)
    s2 = {n for n, c in Counter(pool2).items() if c >= 2}
    
    final_original = sorted(list(s1.intersection(s2)))
    
    mask_mod = hist_series == best_mod_grp.upper()
    final_modified = sorted(fast_get_top_nums(df[mask_mod], s_map_dict, p_map_dict, int(limits_config['mod']), min_votes, use_inverse))
    
    intersect_list = list(set(final_original).intersection(set(final_modified)))
    
    final_intersect = sorted(intersect_list)
    if max_trim and len(intersect_list) > max_trim:
        temp_df = df.copy()
        melted = temp_df.melt(value_name='Val').dropna(subset=['Val'])
        mask_bad = ~melted['Val'].astype(str).str.upper().str.contains(r'N|NGHI|SX|XIT', regex=True)
        melted = melted[mask_bad]
        s_nums = melted['Val'].astype(str).str.findall(r'\d+')
        exploded = melted.assign(Num=s_nums).explode('Num')
        exploded = exploded.dropna(subset=['Num'])
        exploded['Num'] = exploded['Num'].str.strip().str.zfill(2)
        exploded = exploded[exploded['Num'].isin(intersect_list)]
        exploded['Score'] = exploded['variable'].map(p_map_dict).fillna(0) + exploded['variable'].map(s_map_dict).fillna(0)
        final_scores = exploded.groupby('Num')['Score'].sum().reset_index()
        final_scores = final_scores.sort_values(by='Score', ascending=False)
        final_intersect = sorted(final_scores.head(int(max_trim))['Num'].tolist()) 

    return {
        "dan_goc": final_original, 
        "dan_mod": final_modified, 
        "dan_final": final_intersect, 
        "source_col": col_hist_used,
        "debug_top6": top6_std,
        "debug_mod": best_mod_grp
    }

# ==============================================================================
# 2. APP STATE MANAGEMENT
# ==============================================================================
class AppState:
    def __init__(self):
        self.cache = {}
        self.kq_db = {}
        self.file_logs = []
        self.config = {
            "HC_STD": [0, 0, 5, 10, 15, 25, 30, 35, 40, 50, 60],
            "HC_MOD": [0, 5, 10, 20, 25, 45, 50, 40, 30, 25, 40],
            "L12": 82, "L34": 76, "L56": 70, "LMOD": 88,
            "ROLLING": 10, "TRIM": 65
        }

state = AppState()

# ==============================================================================
# 3. FLET GUI
# ==============================================================================

def main(page: ft.Page):
    page.title = "Quang Pro V56 Mobile"
    page.theme_mode = ft.ThemeMode.LIGHT
    page.scroll = ft.ScrollMode.ADAPTIVE
    page.padding = 10

    # --- UI Elements Declarations ---
    status_text = ft.Text("Sẵn sàng.", size=12, color="grey")
    result_list = ft.ListView(expand=1, spacing=10, padding=20)
    
    date_field = ft.TextField(
        label="Ngày soi (DD/MM/YYYY)", 
        value=datetime.datetime.now().strftime("%d/%m/%Y"),
        width=180
    )

    # --- Event Handlers ---

    def process_files(files):
        state.cache = {}
        state.kq_db = {}
        state.file_logs = []
        
        # Sort files
        sorted_files = sorted(files, key=lambda f: f.name)
        
        for file in sorted_files:
            try:
                f_m, f_y, date_from_name = extract_meta_from_filename(file.name)
                # Determine loading method based on file type
                if file.name.upper().endswith(".XLSX"):
                    # Excel loading logic
                    xls = pd.ExcelFile(file.path)
                    for sheet in xls.sheet_names:
                        # Simplified date extraction from sheet for mobile
                        s_date = date_from_name 
                        if not s_date: continue # Skip if cannot determine date
                        
                        preview = pd.read_excel(xls, sheet_name=sheet, header=None, nrows=30)
                        h_row = find_header_row(preview)
                        df = pd.read_excel(xls, sheet_name=sheet, header=h_row)
                        
                        # Process DF (Normalize)
                        df.columns = [str(c).strip().upper() for c in df.columns]
                        hist_map = {}
                        for col in df.columns:
                            if "UNNAMED" in col or col.startswith("M") or col in ["STT"]: continue
                            d_obj = parse_date_smart(col, f_m, f_y)
                            if d_obj: hist_map[d_obj] = col
                        
                        # Get Result (KQ)
                        kq_row = None
                        if not df.empty:
                            for c_idx in range(min(2, len(df.columns))):
                                try:
                                    if df.iloc[:, c_idx].astype(str).str.upper().str.contains(r'KQ|KẾT QUẢ').any():
                                        kq_row = df[df.iloc[:, c_idx].astype(str).str.upper().str.contains(r'KQ|KẾT QUẢ')].iloc[0]
                                        break
                                except: continue
                        if kq_row is not None:
                            for d_val, c_name in hist_map.items():
                                try:
                                    nums = get_nums(str(kq_row[c_name]))
                                    if nums: state.kq_db[d_val] = nums[0]
                                except: pass
                        
                        state.cache[s_date] = {'df': df, 'hist_map': hist_map}

                elif file.name.upper().endswith(".CSV"):
                    # CSV Loading
                    if not date_from_name: continue
                    # Try common encodings
                    df = None
                    for enc in ['utf-8-sig', 'utf-8', 'latin-1']:
                        try:
                            df = pd.read_csv(file.path, header=3, encoding=enc)
                            break
                        except: continue
                    if df is None: continue

                    df.columns = [str(c).strip().upper() for c in df.columns]
                    # Logic fallback M cols mapping (skipped for brevity, assuming standard format)
                    
                    hist_map = {}
                    for col in df.columns:
                        d_obj = parse_date_smart(col, f_m, f_y)
                        if d_obj: hist_map[d_obj] = col
                    
                    state.cache[date_from_name] = {'df': df, 'hist_map': hist_map}
                    
            except Exception as e:
                print(f"Error {file.name}: {e}")

        status_text.value = f"Đã nạp {len(state.cache)} ngày dữ liệu."
        status_text.color = "green"
        status_text.update()

    def pick_files_result(e: ft.FilePickerResultEvent):
        if e.files:
            status_text.value = "Đang đọc dữ liệu..."
            status_text.update()
            process_files(e.files)
            
    file_picker = ft.FilePicker(on_result=pick_files_result)
    page.overlay.append(file_picker)

    def on_run_click(e):
        try:
            d_str = date_field.value
            d_obj = datetime.datetime.strptime(d_str, "%d/%m/%Y").date()
        except:
            status_text.value = "Ngày sai định dạng!"
            status_text.color = "red"
            status_text.update()
            return

        if d_obj not in state.cache:
            status_text.value = f"Không có dữ liệu ngày {d_str}"
            status_text.color = "red"
            status_text.update()
            return

        # Prepare Config
        std_map = {f'M{i}': state.config["HC_STD"][i] for i in range(11)}
        mod_map = {f'M{i}': state.config["HC_MOD"][i] for i in range(11)}
        limits = {'l12': state.config["L12"], 'l34': state.config["L34"], 
                  'l56': state.config["L56"], 'mod': state.config["LMOD"]}
        
        res = calculate_v24_logic_only(
            d_obj, state.config["ROLLING"], state.cache, state.kq_db,
            limits, 1, std_map, mod_map, False, None, state.config["TRIM"]
        )

        result_list.controls.clear()
        
        if not res:
            result_list.controls.append(ft.Text("Lỗi tính toán hoặc thiếu dữ liệu quá khứ.", color="red"))
        else:
            # Display Results
            goc_str = ", ".join(res['dan_goc'])
            mod_str = ", ".join(res['dan_mod'])
            final_str = ", ".join(res['dan_final'])
            
            result_list.controls.append(
                ft.Container(
                    content=ft.Column([
                        ft.Text(f"📅 KẾT QUẢ NGÀY {d_str}", weight="bold", size=18),
                        ft.Divider(),
                        ft.Text(f"GỐC ({len(res['dan_goc'])}):", weight="bold"),
                        ft.Text(goc_str, selectable=True),
                        ft.Divider(),
                        ft.Text(f"MOD ({len(res['dan_mod'])}):", weight="bold"),
                        ft.Text(mod_str, selectable=True),
                        ft.Divider(),
                        ft.Text(f"🛡️ FINAL ({len(res['dan_final'])}):", weight="bold", color="blue", size=16),
                        ft.Text(final_str, selectable=True, weight="bold", size=16),
                        ft.Divider(),
                        ft.Text(f"Nguồn: {res['source_col']} | Top6: {res['debug_top6']}", size=12, italic=True)
                    ]),
                    bgcolor=ft.colors.BLUE_50, padding=10, border_radius=10
                )
            )
            
            if d_obj in state.kq_db:
                real = state.kq_db[d_obj]
                is_win = real in res['dan_final']
                msg = f"🏆 WIN: {real}" if is_win else f"💀 MISS: {real}"
                clr = "green" if is_win else "red"
                result_list.controls.append(
                    ft.Container(content=ft.Text(msg, size=20, weight="bold", color="white"), bgcolor=clr, padding=10, border_radius=5)
                )

        result_list.update()


    # --- Layout ---
    page.add(
        ft.Row([
            ft.Icon(ft.icons.SHIELD_MOON, size=30, color="blue"),
            ft.Text("Quang Handsome V56", size=20, weight="bold")
        ], alignment=ft.MainAxisAlignment.CENTER),
        
        ft.Divider(),
        
        ft.Row([
            ft.ElevatedButton("📂 Nạp File", icon=ft.icons.UPLOAD_FILE, on_click=lambda _: file_picker.pick_files(allow_multiple=True)),
            status_text
        ]),
        
        ft.Divider(),
        
        ft.Text("Cấu hình HardCore (Default):", weight="bold"),
        ft.Row([
            ft.TextField(label="Top 1-2", value=str(state.config["L12"]), width=70, on_change=lambda e: state.config.update({"L12": int(e.control.value)})),
            ft.TextField(label="Top 3-4", value=str(state.config["L34"]), width=70, on_change=lambda e: state.config.update({"L34": int(e.control.value)})),
            ft.TextField(label="Top 5-6", value=str(state.config["L56"]), width=70, on_change=lambda e: state.config.update({"L56": int(e.control.value)})),
            ft.TextField(label="Max Trim", value=str(state.config["TRIM"]), width=70, on_change=lambda e: state.config.update({"TRIM": int(e.control.value)})),
        ], scroll=ft.ScrollMode.HIDDEN),

        ft.Divider(),

        ft.Row([
            date_field,
            ft.ElevatedButton("🚀 CHẠY", on_click=on_run_click, bgcolor="blue", color="white")
        ], alignment=ft.MainAxisAlignment.SPACE_BETWEEN),
        
        ft.Divider(),
        result_list
    )

ft.app(target=main)
