import json
import re
from pathlib import Path

from charge_grid.utils import OUTPUT_DIR

output_folder = OUTPUT_DIR
cache_folder = Path('../cache')
current_dir = Path(__file__).parent
asset_path = current_dir / 'assets'
CSS_PATH = asset_path / 'dashboard_style.css'
JS_PATH = asset_path / 'dashboard_script.js'

def choose_solution_files():
    print(f"🔍 Đang tìm kiếm file tại: {output_folder.resolve()}")

    if not output_folder.exists():
        print(f"❌ Không tìm thấy thư mục {output_folder.resolve()}")
        return []

    json_files = list(output_folder.glob('solution_*.json'))
    if not json_files:
        print(f"❌ Không có file 'solution_*.json' nào trong {output_folder.resolve()}")
        return []

    print("\n--- DANH SÁCH KẾT QUẢ ĐÃ LƯU ---")
    for idx, file_path in enumerate(json_files):
        print(f"[{idx}] {file_path.name}")

    try:
        choice_str = input("\nNhập các số thứ tự file muốn vẽ biểu đồ (cách nhau bằng dấu phẩy hoặc khoảng trắng), hoặc 'all' để chọn tất cả: ")
        
        if choice_str.strip().lower() == 'all':
            return json_files
            
        indices = [int(x) for x in re.findall(r'\d+', choice_str)]
        selected_files = [json_files[i] for i in indices if 0 <= i < len(json_files)]
        
        if not selected_files:
            print("❌ Không có file nào hợp lệ được chọn.")
        return selected_files
        
    except Exception as e:
        print(f"❌ Lỗi: {e}")
        return []

def load_chart_data(selected_files):
    all_data = {}
    
    for file_path in selected_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            history = data.get('generation_history', [])
            if not history:
                print(f"⚠️ Bỏ qua {file_path.name}: Không tìm thấy dữ liệu 'generation_history'.")
                continue
            
            metadata = data.get('metadata', {})
            config = metadata.get('configuration', {})
            
            mut_type = config.get('mutation_type', 'N/A')
            if isinstance(mut_type, list):
                mut_type = f"[{', '.join(map(str, mut_type))}]"
                
            cross_type = config.get('crossover_type', 'N/A')
            display_title = f"Mut: {mut_type} | Cross: {cross_type}"
            
            mut_prob = config.get('mutation_probability')
            if isinstance(mut_prob, list):
                mut_prob = f"[{', '.join(map(str, mut_prob))}]"
            elif mut_prob is None:
                mut_prob = "None (Auto)"

            stop_crit = config.get('stop_criteria', [])
            if isinstance(stop_crit, list):
                stop_crit = ", ".join(stop_crit)
                
            extra_meta = {
                "Alpha": config.get('alpha', 'N/A'),
                "Beta": config.get('beta', 'N/A'),
                "Lambda": config.get('lambda', 'N/A'),
                "Generations": config.get('num_generations', 'N/A'),
                "Pop Size": config.get('sol_per_pop', 'N/A'),
                "Parents Mating": config.get('num_parents_mating', 'N/A'),
                "K-Tournament": config.get('K_tournament', 'N/A'),
                "Elitism": config.get('keep_elitism', 'N/A'),
                "Mut. Prob": mut_prob,
                "Stop Criteria": stop_crit
            }
                
            labels = []
            fitness_data = []
            profit_data = []
            
            for gen_data in history:
                labels.append(gen_data.get('generation', len(labels) + 1))
                fitness_data.append(gen_data.get('global_best_fitness', 0))
                profit_data.append(gen_data.get('population_avg_fitness', 0))
                
            all_data[file_path.name] = {
                'title': display_title,
                'labels': labels,
                'fitness': fitness_data,
                'profit': profit_data,
                'extra_meta': extra_meta  
            }
            print(f"✅ Đã tải dữ liệu từ {file_path.name}")
            
        except Exception as e:
            print(f"❌ Lỗi khi đọc {file_path.name}: {e}")

    return all_data

def generate_dashboard_html(chart_data):
    cache_folder.mkdir(parents=True, exist_ok=True)
    
    try:
        with open(CSS_PATH, 'r', encoding='utf-8') as f:
            css_code = f.read()
    except FileNotFoundError:
        css_code = "/* CSS File missing */"
        print(f"⚠️ Cảnh báo: Không tìm thấy {CSS_PATH}")
        
    try:
        with open(JS_PATH, 'r', encoding='utf-8') as f:
            js_code = f.read()
    except FileNotFoundError:
        js_code = "// JS File missing"
        print(f"⚠️ Cảnh báo: Không tìm thấy {JS_PATH}")

    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Optimization History Dashboard</title>
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.4/css/all.min.css"/>
        <style>{css_code}</style>
    </head>
    <body>

        <div class="header-panel">
            <h2 style="margin-top:0; color: #2c3e50;">So sánh tiến trình tối ưu</h2>
        </div>

        <div id="chart-grid"></div>

        <div id="modal-overlay">
            <div class="modal-content">
                <button class="close-btn" id="closeModal">&times;</button>
                <h3 class="modal-title" id="modalTitle">Biểu đồ chi tiết</h3>
                
                <div id="modalMeta" class="metadata-grid"></div>
                
                <div class="canvas-container">
                    <canvas id="modalCanvas"></canvas>
                </div>
            </div>
        </div>

        <script>
            window.allChartData = {json.dumps(chart_data)};
            {js_code}
        </script>
    </body>
    </html>
    """
    
    output_path = cache_folder / 'dashboard.html'
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
        
    print(f"\n🎉 Dashboard updated! Open: {output_path.resolve()}")

if __name__ == "__main__":
    selected = choose_solution_files()
    if selected:
        chart_data = load_chart_data(selected)
        if chart_data:
            generate_dashboard_html(chart_data)
        else:
            print("❌ Không có dữ liệu để vẽ biểu đồ.")