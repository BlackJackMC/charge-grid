import json
import re
from pathlib import Path

from charge_grid.utils import OUTPUT_DIR

output_folder = OUTPUT_DIR
cache_folder = Path('../cache')

def choose_solution_files():
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
                
            # ĐỌC KEY THEO FORMAT JSON MỚI
            history = data.get('generation_history', [])
            if not history:
                print(f"⚠️ Bỏ qua {file_path.name}: Không tìm thấy dữ liệu 'generation_history'.")
                continue
            
            metadata = data.get('metadata', {})
            config = metadata.get('configuration', {})
            
            # XỬ LÝ MUTATION VÀ CROSSOVER (Hỗ trợ mảng cho mutation)
            mut_val = config.get('mutation_probability', config.get('mutation_type', 'N/A'))
            if isinstance(mut_val, list):
                mut_val = f"[{', '.join(map(str, mut_val))}]"
                
            cross_val = config.get('crossover_type', 'N/A')
            
            # Đặt tên title chứa cấu hình
            display_title = f"{file_path.name} (Mut: {mut_val} | Cross: {cross_val})"
                
            labels = []
            fitness_data = []
            profit_data = []
            
            for gen_data in history:
                labels.append(gen_data.get('generation', len(labels) + 1))
                # Lấy best_average_fitness làm đường chính
                fitness_data.append(gen_data.get('best_average_fitness', 0))
                # Lấy avg_E_profit làm đường phụ (nếu có)
                profit_data.append(gen_data.get('avg_E_profit', 0))
                
            all_data[file_path.name] = {
                'title': display_title,
                'labels': labels,
                'fitness': fitness_data,
                'profit': profit_data
            }
            print(f"✅ Đã tải dữ liệu từ {file_path.name}")
            
        except Exception as e:
            print(f"❌ Lỗi khi đọc {file_path.name}: {e}")

    return all_data

def generate_dashboard_html(chart_data):
    cache_folder.mkdir(parents=True, exist_ok=True)
    
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Optimization History Dashboard</title>
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.4/css/all.min.css"/>
        <style>
            body {{
                font-family: 'Segoe UI', Arial, sans-serif;
                background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
                min-height: 100vh;
                margin: 0;
                padding: 20px;
                color: #333;
            }}
            .header-panel {{
                background: rgba(255, 255, 255, 0.85);
                backdrop-filter: blur(10px);
                border-radius: 12px;
                padding: 20px;
                box-shadow: 0 6px 20px rgba(0,0,0,0.08);
                border: 1px solid rgba(255,255,255,0.6);
                margin-bottom: 20px;
                text-align: center;
            }}
            #chart-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fill, minmax(400px, 1fr));
                gap: 20px;
            }}
            .chart-card {{
                background: rgba(255, 255, 255, 0.95);
                backdrop-filter: blur(10px);
                border-radius: 12px;
                padding: 15px;
                box-shadow: 0 4px 15px rgba(0,0,0,0.1);
                border: 1px solid rgba(255,255,255,0.6);
                cursor: pointer;
                transition: transform 0.2s, box-shadow 0.2s;
                position: relative;
            }}
            .chart-card:hover {{
                transform: translateY(-5px);
                box-shadow: 0 8px 25px rgba(0,0,0,0.15);
            }}
            .chart-card h3 {{
                margin: 0 0 10px 0;
                font-size: 14px;
                color: #17a2b8;
                border-bottom: 1px solid #eee;
                padding-bottom: 8px;
            }}
            .expand-icon {{
                position: absolute;
                top: 15px;
                right: 15px;
                color: #888;
                font-size: 14px;
            }}
            #modal-overlay {{
                position: fixed; top: 0; left: 0; width: 100%; height: 100%;
                background: rgba(0, 0, 0, 0.6);
                backdrop-filter: blur(5px);
                display: flex; align-items: center; justify-content: center;
                z-index: 9999; opacity: 0; pointer-events: none;
                transition: opacity 0.3s ease;
            }}
            #modal-overlay.active {{ opacity: 1; pointer-events: auto; }}
            .modal-content {{
                background: #fff; width: 85%; max-width: 1200px;
                border-radius: 12px; padding: 20px;
                box-shadow: 0 15px 40px rgba(0,0,0,0.3);
                position: relative; transform: scale(0.9);
                transition: transform 0.3s ease;
            }}
            #modal-overlay.active .modal-content {{ transform: scale(1); }}
            .close-btn {{
                position: absolute; top: 15px; right: 20px; background: none;
                border: none; font-size: 28px; color: #aaa; cursor: pointer;
            }}
            .close-btn:hover {{ color: #dc3545; }}
            .modal-title {{
                margin-top: 0; color: #117a8b; font-size: 20px;
                border-bottom: 2px solid #eee; padding-bottom: 10px; margin-bottom: 20px;
            }}
            .canvas-container {{ position: relative; height: 60vh; width: 100%; }}
        </style>
    </head>
    <body>

        <div class="header-panel">
            <h2 style="margin-top:0; color: #2c3e50;"><i class="fas fa-project-diagram"></i> So sánh tiến trình tối ưu</h2>
            <p style="color: #666; margin: 0; font-size: 14px;">Click vào biểu đồ bất kỳ để phóng to</p>
        </div>

        <div id="chart-grid"></div>

        <div id="modal-overlay">
            <div class="modal-content">
                <button class="close-btn" id="closeModal">&times;</button>
                <h3 class="modal-title" id="modalTitle">Biểu đồ chi tiết</h3>
                <div class="canvas-container">
                    <canvas id="modalCanvas"></canvas>
                </div>
            </div>
        </div>

        <script>
            const allChartData = {json.dumps(chart_data)};
            
            const grid = document.getElementById('chart-grid');
            const modalOverlay = document.getElementById('modal-overlay');
            const closeModalBtn = document.getElementById('closeModal');
            const modalCanvas = document.getElementById('modalCanvas');
            const modalTitle = document.getElementById('modalTitle');
            let modalChartInstance = null;

            Object.keys(allChartData).forEach((fileKey, index) => {{
                const dataObj = allChartData[fileKey];
                
                const chartConfig = {{
                    labels: dataObj.labels,
                    datasets: [
                        {{
                            label: 'Fitness (Best/Avg)',
                            data: dataObj.fitness,
                            borderColor: '#28a745',
                            backgroundColor: 'rgba(40, 167, 69, 0.1)',
                            borderWidth: 2,
                            tension: 0.2,
                            pointRadius: 0
                        }},
                        {{
                            label: 'Avg Profit (E)',
                            data: dataObj.profit,
                            borderColor: '#007bff',
                            borderDash: [5, 5], 
                            backgroundColor: 'transparent',
                            borderWidth: 2,
                            tension: 0.2,
                            pointRadius: 0
                        }}
                    ]
                }};

                const card = document.createElement('div');
                card.className = 'chart-card';
                card.innerHTML = `
                    <h3><i class="fas fa-chart-line"></i> ${{dataObj.title}}</h3>
                    <i class="fas fa-expand expand-icon"></i>
                    <div style="position: relative; height: 250px; width: 100%;">
                        <canvas id="chart-${{index}}"></canvas>
                    </div>
                `;
                
                card.addEventListener('click', () => openModal(dataObj.title, chartConfig));
                grid.appendChild(card);

                const ctx = document.getElementById(`chart-${{index}}`).getContext('2d');
                new Chart(ctx, {{
                    type: 'line',
                    data: chartConfig,
                    options: {{
                        responsive: true,
                        maintainAspectRatio: false,
                        plugins: {{ legend: {{ display: false }} }},
                        scales: {{
                            x: {{ display: false }},
                            y: {{ display: true, ticks: {{ font: {{ size: 10 }} }} }}
                        }},
                        animation: false
                    }}
                }});
            }});

            function openModal(displayTitle, chartConfig) {{
                modalTitle.innerHTML = `<i class="fas fa-search-plus"></i> ${{displayTitle}}`;
                
                if (modalChartInstance) {{
                    modalChartInstance.destroy();
                }}

                const ctx = modalCanvas.getContext('2d');
                modalChartInstance = new Chart(ctx, {{
                    type: 'line',
                    data: chartConfig,
                    options: {{
                        responsive: true,
                        maintainAspectRatio: false,
                        interaction: {{ mode: 'index', intersect: false }},
                        plugins: {{
                            legend: {{ display: true, position: 'top' }},
                            tooltip: {{
                                backgroundColor: 'rgba(255, 255, 255, 0.9)',
                                titleColor: '#333',
                                bodyColor: '#333',
                                borderColor: '#ddd',
                                borderWidth: 1
                            }}
                        }},
                        scales: {{
                            x: {{ title: {{ display: true, text: 'Generation' }} }},
                            y: {{ title: {{ display: true, text: 'Value' }} }}
                        }}
                    }}
                }});

                modalOverlay.classList.add('active');
            }}

            closeModalBtn.addEventListener('click', () => modalOverlay.classList.remove('active'));
            modalOverlay.addEventListener('click', (e) => {{
                if(e.target === modalOverlay) modalOverlay.classList.remove('active');
            }});
        </script>
    </body>
    </html>
    """
    
    output_path = cache_folder / 'dashboard.html'
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
        
    print(f"\n🎉 Tạo Dashboard thành công! Mở file này trên trình duyệt: {output_path.resolve()}")

if __name__ == "__main__":
    selected_files = choose_solution_files()
    if selected_files:
        print("\n--- Đang phân tích dữ liệu JSON... ---")
        data = load_chart_data(selected_files)
        if data:
            generate_dashboard_html(data)
        else:
            print("❌ Không có dữ liệu để tạo biểu đồ.")