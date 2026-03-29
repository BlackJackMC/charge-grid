const allChartData = window.allChartData;

const grid = document.getElementById('chart-grid');
const modalOverlay = document.getElementById('modal-overlay');
const closeModalBtn = document.getElementById('closeModal');
const modalCanvas = document.getElementById('modalCanvas');
const modalTitle = document.getElementById('modalTitle');
let modalChartInstance = null;

const tickFilter75 = function(val, index) {
    const num = parseInt(this.getLabelForValue(val));
    return (num === 1 || num % 75 === 0) ? num : '';
};

const tickFilter25 = function(val, index) {
    const num = parseInt(this.getLabelForValue(val));
    return (num === 1 || num % 25 === 0) ? num : '';
};

Object.keys(allChartData).forEach((fileKey, index) => {
    const dataObj = allChartData[fileKey];
    
    const chartConfig = {
        labels: dataObj.labels,
        datasets: [
            {
                label: 'Fitness (Best/Avg)',
                data: dataObj.fitness,
                borderColor: '#28a745',
                backgroundColor: 'rgba(40, 167, 69, 0.1)',
                borderWidth: 2,
                tension: 0.2,
                pointRadius: 0
            },
            {
                label: 'Avg Profit (E)',
                data: dataObj.profit,
                borderColor: '#007bff',
                borderDash: [5, 5], 
                backgroundColor: 'transparent',
                borderWidth: 2,
                tension: 0.2,
                pointRadius: 0
            }
        ]
    };

    const card = document.createElement('div');
    card.className = 'chart-card';
    card.innerHTML = `
        <h3><i class="fas fa-chart-line"></i> ${dataObj.title}</h3>
        <i class="fas fa-expand expand-icon"></i>
        <div style="position: relative; height: 250px; width: 100%;">
            <canvas id="chart-${index}"></canvas>
        </div>
    `;
    
    card.addEventListener('click', () => openModal(dataObj.title, chartConfig, dataObj.extra_meta));
    grid.appendChild(card);

    const ctx = document.getElementById(`chart-${index}`).getContext('2d');
    new Chart(ctx, {
        type: 'line',
        data: chartConfig,
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { display: false } },
            scales: {
                x: { 
                    display: true,
                    title: { display: true, text: 'Generation', font: {size: 10} },
                    ticks: { 
                        callback: tickFilter75, 
                        autoSkip: false,        
                        maxRotation: 0,
                        font: { size: 9 }
                    },
                    grid: { color: (context) => context.tick && context.tick.label !== '' ? 'rgba(0,0,0,0.1)' : 'transparent' }
                },
                y: { display: true, ticks: { font: { size: 10 } } }
            },
            animation: false
        }
    });
});

function openModal(displayTitle, chartConfig, extraMeta) {
    modalTitle.innerHTML = `<i class="fas fa-search-plus"></i> ${displayTitle}`;
    
    const metaContainer = document.getElementById('modalMeta');
    metaContainer.innerHTML = '';
    if (extraMeta) {
        Object.entries(extraMeta).forEach(([key, value]) => {
            metaContainer.innerHTML += `
                <div class="meta-item">
                    <strong>${key}</strong>
                    <span>${value}</span>
                </div>
            `;
        });
    }

    if (modalChartInstance) {
        modalChartInstance.destroy();
    } 

    const ctx = modalCanvas.getContext('2d');
    modalChartInstance = new Chart(ctx, {
        type: 'line',
        data: chartConfig,
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: { mode: 'index', intersect: false },
            plugins: {
                legend: { display: true, position: 'top' },
                tooltip: {
                    backgroundColor: 'rgba(255, 255, 255, 0.9)',
                    titleColor: '#333',
                    bodyColor: '#333',
                    borderColor: '#ddd',
                    borderWidth: 1
                }
            },
            scales: {
                x: { 
                    title: { display: true, text: 'Generation' },
                    ticks: { callback: tickFilter25, autoSkip: false },
                    grid: { color: (context) => context.tick && context.tick.label !== '' ? 'rgba(0,0,0,0.1)' : 'transparent' }
                },
                y: { title: { display: true, text: 'Value' } }
            }
        }
    });

    modalOverlay.classList.add('active');
}

closeModalBtn.addEventListener('click', () => modalOverlay.classList.remove('active'));
modalOverlay.addEventListener('click', (e) => {
    if(e.target === modalOverlay) modalOverlay.classList.remove('active');
});