const API_BASE = "http://localhost:8000/api";

// --- State ---
let portfolioData = { balance: 0, positions: [], history: [] };
let marketFeed = [];

// --- Init ---
document.addEventListener('DOMContentLoaded', () => {
    initChart();
    fetchPortfolio();
    fetchMarkets();

    // Auto-refresh every 30s
    setInterval(fetchPortfolio, 30000);
});

// --- API Calls ---
async function fetchPortfolio() {
    try {
        const res = await fetch(`${API_BASE}/portfolio`);
        const data = await res.json();

        // Update Balance
        document.getElementById('balance-display').innerText = `$${data.balance.toLocaleString(undefined, { minimumFractionDigits: 2 })}`;
        portfolioData = data;

        renderPositions(data.positions);
        updateChart(data.history); // Mock history for now
    } catch (e) {
        console.error("Portfolio fetch failed", e);
    }
}

async function fetchMarkets() {
    try {
        document.getElementById('markets-feed').innerHTML = '<div class="text-center text-slate-500">Scanning markets...</div>';
        const res = await fetch(`${API_BASE}/markets`);
        const data = await res.json();
        marketFeed = data;
        renderMarkets(data);
    } catch (e) {
        console.error("Market fetch failed", e);
        document.getElementById('markets-feed').innerHTML = '<div class="text-center text-red-400">Failed to load markets</div>';
    }
}

async function executeTrade(marketSlug, outcome, side, price, shares) {
    if (!confirm(`Confirm ${side} ${shares} shares of ${outcome} @ ${price}¢?`)) return;

    try {
        const res = await fetch(`${API_BASE}/trade`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                market_slug: marketSlug,
                outcome: outcome,
                side: side,
                price: price,
                shares: shares
            })
        });

        const result = await res.json();
        if (result.success) {
            alert("Trade Executed! 🚀");
            fetchPortfolio();
        } else {
            alert(`Error: ${result.message}`);
        }
    } catch (e) {
        alert("Trade Failed");
    }
}

// --- Rendering ---
function renderPositions(positions) {
    const tbody = document.getElementById('positions-table');
    tbody.innerHTML = '';

    if (positions.length === 0) {
        tbody.innerHTML = '<tr><td colspan="7" class="text-center py-4 text-slate-500">No active positions. Go trade!</td></tr>';
        return;
    }

    positions.forEach(pos => {
        const pnl = (pos.current_value - pos.avg_price) * pos.shares / 100; // PnL in dollars
        const pnlClass = pnl >= 0 ? 'text-green-400' : 'text-red-400';

        const tr = document.createElement('tr');
        tr.className = 'border-b border-slate-700 hover:bg-slate-800/50';
        tr.innerHTML = `
            <td class="py-3 font-medium text-slate-200">${pos.market_name}</td>
            <td class="py-3"><span class="bg-slate-700 px-2 py-1 rounded text-xs font-mono">${pos.outcome}</span></td>
            <td class="py-3 text-right font-mono">${pos.shares}</td>
            <td class="py-3 text-right font-mono text-slate-400">${pos.avg_price.toFixed(1)}¢</td>
            <td class="py-3 text-right font-mono text-white">${pos.current_value.toFixed(1)}¢</td>
            <td class="py-3 text-right font-mono ${pnlClass}">$${pnl.toFixed(2)}</td>
            <td class="py-3 text-center">
                <button onclick="executeTrade('${pos.market_slug}', '${pos.outcome}', 'SELL', ${pos.current_value}, ${pos.shares})" 
                        class="text-xs bg-red-600/20 hover:bg-red-600/40 text-red-400 border border-red-600/50 px-2 py-1 rounded">
                    Close
                </button>
            </td>
        `;
        tbody.appendChild(tr);
    });
}

function renderMarkets(markets) {
    const feed = document.getElementById('markets-feed');
    feed.innerHTML = '';

    markets.forEach(m => {
        const div = document.createElement('div');
        div.className = 'glass p-4 rounded-lg hover:bg-white/5 transition border-l-4 ' + (m.edge > 5 ? 'border-green-500' : 'border-slate-500');

        div.innerHTML = `
            <div class="flex justify-between items-start mb-2">
                <h3 class="text-sm font-semibold leading-tight">${m.title}</h3>
                <span class="text-xs bg-slate-800 px-1 rounded text-slate-400">${m.days_left}d</span>
            </div>
            <div class="space-y-2">
                ${m.buckets.slice(0, 3).map(b => `
                    <div class="flex justify-between items-center text-sm">
                        <span class="text-slate-400 font-mono text-xs">${b.name}</span>
                        <div class="flex items-center space-x-2">
                            <span class="text-slate-500 text-xs">${b.ask}¢</span>
                            ${b.edge > 5 ? `<span class="text-green-400 text-xs font-bold">+${b.edge.toFixed(1)}%</span>` : ''}
                            <button onclick="executeTrade('${m.slug}', '${b.name}', 'BUY', ${b.ask}, 100)" 
                                    class="bg-blue-600 hover:bg-blue-500 text-xs px-2 rounded py-0.5">
                                Buy
                            </button>
                        </div>
                    </div>
                `).join('')}
            </div>
        `;
        feed.appendChild(div);
    });
}

// --- Chart ---
let chartInstance = null;
function initChart() {
    const ctx = document.getElementById('portfolioChart').getContext('2d');
    chartInstance = new Chart(ctx, {
        type: 'line',
        data: {
            labels: ['Day 1', 'Day 2', 'Day 3', 'Day 4', 'Day 5'],
            datasets: [{
                label: 'Portfolio Value',
                data: [1000, 1020, 1015, 1050, 1080], // Mock data
                borderColor: '#60a5fa', // Blue 400
                backgroundColor: 'rgba(96, 165, 250, 0.1)',
                tension: 0.4,
                fill: true
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { display: false } },
            scales: {
                y: { grid: { color: 'rgba(255,255,255,0.05)' } },
                x: { grid: { display: false } }
            }
        }
    });
}

function updateChart(history) {
    if (!history || history.length === 0) return;
    // Update chart data logic here
}
