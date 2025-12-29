// Global state
let envInfo = null;
let currentPolicy = null;
let currentValues = null;
let trainHistory = null;

// DOM elements
const envSelect = document.getElementById('env-select');
const algoSelect = document.getElementById('algo-select');
const initEnvBtn = document.getElementById('init-env-btn');
const trainBtn = document.getElementById('train-btn');
const runBtn = document.getElementById('run-btn');
const resetBtn = document.getElementById('reset-btn');

const gammaSlider = document.getElementById('gamma');
const alphaSlider = document.getElementById('alpha');
const epsilonSlider = document.getElementById('epsilon');
const episodesSlider = document.getElementById('episodes');
const nStepSlider = document.getElementById('n-step');

const envCanvas = document.getElementById('env-canvas');
const envCtx = envCanvas.getContext('2d');
const valueCanvas = document.getElementById('value-canvas');
const valueCtx = valueCanvas.getContext('2d');
const chartCanvas = document.getElementById('chart-canvas');
const chartCtx = chartCanvas.getContext('2d');

// update parameter displays
gammaSlider.oninput = () => document.getElementById('gamma-val').textContent = gammaSlider.value;
alphaSlider.oninput = () => document.getElementById('alpha-val').textContent = alphaSlider.value;
epsilonSlider.oninput = () => document.getElementById('epsilon-val').textContent = epsilonSlider.value;
episodesSlider.oninput = () => document.getElementById('episodes-val').textContent = episodesSlider.value;
nStepSlider.oninput = () => document.getElementById('n-step-val').textContent = nStepSlider.value;

// algorithm descriptions
const algoDescriptions = {
    'policy_iteration': 'Iteratively evaluates current policy then improves it. Guaranteed to converge to optimal policy.',
    'value_iteration': 'Directly computes optimal value function by taking max over actions at each state.',
    'monte_carlo': 'Learns from complete episodes. Updates Q-values based on actual returns.',
    'td': 'TD(0) - Updates value estimates after each step using bootstrapping.',
    'n_step_td': 'N-step TD - Uses n-step returns for value updates. Trades off between MC and TD(0).',
    'sarsa': 'On-policy TD control. Updates Q(s,a) using the action actually taken in next state.',
    'q_learning': 'Off-policy TD control. Updates Q(s,a) using max action value in next state.'
};

// algorithm-level parameter defaults (simplified)
const algorithmDefaults = {
    'policy_iteration': {gamma: 0.99, alpha: 0.1, epsilon: 0.1, episodes: 100, n_step: 3},
    'value_iteration': {gamma: 0.99, alpha: 0.1, epsilon: 0.1, episodes: 100, n_step: 3},
    'monte_carlo': {gamma: 0.99, alpha: 0.1, epsilon: 0.2, episodes: 1000, n_step: 3},
    'td': {gamma: 0.99, alpha: 0.5, epsilon: 0.2, episodes: 1500, n_step: 3},
    'n_step_td': {gamma: 0.99, alpha: 0.5, epsilon: 0.2, episodes: 1500, n_step: 5},
    'sarsa': {gamma: 0.99, alpha: 0.1, epsilon: 0.2, episodes: 1000, n_step: 3},
    'q_learning': {gamma: 0.99, alpha: 0.1, epsilon: 0.2, episodes: 1000, n_step: 3}
};

// load parameters for current algorithm
function loadParameterPreset() {
    const algo = algoSelect.value;

    if (algorithmDefaults[algo]) {
        const preset = algorithmDefaults[algo];

        gammaSlider.value = preset.gamma;
        alphaSlider.value = preset.alpha;
        epsilonSlider.value = preset.epsilon;
        episodesSlider.value = preset.episodes;
        nStepSlider.value = preset.n_step;

        document.getElementById('gamma-val').textContent = preset.gamma;
        document.getElementById('alpha-val').textContent = preset.alpha;
        document.getElementById('epsilon-val').textContent = preset.epsilon;
        document.getElementById('episodes-val').textContent = preset.episodes;
        document.getElementById('n-step-val').textContent = preset.n_step;
    }
}

// update algorithm description when changed
algoSelect.onchange = () => {
    const desc = algoDescriptions[algoSelect.value] || '';
    document.getElementById('algo-description').textContent = desc;
    loadParameterPreset();
};

// environment tooltips
const envTooltips = {
    'gridworld': 'Navigate from start to goal in a grid. Avoid obstacles.',
    'frozenlake': 'Walk across frozen lake to goal. Ice is slippery! Avoid holes.',
    'cliffwalking': 'Walk along cliff edge to goal. Falling off cliff gives big penalty.',
    'mountaincar': 'Drive car up the hill. Build momentum to reach the top.',
    'cartpole': 'Balance a pole on a moving cart. Dont let it fall!'
};

envSelect.onchange = () => {
    const tip = envTooltips[envSelect.value] || '';
    document.getElementById('env-tooltip').textContent = tip;
};

// Initialize environment
initEnvBtn.onclick = async () => {
    const envName = envSelect.value;

    try {
        const response = await fetch('/api/init_env', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ env: envName, params: {} })
        });

        envInfo = await response.json();

        if (envInfo.error) {
            alert('Error initializing environment: ' + envInfo.error);
            console.error('Full error:', envInfo);
            return;
        }

        currentPolicy = null;
        currentValues = null;

        document.getElementById('current-state').textContent = JSON.stringify(envInfo.state);
        document.getElementById('current-reward').textContent = '-';
        document.getElementById('is-done').textContent = 'false';

        drawEnvironment();
        clearValueCanvas();
        clearChart();
        document.getElementById('policy-display').textContent = 'Train to see policy';

    } catch (err) {
        console.error('Error initializing env:', err);
        alert('Error initializing environment: ' + err.message);
    }
};

// Train agent
trainBtn.onclick = async () => {
    if (!envInfo) {
        alert('Initialize environment first!');
        return;
    }

    const algo = algoSelect.value;
    const params = {
        gamma: parseFloat(gammaSlider.value),
        alpha: parseFloat(alphaSlider.value),
        epsilon: parseFloat(epsilonSlider.value),
        n_episodes: parseInt(episodesSlider.value),
        n_step: parseInt(nStepSlider.value),
        theta: 1e-6
    };

    // Show training status
    const trainingStatus = document.getElementById('training-status');
    const statusMessage = trainingStatus.querySelector('.status-message');
    const progressBar = trainingStatus.querySelector('.progress-bar');
    const progressText = trainingStatus.querySelector('.progress-text');
    const currentEpisode = document.getElementById('current-episode');
    const totalEpisodes = document.getElementById('total-episodes');
    const convergenceVal = document.getElementById('convergence-val');

    trainingStatus.style.display = 'block';
    statusMessage.textContent = `Training ${algo.replace('_', ' ')}...`;
    progressBar.style.width = '0%';
    progressText.textContent = '0%';
    currentEpisode.textContent = '0';
    totalEpisodes.textContent = params.n_episodes;
    convergenceVal.textContent = 'initializing...';

    // Disable controls
    trainBtn.disabled = true;
    runBtn.disabled = true;
    resetBtn.disabled = true;
    initEnvBtn.disabled = true;
    trainBtn.textContent = 'Training...';

    const startTime = Date.now();

    // Simulate progress (since backend is synchronous)
    const progressInterval = setInterval(() => {
        const elapsed = Date.now() - startTime;
        const estimatedTotal = 5000; // rough estimate
        const progress = Math.min(95, (elapsed / estimatedTotal) * 100);
        progressBar.style.width = progress + '%';
        progressText.textContent = Math.floor(progress) + '%';

        const estimatedEpisode = Math.floor((progress / 100) * params.n_episodes);
        currentEpisode.textContent = estimatedEpisode;
        convergenceVal.textContent = 'computing...';
    }, 100);

    try {
        const response = await fetch('/api/train', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ algorithm: algo, params: params })
        });

        const result = await response.json();
        const endTime = Date.now();

        clearInterval(progressInterval);

        // Complete progress
        progressBar.style.width = '100%';
        progressText.textContent = '100%';
        currentEpisode.textContent = params.n_episodes;

        currentPolicy = result.policy;
        currentValues = result.values;
        trainHistory = result.history;

        // Calculate final convergence
        if (trainHistory.length > 0) {
            const lastVal = trainHistory[trainHistory.length - 1];
            convergenceVal.textContent = lastVal.toFixed(4);
        }

        // update stats
        document.getElementById('train-time').textContent = ((endTime - startTime) / 1000).toFixed(2) + 's';
        document.getElementById('iterations').textContent = trainHistory.length;

        if (trainHistory.length > 0) {
            const lastVals = trainHistory.slice(-10);
            const avg = lastVals.reduce((a, b) => a + b, 0) / lastVals.length;
            document.getElementById('avg-reward').textContent = avg.toFixed(2);
        }

        drawValueFunction();
        drawChart();
        displayPolicy();

        statusMessage.textContent = 'Training Complete!';
        statusMessage.style.color = '#4caf50';

        // Hide status after 3 seconds
        setTimeout(() => {
            trainingStatus.style.display = 'none';
            statusMessage.style.color = '#667eea';
        }, 3000);

    } catch (err) {
        console.error('Error training:', err);
        clearInterval(progressInterval);
        statusMessage.textContent = 'Training Failed!';
        statusMessage.style.color = '#f44336';
    }

    trainBtn.disabled = false;
    runBtn.disabled = false;
    resetBtn.disabled = false;
    initEnvBtn.disabled = false;
    trainBtn.textContent = 'Train Agent';
};

// Run episode with learned policy
runBtn.onclick = async () => {
    if (!currentPolicy) {
        alert('Train agent first!');
        return;
    }

    try {
        const response = await fetch('/api/run_episode', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ policy: currentPolicy })
        });

        const result = await response.json();

        // animate trajectory
        animateTrajectory(result.trajectory);

    } catch (err) {
        console.error('Error running episode:', err);
    }
};

// Reset environment
resetBtn.onclick = async () => {
    if (!envInfo) return;

    try {
        const response = await fetch('/api/reset', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' }
        });

        const result = await response.json();
        envInfo.state = result.state;

        document.getElementById('current-state').textContent = JSON.stringify(result.state);
        document.getElementById('current-reward').textContent = '-';
        document.getElementById('is-done').textContent = 'false';

        drawEnvironment();

    } catch (err) {
        console.error('Error resetting:', err);
    }
};

// Manual control
async function takeAction(action) {
    if (!envInfo) return;

    try {
        const response = await fetch('/api/step', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ action: action })
        });

        const result = await response.json();
        envInfo.state = result.state;

        document.getElementById('current-state').textContent = JSON.stringify(result.state);
        document.getElementById('current-reward').textContent = result.reward.toFixed(2);
        document.getElementById('is-done').textContent = result.done.toString();

        drawEnvironment();

        if (result.done) {
            setTimeout(() => {
                resetBtn.click();
            }, 1000);
        }

    } catch (err) {
        console.error('Error stepping:', err);
    }
}

document.getElementById('up-btn').onclick = () => takeAction(0);
document.getElementById('down-btn').onclick = () => takeAction(1);
document.getElementById('left-btn').onclick = () => takeAction(2);
document.getElementById('right-btn').onclick = () => takeAction(3);

// keyboard controls
document.addEventListener('keydown', (e) => {
    if (e.key === 'ArrowUp') takeAction(0);
    else if (e.key === 'ArrowDown') takeAction(1);
    else if (e.key === 'ArrowLeft') takeAction(2);
    else if (e.key === 'ArrowRight') takeAction(3);
});

// Drawing functions
function drawEnvironment() {
    if (!envInfo) return;

    const canvas = envCanvas;
    const ctx = envCtx;
    const width = canvas.width;
    const height = canvas.height;

    ctx.clearRect(0, 0, width, height);

    const envName = envSelect.value;

    if (envName === 'gridworld' || envName === 'frozenlake') {
        drawGridEnv(ctx, width, height);
    } else if (envName === 'cliffwalking') {
        drawCliffEnv(ctx, width, height);
    } else if (envName === 'mountaincar') {
        drawMountainCar(ctx, width, height);
    } else if (envName === 'cartpole') {
        drawCartPole(ctx, width, height);
    }
}

function drawGridEnv(ctx, width, height) {
    const size = envInfo.grid_size || 5;
    const cellW = width / size;
    const cellH = height / size;

    // draw grid
    for (let r = 0; r < size; r++) {
        for (let c = 0; c < size; c++) {
            const x = c * cellW;
            const y = r * cellH;

            ctx.fillStyle = '#ffffff';
            ctx.fillRect(x, y, cellW, cellH);
            ctx.strokeStyle = '#ccc';
            ctx.strokeRect(x, y, cellW, cellH);
        }
    }

    // draw obstacles
    if (envInfo.obstacles) {
        ctx.fillStyle = '#333';
        for (const obs of envInfo.obstacles) {
            ctx.fillRect(obs[1] * cellW, obs[0] * cellH, cellW, cellH);
        }
    }

    // draw holes for frozenlake
    if (envInfo.holes) {
        ctx.fillStyle = '#4a90d9';
        for (const hole of envInfo.holes) {
            ctx.fillRect(hole[1] * cellW, hole[0] * cellH, cellW, cellH);
            ctx.fillStyle = '#fff';
            ctx.font = '20px Arial';
            ctx.fillText('H', hole[1] * cellW + cellW/3, hole[0] * cellH + cellH/1.5);
            ctx.fillStyle = '#4a90d9';
        }
    }

    // draw goal
    if (envInfo.goal) {
        ctx.fillStyle = '#4caf50';
        ctx.fillRect(envInfo.goal[1] * cellW, envInfo.goal[0] * cellH, cellW, cellH);
        ctx.fillStyle = '#fff';
        ctx.font = '20px Arial';
        ctx.fillText('G', envInfo.goal[1] * cellW + cellW/3, envInfo.goal[0] * cellH + cellH/1.5);
    }

    // draw agent
    if (envInfo.state) {
        const agentR = envInfo.state[0];
        const agentC = envInfo.state[1];
        ctx.fillStyle = '#f44336';
        ctx.beginPath();
        ctx.arc(agentC * cellW + cellW/2, agentR * cellH + cellH/2, cellW/3, 0, Math.PI * 2);
        ctx.fill();
    }
}

function drawCliffEnv(ctx, width, height) {
    const h = envInfo.grid_height || 4;
    const w = envInfo.grid_width || 12;
    const cellW = width / w;
    const cellH = height / h;

    // draw grid
    for (let r = 0; r < h; r++) {
        for (let c = 0; c < w; c++) {
            const x = c * cellW;
            const y = r * cellH;

            ctx.fillStyle = '#ffffff';
            ctx.fillRect(x, y, cellW, cellH);
            ctx.strokeStyle = '#ccc';
            ctx.strokeRect(x, y, cellW, cellH);
        }
    }

    // draw cliff
    if (envInfo.cliff) {
        ctx.fillStyle = '#8b4513';
        for (const cliff of envInfo.cliff) {
            ctx.fillRect(cliff[1] * cellW, cliff[0] * cellH, cellW, cellH);
        }
    }

    // draw start
    if (envInfo.start) {
        ctx.fillStyle = '#2196f3';
        ctx.fillRect(envInfo.start[1] * cellW, envInfo.start[0] * cellH, cellW, cellH);
        ctx.fillStyle = '#fff';
        ctx.font = '12px Arial';
        ctx.fillText('S', envInfo.start[1] * cellW + cellW/3, envInfo.start[0] * cellH + cellH/1.5);
    }

    // draw goal
    if (envInfo.goal) {
        ctx.fillStyle = '#4caf50';
        ctx.fillRect(envInfo.goal[1] * cellW, envInfo.goal[0] * cellH, cellW, cellH);
        ctx.fillStyle = '#fff';
        ctx.font = '12px Arial';
        ctx.fillText('G', envInfo.goal[1] * cellW + cellW/3, envInfo.goal[0] * cellH + cellH/1.5);
    }

    // draw agent
    if (envInfo.state) {
        const agentR = envInfo.state[0];
        const agentC = envInfo.state[1];
        ctx.fillStyle = '#f44336';
        ctx.beginPath();
        ctx.arc(agentC * cellW + cellW/2, agentR * cellH + cellH/2, cellW/3, 0, Math.PI * 2);
        ctx.fill();
    }
}

function drawMountainCar(ctx, width, height) {
    ctx.fillStyle = '#87ceeb';
    ctx.fillRect(0, 0, width, height);

    // simple mountain
    ctx.beginPath();
    ctx.moveTo(0, height);
    for (let x = 0; x <= width; x += 5) {
        const pos = -1.2 + (x / width) * 1.8;
        const y = height - (Math.sin(3 * pos) * 0.45 + 0.55) * height * 0.8;
        ctx.lineTo(x, y);
    }
    ctx.lineTo(width, height);
    ctx.closePath();
    ctx.fillStyle = '#228b22';
    ctx.fill();

    // goal line
    const goalX = (0.5 - (-1.2)) / 1.8 * width;
    ctx.strokeStyle = '#ffd700';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(goalX, 0);
    ctx.lineTo(goalX, height);
    ctx.stroke();

    // car
    if (envInfo.state) {
        const posIdx = Array.isArray(envInfo.state) ? envInfo.state[0] : 10;
        const pos = -1.2 + (posIdx / 19) * 1.8;
        const carX = (pos - (-1.2)) / 1.8 * width;
        const carY = height - (Math.sin(3 * pos) * 0.45 + 0.55) * height * 0.8;

        ctx.fillStyle = '#f44336';
        ctx.fillRect(carX - 10, carY - 8, 20, 12);
    }
}

function drawCartPole(ctx, width, height) {
    ctx.fillStyle = '#f0f0f0';
    ctx.fillRect(0, 0, width, height);

    // ground
    ctx.fillStyle = '#8b4513';
    ctx.fillRect(0, height * 0.8, width, height * 0.2);

    // cart and pole
    if (envInfo.state) {
        const posIdx = envInfo.state[0];
        const angIdx = envInfo.state[2];

        const cartX = width/2 + (posIdx - 5) * 30;
        const cartY = height * 0.75;

        // cart
        ctx.fillStyle = '#333';
        ctx.fillRect(cartX - 25, cartY, 50, 15);

        // pole
        const angle = (angIdx - 5) * 0.04;
        const poleLen = 80;
        const poleEndX = cartX + Math.sin(angle) * poleLen;
        const poleEndY = cartY - Math.cos(angle) * poleLen;

        ctx.strokeStyle = '#d2691e';
        ctx.lineWidth = 6;
        ctx.beginPath();
        ctx.moveTo(cartX, cartY);
        ctx.lineTo(poleEndX, poleEndY);
        ctx.stroke();
    }
}

function drawValueFunction() {
    if (!currentValues || !envInfo) return;

    const ctx = valueCtx;
    const width = valueCanvas.width;
    const height = valueCanvas.height;

    ctx.clearRect(0, 0, width, height);

    const envName = envSelect.value;

    if (envName === 'gridworld' || envName === 'frozenlake') {
        drawGridValues(ctx, width, height, envInfo.grid_size || 5);
    } else if (envName === 'cliffwalking') {
        drawCliffValues(ctx, width, height);
    } else {
        // for other envs just show text
        ctx.fillStyle = '#333';
        ctx.font = '14px Arial';
        ctx.fillText('Value function visualization', 10, 30);
        ctx.fillText('not available for this env', 10, 50);

        // show some stats
        const minV = Math.min(...currentValues);
        const maxV = Math.max(...currentValues);
        const avgV = currentValues.reduce((a, b) => a + b, 0) / currentValues.length;
        ctx.fillText('Min: ' + minV.toFixed(2), 10, 80);
        ctx.fillText('Max: ' + maxV.toFixed(2), 10, 100);
        ctx.fillText('Avg: ' + avgV.toFixed(2), 10, 120);
    }
}

function drawGridValues(ctx, width, height, size) {
    const cellW = width / size;
    const cellH = height / size;

    const minV = Math.min(...currentValues);
    const maxV = Math.max(...currentValues);
    const range = maxV - minV || 1;

    for (let r = 0; r < size; r++) {
        for (let c = 0; c < size; c++) {
            const idx = r * size + c;
            const val = currentValues[idx];
            const norm = (val - minV) / range;

            // color from red to green
            const red = Math.floor((1 - norm) * 255);
            const green = Math.floor(norm * 255);
            ctx.fillStyle = `rgb(${red}, ${green}, 100)`;

            ctx.fillRect(c * cellW, r * cellH, cellW, cellH);
            ctx.strokeStyle = '#ccc';
            ctx.strokeRect(c * cellW, r * cellH, cellW, cellH);

            // show value
            ctx.fillStyle = '#000';
            ctx.font = '12px Arial';
            ctx.fillText(val.toFixed(1), c * cellW + 5, r * cellH + cellH/2);
        }
    }
}

function drawCliffValues(ctx, width, height) {
    const h = envInfo.grid_height || 4;
    const w = envInfo.grid_width || 12;
    const cellW = width / w;
    const cellH = height / h;

    const minV = Math.min(...currentValues);
    const maxV = Math.max(...currentValues);
    const range = maxV - minV || 1;

    for (let r = 0; r < h; r++) {
        for (let c = 0; c < w; c++) {
            const idx = r * w + c;
            const val = currentValues[idx];
            const norm = (val - minV) / range;

            const red = Math.floor((1 - norm) * 255);
            const green = Math.floor(norm * 255);
            ctx.fillStyle = `rgb(${red}, ${green}, 100)`;

            ctx.fillRect(c * cellW, r * cellH, cellW, cellH);
            ctx.strokeStyle = '#ccc';
            ctx.strokeRect(c * cellW, r * cellH, cellW, cellH);

            ctx.fillStyle = '#000';
            ctx.font = '10px Arial';
            ctx.fillText(val.toFixed(0), c * cellW + 2, r * cellH + cellH/2);
        }
    }
}

function clearValueCanvas() {
    const ctx = valueCtx;
    ctx.clearRect(0, 0, valueCanvas.width, valueCanvas.height);
    ctx.fillStyle = '#eee';
    ctx.fillRect(0, 0, valueCanvas.width, valueCanvas.height);
    ctx.fillStyle = '#999';
    ctx.font = '14px Arial';
    ctx.fillText('Train to see value function', 120, 200);
}

function drawChart() {
    if (!trainHistory || trainHistory.length === 0) return;

    const ctx = chartCtx;
    const width = chartCanvas.width;
    const height = chartCanvas.height;

    ctx.clearRect(0, 0, width, height);

    const padding = 40;
    const chartW = width - padding * 2;
    const chartH = height - padding * 2;

    // axes
    ctx.strokeStyle = '#333';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(padding, padding);
    ctx.lineTo(padding, height - padding);
    ctx.lineTo(width - padding, height - padding);
    ctx.stroke();

    // plot data
    const data = trainHistory;
    const n = data.length;
    const minY = Math.min(...data);
    const maxY = Math.max(...data);
    const rangeY = maxY - minY || 1;

    ctx.strokeStyle = '#667eea';
    ctx.lineWidth = 2;
    ctx.beginPath();

    for (let i = 0; i < n; i++) {
        const x = padding + (i / (n - 1)) * chartW;
        const y = height - padding - ((data[i] - minY) / rangeY) * chartH;

        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
    }
    ctx.stroke();

    // labels
    ctx.fillStyle = '#333';
    ctx.font = '12px Arial';
    ctx.fillText('Iterations', width/2 - 30, height - 5);
    ctx.save();
    ctx.translate(12, height/2);
    ctx.rotate(-Math.PI/2);
    ctx.fillText('Value', 0, 0);
    ctx.restore();
}

function clearChart() {
    const ctx = chartCtx;
    ctx.clearRect(0, 0, chartCanvas.width, chartCanvas.height);
    ctx.fillStyle = '#eee';
    ctx.fillRect(0, 0, chartCanvas.width, chartCanvas.height);
    ctx.fillStyle = '#999';
    ctx.font = '14px Arial';
    ctx.fillText('Train to see progress', 100, 100);
}

function displayPolicy() {
    if (!currentPolicy) return;

    const actions = ['↑', '↓', '←', '→', 'P', 'D'];
    const policyDiv = document.getElementById('policy-display');

    const envName = envSelect.value;

    if (envName === 'gridworld' || envName === 'frozenlake') {
        const size = envInfo.grid_size || 5;
        let html = '<table style="border-collapse: collapse;">';

        for (let r = 0; r < size; r++) {
            html += '<tr>';
            for (let c = 0; c < size; c++) {
                const idx = r * size + c;
                const action = currentPolicy[idx];
                html += `<td style="border:1px solid #ccc; width:30px; height:30px; text-align:center;">${actions[action]}</td>`;
            }
            html += '</tr>';
        }
        html += '</table>';
        policyDiv.innerHTML = html;
    } else if (envName === 'cliffwalking') {
        const h = envInfo.grid_height || 4;
        const w = envInfo.grid_width || 12;
        let html = '<table style="border-collapse: collapse;">';

        for (let r = 0; r < h; r++) {
            html += '<tr>';
            for (let c = 0; c < w; c++) {
                const idx = r * w + c;
                const action = currentPolicy[idx];
                html += `<td style="border:1px solid #ccc; width:25px; height:25px; text-align:center; font-size:12px;">${actions[action]}</td>`;
            }
            html += '</tr>';
        }
        html += '</table>';
        policyDiv.innerHTML = html;
    } else {
        policyDiv.textContent = 'Policy: ' + currentPolicy.length + ' state-action pairs';
    }
}

async function animateTrajectory(trajectory) {
    const episodeInfo = document.getElementById('episode-info');
    episodeInfo.style.display = 'block';

    const actionNames = ['↑ Up', '↓ Down', '← Left', '→ Right'];
    let cumulativeReward = 0;

    runBtn.disabled = true;
    runBtn.textContent = 'Playing...';

    for (let i = 0; i < trajectory.length; i++) {
        const step = trajectory[i];
        envInfo.state = step.state;
        cumulativeReward += step.reward;

        document.getElementById('episode-step').textContent = i + 1;
        document.getElementById('episode-action').textContent = actionNames[step.action] || step.action;
        document.getElementById('episode-total-reward').textContent = cumulativeReward.toFixed(2);

        document.getElementById('current-state').textContent = JSON.stringify(step.state);
        document.getElementById('current-reward').textContent = step.reward.toFixed(2);
        document.getElementById('is-done').textContent = step.done.toString();

        drawEnvironment();
        await sleep(200);
    }

    runBtn.disabled = false;
    runBtn.textContent = 'Run Episode';

    setTimeout(() => {
        episodeInfo.style.display = 'none';
    }, 2000);
}


function sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}


const logPanel = document.getElementById('log-panel');

function addLogEntry(timestamp, type, details) {
    const entry = document.createElement('div');
    entry.className = 'log-entry';

    const time = timestamp.split(' ')[1];
    const detailsStr = typeof details === 'object' ? JSON.stringify(details) : details;

    entry.textContent = `${time} ${type}: ${detailsStr}`;
    logPanel.appendChild(entry);
    logPanel.scrollTop = logPanel.scrollHeight;
}

async function fetchLogs() {
    try {
        const response = await fetch('/api/get_logs');
        const data = await response.json();

        logPanel.innerHTML = '';
        data.logs.forEach(log => {
            addLogEntry(log.timestamp, log.type, log.details);
        });
    } catch (error) {
        console.error('Error fetching logs:', error);
    }
}

setInterval(fetchLogs, 1000);

// initialize on page load
window.onload = () => {
    envSelect.dispatchEvent(new Event('change'));
    algoSelect.dispatchEvent(new Event('change'));
    clearValueCanvas();
    clearChart();
    fetchLogs();
};
