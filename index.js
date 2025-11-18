import init, { init_app, render_frame, set_feed_rate, set_kill_rate, set_paused, reset, handle_mouse_down, handle_mouse_up, handle_mouse_move, apply_preset, set_color_palette, set_kernel, set_zoom, set_pan, clear_canvas, add_random_blobs, set_emboss, set_map_mode, set_diffusion_u, set_diffusion_v, set_steps_per_frame, set_noise, set_boundary, set_gradient } from './pkg/gray_scott.js';

// Tab switching functionality
const tabs = document.querySelectorAll('.tab');
const tabContents = document.querySelectorAll('.tab-content');

tabs.forEach(tab => {
    tab.addEventListener('click', () => {
        // Remove active class from all tabs and contents
        tabs.forEach(t => t.classList.remove('active'));
        tabContents.forEach(tc => tc.classList.remove('active'));

        // Add active class to clicked tab
        tab.classList.add('active');

        // Show corresponding content
        const tabName = tab.getAttribute('data-tab');
        const content = document.getElementById(`${tabName}-tab`);
        if (content) {
            content.classList.add('active');
        }
    });
});

const canvas = document.getElementById('canvas');
const feedSlider = document.getElementById('feed');
const killSlider = document.getElementById('kill');
const feedValue = document.getElementById('feed-value');
const killValue = document.getElementById('kill-value');
const duSlider = document.getElementById('du');
const duValue = document.getElementById('du-value');
const dvSlider = document.getElementById('dv');
const dvValue = document.getElementById('dv-value');
const speedSlider = document.getElementById('speed');
const speedValue = document.getElementById('speed-value');
const noiseSlider = document.getElementById('noise');
const noiseValue = document.getElementById('noise-value');
const mapModeCheckbox = document.getElementById('map-mode');
const kernelSelect = document.getElementById('kernel');
const boundarySelect = document.getElementById('boundary');
const embossCheckbox = document.getElementById('emboss');
const invertPaletteCheckbox = document.getElementById('invert-palette');
const zoomSlider = document.getElementById('zoom');
const zoomValue = document.getElementById('zoom-value');
const panXSlider = document.getElementById('pan-x');
const panXValue = document.getElementById('pan-x-value');
const panYSlider = document.getElementById('pan-y');
const panYValue = document.getElementById('pan-y-value');
const gradientPresetSelect = document.getElementById('gradient-preset');
const pauseBtn = document.getElementById('pause');
const resetBtn = document.getElementById('reset');
const clearBtn = document.getElementById('clear');
const randomBtn = document.getElementById('random');
const presetSelect = document.getElementById('preset');
const status = document.getElementById('status');

let isPaused = false;
let isMouseDown = false;

// FPS tracking
let lastFrameTime = performance.now();
let frameCount = 0;
let fps = 0;
const fpsDisplay = document.getElementById('fps-display');

function showStatus(msg) {
    status.textContent = msg;
    status.style.display = 'block';
    setTimeout(() => {
        status.style.display = 'none';
    }, 2000);
}

async function start() {
    try {
        showStatus('Initializing WebGPU...');
        await init();
        await init_app('canvas');

        // Apply initial gradient to shader now that WASM is loaded
        updateGradientFromGrapick();
        
        // Mark WASM as initialized so resize can work
        wasmInitialized = true;

        showStatus('Ready!');
        requestAnimationFrame(renderLoop);
    } catch (e) {
        showStatus('Error: ' + e);
        console.error(e);
    }
}

function renderLoop() {
    try {
        render_frame();
        
        // Calculate FPS
        frameCount++;
        const currentTime = performance.now();
        const elapsed = currentTime - lastFrameTime;
        
        if (elapsed >= 1000) { // Update every second
            fps = Math.round((frameCount * 1000) / elapsed);
            fpsDisplay.textContent = `FPS: ${fps}`;
            frameCount = 0;
            lastFrameTime = currentTime;
        }
    } catch (e) {
        console.error('Render error:', e);
    }
    requestAnimationFrame(renderLoop);
}

// Feed rate control
feedSlider.addEventListener('input', (e) => {
    const value = parseFloat(e.target.value);
    feedValue.textContent = value.toFixed(4);
    set_feed_rate(value);
});

// Kill rate control
killSlider.addEventListener('input', (e) => {
    const value = parseFloat(e.target.value);
    killValue.textContent = value.toFixed(4);
    set_kill_rate(value);
});

// Kernel control
kernelSelect.addEventListener('change', (e) => {
    const value = parseInt(e.target.value);
    set_kernel(value);
});

// Boundary condition control
boundarySelect.addEventListener('change', (e) => {
    set_boundary(parseInt(e.target.value));
});

// Gradient picker control
const grapick = new Grapick({
    el: '#grapick-container',
    colorEl: '<input id="gp-color-picker" type="color"/>',
});

// Set up color picker integration
grapick.setColorPicker(handler => {
    const el = handler.getEl().querySelector('#gp-color-picker');

    if (el) {
        // Set initial color
        el.value = handler.getColor();

        // Listen for color changes
        const onChange = (e) => {
            handler.setColor(e.target.value);
            if (typeof set_gradient !== 'undefined') {
                updateGradientFromGrapick();
            }
        };

        const onInput = (e) => {
            handler.setColor(e.target.value, 0); // 0 = don't trigger change event
        };

        el.addEventListener('change', onChange);
        el.addEventListener('input', onInput);

        // Return cleanup function
        return () => {
            el.removeEventListener('change', onChange);
            el.removeEventListener('input', onInput);
        };
    }
});

// Gradient presets
const gradientPresets = {
    rainbow: [
        { pos: 0, color: '#330033' },
        { pos: 20, color: '#8000ff' },
        { pos: 40, color: '#0080ff' },
        { pos: 60, color: '#00ffcc' },
        { pos: 80, color: '#ff4d00' },
        { pos: 100, color: '#ff0000' }
    ],
    sunset: [
        { pos: 0, color: '#0a0a2e' },
        { pos: 25, color: '#6b2d5c' },
        { pos: 50, color: '#e05297' },
        { pos: 75, color: '#ff6f3c' },
        { pos: 100, color: '#ffe66d' }
    ],
    ocean: [
        { pos: 0, color: '#000428' },
        { pos: 30, color: '#004e92' },
        { pos: 55, color: '#0077be' },
        { pos: 75, color: '#00a8cc' },
        { pos: 90, color: '#00d9ff' },
        { pos: 100, color: '#8ef6e4' }
    ],
    fire: [
        { pos: 0, color: '#0d0d0d' },
        { pos: 20, color: '#4a0000' },
        { pos: 40, color: '#8b0000' },
        { pos: 60, color: '#ff4500' },
        { pos: 80, color: '#ffa500' },
        { pos: 100, color: '#ffff00' }
    ],
    neon: [
        { pos: 0, color: '#0f0f23' },
        { pos: 25, color: '#ff00ff' },
        { pos: 50, color: '#00ffff' },
        { pos: 75, color: '#00ff00' },
        { pos: 100, color: '#ff1493' }
    ],
    forest: [
        { pos: 0, color: '#0c1445' },
        { pos: 30, color: '#1a3a2e' },
        { pos: 50, color: '#0e5e3a' },
        { pos: 70, color: '#2d8659' },
        { pos: 85, color: '#76c893' },
        { pos: 100, color: '#b8f3d8' }
    ]
};

function loadGradientPreset(presetName) {
    const preset = gradientPresets[presetName];
    if (!preset) return;

    // Clear ALL existing handlers
    let handlers = grapick.getHandlers();
    while (handlers.length > 0) {
        handlers[0].remove();
        handlers = grapick.getHandlers();
    }

    console.log('Cleared all handlers, now adding preset:', presetName);

    // Add new handlers from preset
    preset.forEach(stop => {
        grapick.addHandler(stop.pos, stop.color);
    });

    // Update shader with new gradient (only if WASM is loaded)
    if (typeof set_gradient !== 'undefined') {
        updateGradientFromGrapick();
    }
}

// Initialize with default Rainbow gradient (visual only)
gradientPresets.rainbow.forEach(stop => {
    grapick.addHandler(stop.pos, stop.color);
});

function updateGradientFromGrapick() {
    const handlers = grapick.getHandlers();

    // Sort handlers by position
    handlers.sort((a, b) => a.getPosition() - b.getPosition());

    const positions = [];
    const colors = [];

    handlers.forEach((handler, idx) => {
        const pos = handler.getPosition() / 100.0;
        let color = handler.getColor();

        let r = 0, g = 0, b = 0;

        // Grapick can return color in various formats, try to parse intelligently
        if (typeof color === 'string') {
            // String format (hex)
            const hex = color.replace('#', '');
            if (hex.length === 6) {
                r = parseInt(hex.substring(0, 2), 16) / 255.0;
                g = parseInt(hex.substring(2, 4), 16) / 255.0;
                b = parseInt(hex.substring(4, 6), 16) / 255.0;
            }
        } else if (color && typeof color === 'object') {
            console.log(`  Color object keys:`, Object.keys(color));
            console.log(`  Color.r=${color.r}, Color.g=${color.g}, Color.b=${color.b}`);

            // Object format - try multiple property names that Grapick might use
            if ('r' in color && 'g' in color && 'b' in color) {
                r = (Number(color.r) || 0) / 255.0;
                g = (Number(color.g) || 0) / 255.0;
                b = (Number(color.b) || 0) / 255.0;
            } else if ('red' in color && 'green' in color && 'blue' in color) {
                r = (Number(color.red) || 0) / 255.0;
                g = (Number(color.green) || 0) / 255.0;
                b = (Number(color.blue) || 0) / 255.0;
            } else if (color.toString) {
                // Try converting to string and parse as hex
                const colorStr = color.toString();
                if (colorStr.includes('#')) {
                    const hex = colorStr.replace('#', '').substring(0, 6);
                    r = parseInt(hex.substring(0, 2), 16) / 255.0;
                    g = parseInt(hex.substring(2, 4), 16) / 255.0;
                    b = parseInt(hex.substring(4, 6), 16) / 255.0;
                }
            }
        }

        // Validate all values are numbers
        if (isNaN(r) || isNaN(g) || isNaN(b)) {
            console.error('FAILED to parse color - Final RGB:', r, g, b);
            // Use gray as fallback
            r = 0.5;
            g = 0.5;
            b = 0.5;
        }

        positions.push(pos);
        colors.push(r, g, b, 1.0);

    });

    set_gradient(new Float32Array(positions), new Float32Array(colors));
}

// Listen to Grapick events for handler changes (moving, adding, removing)
grapick.on('change', () => {
    console.log('Handler position changed');
    if (typeof set_gradient !== 'undefined') {
        updateGradientFromGrapick();
    }
});

grapick.on('handler:add', () => {
    console.log('Handler added');
    if (typeof set_gradient !== 'undefined') {
        updateGradientFromGrapick();
    }
});

grapick.on('handler:remove', () => {
    console.log('Handler removed');
    if (typeof set_gradient !== 'undefined') {
        updateGradientFromGrapick();
    }
});

// Gradient preset selector
gradientPresetSelect.addEventListener('change', (e) => {
    loadGradientPreset(e.target.value);
});

// Emboss control
embossCheckbox.addEventListener('change', (e) => {
    set_emboss(e.target.checked);
});

// Diffusion U control
duSlider.addEventListener('input', (e) => {
    const value = parseFloat(e.target.value);
    duValue.textContent = value.toFixed(2);
    set_diffusion_u(value);
});

// Diffusion V control
dvSlider.addEventListener('input', (e) => {
    const value = parseFloat(e.target.value);
    dvValue.textContent = value.toFixed(2);
    set_diffusion_v(value);
});

// Speed control (steps per frame)
speedSlider.addEventListener('input', (e) => {
    const value = parseInt(e.target.value);
    speedValue.textContent = value;
    set_steps_per_frame(value);
});

// Noise injection control
noiseSlider.addEventListener('input', (e) => {
    const value = parseFloat(e.target.value);
    noiseValue.textContent = value.toFixed(4);
    set_noise(value);
});

// Map mode control
const mapLabelsContainer = document.getElementById('map-labels');

function createMapModeGrid() {
    // Draw a dense 32x32 grid to fill the canvas quickly
    const gridSize = 32;

    handle_mouse_down(0, 0); // Start painting mode

    for (let i = 0; i < gridSize; i++) {
        for (let j = 0; j < gridSize; j++) {
            // Use normalized coordinates (0.0 to 1.0)
            const x = (i + 0.5) / gridSize;
            const y = (j + 0.5) / gridSize;

            // Paint a single point at each grid location
            handle_mouse_move(x, y);
        }
    }

    handle_mouse_up(); // Stop painting mode
}

function updateMapLabels() {
    mapLabelsContainer.innerHTML = '';

    const rect = canvas.getBoundingClientRect();
    const canvasWidth = rect.width;
    const canvasHeight = rect.height;
    const labelCount = 5;

    // Get current zoom and pan values
    const currentZoom = parseFloat(zoomSlider.value);
    const currentPanX = parseFloat(panXSlider.value);
    const currentPanY = parseFloat(panYSlider.value);

    // X-axis labels (k values) - adjusted for zoom and pan
    for (let i = 0; i < labelCount; i++) {
        const screenT = i / (labelCount - 1);
        // Convert screen position to simulation space
        const simX = (screenT - 0.5) / currentZoom + 0.5 - currentPanX;

        // Only show labels that are within visible bounds
        if (simX >= 0.0 && simX <= 1.0) {
            const k = 0.045 + simX * (0.070 - 0.045);
            const x = screenT * canvasWidth;

            const label = document.createElement('div');
            label.className = 'map-label-x';
            label.textContent = `k=${k.toFixed(3)}`;
            label.style.left = `${x}px`;
            mapLabelsContainer.appendChild(label);
        }
    }

    // Y-axis labels (F values) - adjusted for zoom and pan
    for (let i = 0; i < labelCount; i++) {
        const screenT = i / (labelCount - 1);
        // Convert screen position to simulation space
        const simY = (screenT - 0.5) / currentZoom + 0.5 - currentPanY;

        // Only show labels that are within visible bounds
        if (simY >= 0.0 && simY <= 1.0) {
            const f = 0.010 + simY * (0.100 - 0.010);
            const y = screenT * canvasHeight;

            const label = document.createElement('div');
            label.className = 'map-label-y';
            label.textContent = `F=${f.toFixed(3)}`;
            label.style.top = `${y}px`;
            mapLabelsContainer.appendChild(label);
        }
    }
}

mapModeCheckbox.addEventListener('change', (e) => {
    const enabled = e.target.checked;
    set_map_mode(enabled);

    if (enabled) {
        // Reset canvas and create grid when entering map mode
        clear_canvas();
        mapLabelsContainer.style.display = 'block';
        updateMapLabels();

        // Create an even grid of initial dots
        setTimeout(() => {
            createMapModeGrid();
        }, 50);
    } else {
        // Hide labels and reset when exiting map mode
        mapLabelsContainer.style.display = 'none';
        reset();
    }
});

// Zoom control with mouse wheel
zoomSlider.addEventListener('input', (e) => {
    const value = parseFloat(e.target.value);
    zoomValue.textContent = value.toFixed(1) + 'x';
    set_zoom(value);
    if (mapModeCheckbox.checked) {
        updateMapLabels();
    }
});

canvas.addEventListener('wheel', (e) => {
    e.preventDefault();
    const delta = -Math.sign(e.deltaY) * 0.5;
    let newZoom = Math.max(1.0, Math.min(10.0, parseFloat(zoomSlider.value) + delta));
    zoomSlider.value = newZoom;
    zoomValue.textContent = newZoom.toFixed(1) + 'x';
    set_zoom(newZoom);
    if (mapModeCheckbox.checked) {
        updateMapLabels();
    }
});

// Pan X control
panXSlider.addEventListener('input', (e) => {
    const value = parseFloat(e.target.value);
    panXValue.textContent = value.toFixed(2);
    set_pan(value, parseFloat(panYSlider.value));
    if (mapModeCheckbox.checked) {
        updateMapLabels();
    }
});

// Pan Y control
panYSlider.addEventListener('input', (e) => {
    const value = parseFloat(e.target.value);
    panYValue.textContent = value.toFixed(2);
    set_pan(parseFloat(panXSlider.value), value);
    if (mapModeCheckbox.checked) {
        updateMapLabels();
    }
});

// WASD keyboard controls for panning
const panSpeed = 0.05;
document.addEventListener('keydown', (e) => {
    let panX = parseFloat(panXSlider.value);
    let panY = parseFloat(panYSlider.value);
    let changed = false;

    if (e.key === 'w' || e.key === 'W') {
        panY = Math.max(-0.5, Math.min(0.5, panY - panSpeed));
        changed = true;
    } else if (e.key === 's' || e.key === 'S') {
        panY = Math.max(-0.5, Math.min(0.5, panY + panSpeed));
        changed = true;
    } else if (e.key === 'a' || e.key === 'A') {
        panX = Math.max(-0.5, Math.min(0.5, panX - panSpeed));
        changed = true;
    } else if (e.key === 'd' || e.key === 'D') {
        panX = Math.max(-0.5, Math.min(0.5, panX + panSpeed));
        changed = true;
    }

    if (changed) {
        e.preventDefault();
        panXSlider.value = panX;
        panYSlider.value = panY;
        panXValue.textContent = panX.toFixed(2);
        panYValue.textContent = panY.toFixed(2);
        set_pan(panX, panY);
        if (mapModeCheckbox.checked) {
            updateMapLabels();
        }
    }
});

// Preset selection
presetSelect.addEventListener('change', (e) => {
    const [feed, kill] = e.target.value.split(',').map(parseFloat);
    feedSlider.value = feed;
    killSlider.value = kill;
    feedValue.textContent = feed.toFixed(4);
    killValue.textContent = kill.toFixed(4);
    apply_preset(feed, kill);
});

// Pause button
pauseBtn.addEventListener('click', () => {
    isPaused = !isPaused;
    pauseBtn.textContent = isPaused ? 'Resume' : 'Pause';
    pauseBtn.style.background = isPaused ? '#4a9eff' : '#555';
    set_paused(isPaused);
});

// Reset button
resetBtn.addEventListener('click', () => {
    reset();
    // Reset zoom and pan
    zoomSlider.value = 1.0;
    panXSlider.value = 0;
    panYSlider.value = 0;
    zoomValue.textContent = '1.0x';
    panXValue.textContent = '0.0';
    panYValue.textContent = '0.0';
    set_zoom(1.0);
    set_pan(0, 0);
    showStatus('Reset!');
});

// Clear button
clearBtn.addEventListener('click', () => {
    clear_canvas();
    showStatus('Cleared!');
});

// Random blobs button
randomBtn.addEventListener('click', () => {
    add_random_blobs();
    showStatus('Added random blobs!');
});

// Mouse painting
canvas.addEventListener('mousedown', (e) => {
    isMouseDown = true;
    const rect = canvas.getBoundingClientRect();
    const x = (e.clientX - rect.left) / rect.width;
    const y = (e.clientY - rect.top) / rect.height;
    handle_mouse_down(x, y);
});

canvas.addEventListener('mouseup', () => {
    isMouseDown = false;
    handle_mouse_up();
});

canvas.addEventListener('mousemove', (e) => {
    if (isMouseDown) {
        const rect = canvas.getBoundingClientRect();
        const x = (e.clientX - rect.left) / rect.width;
        const y = (e.clientY - rect.top) / rect.height;
        handle_mouse_move(x, y);
    }
});

canvas.addEventListener('mouseleave', () => {
    if (isMouseDown) {
        isMouseDown = false;
        handle_mouse_up();
    }
});

// Handle canvas resize
let resizeTimeout;
let wasmInitialized = false;

async function resizeCanvas() {
    const displayWidth = canvas.clientWidth;
    const displayHeight = canvas.clientHeight;

    if (canvas.width !== displayWidth || canvas.height !== displayHeight) {
        canvas.width = displayWidth;
        canvas.height = displayHeight;
        
        // Only reinitialize if WASM is already loaded
        if (wasmInitialized) {
            // Debounce the reinitialization to avoid too many calls
            clearTimeout(resizeTimeout);
            resizeTimeout = setTimeout(async () => {
                try {
                    await init_app('canvas');
                    updateGradientFromGrapick();
                } catch (e) {
                    console.error('Resize reinit error:', e);
                }
            }, 250);
        }
    }
}

window.addEventListener('resize', resizeCanvas);
resizeCanvas();

// Draggable sidebar resize
const sidebar = document.querySelector('.sidebar');
const canvasWrapper = document.querySelector('.canvas-wrapper');
let isResizing = false;
let lastX = 0;

// Create resize handle
const resizeHandle = document.createElement('div');
resizeHandle.className = 'sidebar-resize-handle';
sidebar.appendChild(resizeHandle);

resizeHandle.addEventListener('mousedown', (e) => {
    isResizing = true;
    lastX = e.clientX;
    document.body.style.cursor = 'ew-resize';
    document.body.style.userSelect = 'none';
    e.preventDefault();
});

document.addEventListener('mousemove', (e) => {
    if (!isResizing) return;
    
    const delta = e.clientX - lastX;
    const currentWidth = sidebar.offsetWidth;
    // Subtract delta because sidebar is on the right side
    // Dragging left (negative delta) should increase width
    const newWidth = Math.max(280, Math.min(600, currentWidth - delta));
    
    sidebar.style.width = newWidth + 'px';
    lastX = e.clientX;
    
    // Trigger canvas resize after sidebar width change
    resizeCanvas();
});

document.addEventListener('mouseup', () => {
    if (isResizing) {
        isResizing = false;
        document.body.style.cursor = '';
        document.body.style.userSelect = '';
    }
});

// Start the application
start();