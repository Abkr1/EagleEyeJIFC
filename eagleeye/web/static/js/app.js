/* ============================================================
   EagleEye — Main Application JavaScript
   Shared utilities used across all pages
   ============================================================ */

// ---- API Helper ----
async function api(url, method = 'GET', body = null) {
    try {
        const opts = {
            method,
            headers: { 'Content-Type': 'application/json' },
        };
        if (body) opts.body = JSON.stringify(body);
        const res = await fetch(url, opts);
        return await res.json();
    } catch (err) {
        console.error('API Error:', err);
        return null;
    }
}

// ---- Tab Switching ----
function switchTab(tabEl, panelId) {
    // Deactivate all tabs in same group
    tabEl.parentElement.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
    tabEl.classList.add('active');

    // Hide all panels, show target
    const panels = tabEl.closest('.content-area, .tab-panel')
        ?.parentElement?.querySelectorAll('.tab-panel')
        || document.querySelectorAll('.tab-panel');
    panels.forEach(p => p.classList.remove('active'));
    const target = document.getElementById(panelId);
    if (target) target.classList.add('active');
}

// ---- Toast Notifications ----
function showToast(message, type = 'info') {
    let container = document.querySelector('.toast-container');
    if (!container) {
        container = document.createElement('div');
        container.className = 'toast-container';
        document.body.appendChild(container);
    }
    const toast = document.createElement('div');
    toast.className = `toast toast-${type}`;
    toast.textContent = message;
    container.appendChild(toast);
    setTimeout(() => { toast.remove(); }, 4000);
}

// ---- Date Formatting ----
function formatDate(dateStr) {
    if (!dateStr) return '—';
    const d = new Date(dateStr);
    if (isNaN(d.getTime())) return dateStr;
    const month = String(d.getMonth() + 1).padStart(2, '0');
    const day = String(d.getDate()).padStart(2, '0');
    return `${d.getFullYear()}-${month}-${day}`;
}

// ---- Incident Type Formatting ----
function formatType(type) {
    if (!type) return '—';
    return type.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
}

// ---- Badge CSS Class ----
function getBadgeClass(type) {
    const map = {
        village_raid: 'badge-raid',
        kidnapping: 'badge-kidnapping',
        highway_ambush: 'badge-ambush',
        cattle_rustling: 'badge-rustling',
        attack_on_security_forces: 'badge-military',
        market_attack: 'badge-raid',
        reprisal_attack: 'badge-raid',
    };
    return map[type] || 'badge-default';
}

// ---- Threat Level Helpers ----
function getThreatLabel(level) {
    return ['', 'LOW', 'MODERATE', 'ELEVATED', 'HIGH', 'CRITICAL'][level] || 'UNKNOWN';
}

function getThreatColor(level) {
    const colors = {
        1: '#10b981',
        2: '#06b6d4',
        3: '#f59e0b',
        4: '#f97316',
        5: '#ef4444',
    };
    return colors[level] || '#94a3b8';
}

// ---- Map Search (Coordinates + Location Name) ----
let _mapSearchMarkers = {};

function _placeSearchMarker(mapInstance, markerId, lat, lon, label) {
    if (_mapSearchMarkers[markerId]) {
        mapInstance.removeLayer(_mapSearchMarkers[markerId]);
    }
    mapInstance.flyTo([lat, lon], 12);
    _mapSearchMarkers[markerId] = L.marker([lat, lon])
        .addTo(mapInstance)
        .bindPopup(label)
        .openPopup();
}

function searchCoordinates(mapInstance, inputId) {
    const input = document.getElementById(inputId);
    if (!input || !mapInstance) return;

    const raw = input.value.trim();
    if (!raw) { showToast('Enter coordinates (e.g. 12.11, 5.96)', 'error'); return; }

    const parts = raw.split(/[\s,]+/).map(Number);
    if (parts.length < 2 || isNaN(parts[0]) || isNaN(parts[1])) {
        showToast('Invalid format. Use: latitude, longitude', 'error');
        return;
    }

    const lat = parts[0], lon = parts[1];
    if (lat < -90 || lat > 90 || lon < -180 || lon > 180) {
        showToast('Coordinates out of range', 'error');
        return;
    }

    _placeSearchMarker(mapInstance, inputId, lat, lon,
        `<b>Search Result</b><br>Lat: ${lat.toFixed(4)}<br>Lon: ${lon.toFixed(4)}`);
}

async function searchLocation(mapInstance, inputId) {
    const input = document.getElementById(inputId);
    if (!input || !mapInstance) return;

    const query = input.value.trim();
    if (!query) { showToast('Enter a location name', 'error'); return; }

    try {
        const url = `https://nominatim.openstreetmap.org/search?format=json&q=${encodeURIComponent(query + ', Nigeria')}&limit=1`;
        const resp = await fetch(url);
        const results = await resp.json();

        if (!results.length) {
            showToast('Location not found. Try a more specific name.', 'error');
            return;
        }

        const r = results[0];
        const lat = parseFloat(r.lat);
        const lon = parseFloat(r.lon);
        const name = r.display_name.split(',').slice(0, 3).join(',');

        _placeSearchMarker(mapInstance, inputId, lat, lon,
            `<b>${name}</b><br>Lat: ${lat.toFixed(4)}<br>Lon: ${lon.toFixed(4)}`);
    } catch (e) {
        showToast('Search failed: ' + e.message, 'error');
    }
}

function bindMapSearchEnter(inputId, mapGetter, searchFn) {
    const input = document.getElementById(inputId);
    if (input) {
        input.addEventListener('keydown', (e) => {
            if (e.key === 'Enter') {
                e.preventDefault();
                const map = typeof mapGetter === 'function' ? mapGetter() : mapGetter;
                if (map) searchFn(map, inputId);
                else showToast('Load the map first', 'error');
            }
        });
    }
}

// Legacy wrapper
function bindCoordSearchEnter(inputId, mapGetter) {
    bindMapSearchEnter(inputId, mapGetter, searchCoordinates);
}

// ---- Clock ----
function updateClock() {
    const el = document.getElementById('currentTime');
    if (el) {
        const now = new Date();
        const wat = new Date(now.getTime() + 60 * 60 * 1000); // UTC+1
        el.textContent = wat.toISOString().replace('T', ' ').substring(0, 19) + ' WAT';
    }
}
setInterval(updateClock, 1000);
updateClock();

// ---- System Status ----
async function checkSystemStatus() {
    const data = await api('/api/status');
    const dot = document.getElementById('systemStatusDot');
    const text = document.getElementById('systemStatusText');
    if (data && data.status === 'operational') {
        dot.className = 'status-dot';
        text.textContent = 'System Operational';
    } else {
        dot.className = 'status-dot offline';
        text.textContent = 'System Offline';
    }
}
checkSystemStatus();
