/**
 * GenAI RAG Q&A System - Frontend Application
 * Handles all UI interactions, API calls, and state management
 */

// ==================== Configuration ====================
const CONFIG = {
    apiUrl: 'http://localhost:8000',
    maxFileSize: 10 * 1024 * 1024, // 10MB
    allowedFileTypes: ['.pdf', '.docx', '.txt'],
};

// ==================== State ====================
let state = {
    messages: [],
    files: [],
    isLoading: false,
    documentCount: 0,
};

// ==================== DOM Elements ====================
const elements = {
    // Tabs
    navItems: document.querySelectorAll('.nav-item'),
    tabContents: document.querySelectorAll('.tab-content'),

    // Chat
    messagesContainer: document.getElementById('messages'),
    userInput: document.getElementById('user-input'),
    sendBtn: document.getElementById('send-btn'),

    // Upload
    urlInput: document.getElementById('url-input'),
    scrapeBtn: document.getElementById('scrape-btn'),
    fileInput: document.getElementById('file-input'),
    fileDropZone: document.getElementById('file-drop-zone'),
    fileList: document.getElementById('file-list'),
    uploadBtn: document.getElementById('upload-btn'),
    textTitle: document.getElementById('text-title'),
    textInput: document.getElementById('text-input'),
    addTextBtn: document.getElementById('add-text-btn'),

    // Settings
    apiUrlInput: document.getElementById('api-url'),
    testConnectionBtn: document.getElementById('test-connection-btn'),
    clearKbBtn: document.getElementById('clear-kb-btn'),

    // Stats
    docCount: document.getElementById('doc-count'),
    totalDocs: document.getElementById('total-docs'),
    statusBadge: document.getElementById('status-badge'),

    // Notifications
    toastContainer: document.getElementById('toast-container'),
    loadingOverlay: document.getElementById('loading-overlay'),
    loadingText: document.getElementById('loading-text'),
};

// ==================== Initialization ====================
document.addEventListener('DOMContentLoaded', () => {
    initTabs();
    initChat();
    initUpload();
    initSettings();
    loadStats();

    // Load saved API URL
    const savedUrl = localStorage.getItem('apiUrl');
    if (savedUrl) {
        CONFIG.apiUrl = savedUrl;
        elements.apiUrlInput.value = savedUrl;
    }
});

// ==================== Tab Navigation ====================
function initTabs() {
    elements.navItems.forEach(item => {
        item.addEventListener('click', () => {
            const tabId = item.dataset.tab;

            // Update active nav item
            elements.navItems.forEach(nav => nav.classList.remove('active'));
            item.classList.add('active');

            // Show active tab content
            elements.tabContents.forEach(tab => {
                tab.classList.remove('active');
                if (tab.id === `${tabId}-tab`) {
                    tab.classList.add('active');
                }
            });
        });
    });
}

// ==================== Chat Functionality ====================
function initChat() {
    // Send message on button click
    elements.sendBtn.addEventListener('click', sendMessage);

    // Send message on Enter (Shift+Enter for new line)
    elements.userInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });

    // Auto-resize textarea
    elements.userInput.addEventListener('input', () => {
        elements.userInput.style.height = 'auto';
        elements.userInput.style.height = Math.min(elements.userInput.scrollHeight, 120) + 'px';
    });
}

async function sendMessage() {
    const message = elements.userInput.value.trim();
    if (!message || state.isLoading) return;

    // Add user message to UI
    addMessage('user', message);
    elements.userInput.value = '';
    elements.userInput.style.height = 'auto';

    // Show loading state
    setLoading(true, 'Thinking...');

    try {
        const response = await fetch(`${CONFIG.apiUrl}/query`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ question: message, top_k: 5 }),
        });

        if (!response.ok) throw new Error('Failed to get response');

        const data = await response.json();

        // Add assistant message
        addMessage('assistant', data.answer, data.sources);

    } catch (error) {
        console.error('Error:', error);
        addMessage('assistant', 'Sorry, I encountered an error. Please check if the backend is running and try again.');
        showToast('Failed to get response. Is the server running?', 'error');
    } finally {
        setLoading(false);
    }
}

function addMessage(role, content, sources = []) {
    const timestamp = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });

    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${role}`;

    let sourcesHtml = '';
    if (sources && sources.length > 0) {
        sourcesHtml = `
            <div class="sources">
                <div class="sources-title">📚 Sources</div>
                ${sources.map(s => `
                    <div class="source-item">
                        ${s.title || s.source} (${(s.relevance_score * 100).toFixed(0)}% relevant)
                    </div>
                `).join('')}
            </div>
        `;
    }

    messageDiv.innerHTML = `
        <div class="message-avatar">${role === 'user' ? '👤' : '🤖'}</div>
        <div class="message-content">
            <p>${formatMessage(content)}</p>
            ${sourcesHtml}
            <div class="message-meta">
                <span class="timestamp">${timestamp}</span>
            </div>
        </div>
    `;

    elements.messagesContainer.appendChild(messageDiv);
    elements.messagesContainer.scrollTop = elements.messagesContainer.scrollHeight;

    // Save to state
    state.messages.push({ role, content, sources, timestamp });
}

function formatMessage(text) {
    // Basic markdown-like formatting
    return text
        .replace(/\n/g, '<br>')
        .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
        .replace(/\*(.*?)\*/g, '<em>$1</em>')
        .replace(/`(.*?)`/g, '<code>$1</code>');
}

// ==================== Upload Functionality ====================
function initUpload() {
    // URL Scraping
    elements.scrapeBtn.addEventListener('click', scrapeUrls);

    // File Upload
    elements.fileDropZone.addEventListener('click', () => elements.fileInput.click());
    elements.fileDropZone.addEventListener('dragover', handleDragOver);
    elements.fileDropZone.addEventListener('dragleave', handleDragLeave);
    elements.fileDropZone.addEventListener('drop', handleFileDrop);
    elements.fileInput.addEventListener('change', handleFileSelect);
    elements.uploadBtn.addEventListener('click', uploadFiles);

    // Text Input
    elements.addTextBtn.addEventListener('click', addText);
}

async function scrapeUrls() {
    const urlText = elements.urlInput.value.trim();
    if (!urlText) {
        showToast('Please enter at least one URL', 'warning');
        return;
    }

    const urls = urlText.split('\n').map(u => u.trim()).filter(u => u);

    if (urls.length === 0) {
        showToast('Please enter valid URLs', 'warning');
        return;
    }

    setLoading(true, 'Scraping web pages...');

    try {
        const response = await fetch(`${CONFIG.apiUrl}/ingest/url`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ urls }),
        });

        const data = await response.json();

        if (data.success) {
            showToast(`Successfully added ${data.documents_added} documents (${data.chunks_created} chunks)`, 'success');
            elements.urlInput.value = '';
            loadStats();
        } else {
            showToast(data.message || 'Failed to scrape URLs', 'error');
        }

    } catch (error) {
        console.error('Error:', error);
        showToast('Failed to scrape URLs. Check the server connection.', 'error');
    } finally {
        setLoading(false);
    }
}

function handleDragOver(e) {
    e.preventDefault();
    e.currentTarget.classList.add('dragover');
}

function handleDragLeave(e) {
    e.currentTarget.classList.remove('dragover');
}

function handleFileDrop(e) {
    e.preventDefault();
    e.currentTarget.classList.remove('dragover');

    const files = Array.from(e.dataTransfer.files);
    addFiles(files);
}

function handleFileSelect(e) {
    const files = Array.from(e.target.files);
    addFiles(files);
}

function addFiles(files) {
    const validFiles = files.filter(file => {
        const ext = '.' + file.name.split('.').pop().toLowerCase();
        const isValid = CONFIG.allowedFileTypes.includes(ext);
        const isValidSize = file.size <= CONFIG.maxFileSize;

        if (!isValid) showToast(`${file.name}: Unsupported file type`, 'warning');
        if (!isValidSize) showToast(`${file.name}: File too large (max 10MB)`, 'warning');

        return isValid && isValidSize;
    });

    state.files = [...state.files, ...validFiles];
    renderFileList();
}

function renderFileList() {
    elements.fileList.innerHTML = state.files.map((file, index) => `
        <div class="file-item">
            <span>📄 ${file.name}</span>
            <button class="file-remove" onclick="removeFile(${index})">×</button>
        </div>
    `).join('');

    elements.uploadBtn.disabled = state.files.length === 0;
}

function removeFile(index) {
    state.files.splice(index, 1);
    renderFileList();
}

async function uploadFiles() {
    if (state.files.length === 0) return;

    setLoading(true, 'Uploading files...');

    let successCount = 0;
    let totalChunks = 0;

    for (const file of state.files) {
        try {
            const formData = new FormData();
            formData.append('file', file);

            const response = await fetch(`${CONFIG.apiUrl}/ingest/file`, {
                method: 'POST',
                body: formData,
            });

            const data = await response.json();

            if (data.success) {
                successCount++;
                totalChunks += data.chunks_created;
            }
        } catch (error) {
            console.error(`Error uploading ${file.name}:`, error);
        }
    }

    if (successCount > 0) {
        showToast(`Uploaded ${successCount} files (${totalChunks} chunks)`, 'success');
        state.files = [];
        renderFileList();
        loadStats();
    } else {
        showToast('Failed to upload files', 'error');
    }

    setLoading(false);
}

async function addText() {
    const text = elements.textInput.value.trim();

    if (!text || text.length < 50) {
        showToast('Please enter at least 50 characters', 'warning');
        return;
    }

    setLoading(true, 'Adding text...');

    try {
        const response = await fetch(`${CONFIG.apiUrl}/ingest/text`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                text,
                title: elements.textTitle.value.trim() || 'User Input',
                source: 'manual_input',
            }),
        });

        const data = await response.json();

        if (data.success) {
            showToast(`Added text (${data.chunks_created} chunks)`, 'success');
            elements.textInput.value = '';
            elements.textTitle.value = '';
            loadStats();
        } else {
            showToast(data.message || 'Failed to add text', 'error');
        }

    } catch (error) {
        console.error('Error:', error);
        showToast('Failed to add text', 'error');
    } finally {
        setLoading(false);
    }
}

// ==================== Settings Functionality ====================
function initSettings() {
    elements.testConnectionBtn.addEventListener('click', testConnection);
    elements.clearKbBtn.addEventListener('click', clearKnowledgeBase);

    elements.apiUrlInput.addEventListener('change', () => {
        CONFIG.apiUrl = elements.apiUrlInput.value;
        localStorage.setItem('apiUrl', CONFIG.apiUrl);
    });
}

async function testConnection() {
    setLoading(true, 'Testing connection...');

    try {
        const response = await fetch(`${CONFIG.apiUrl}/health`);
        const data = await response.json();

        if (data.status === 'healthy') {
            showToast('Connection successful! Server is healthy.', 'success');
            elements.statusBadge.textContent = 'Connected';
            elements.statusBadge.style.background = 'var(--success)';
        } else {
            throw new Error('Unhealthy response');
        }
    } catch (error) {
        console.error('Error:', error);
        showToast('Connection failed. Check the server URL.', 'error');
        elements.statusBadge.textContent = 'Offline';
        elements.statusBadge.style.background = 'var(--error)';
    } finally {
        setLoading(false);
    }
}

async function clearKnowledgeBase() {
    if (!confirm('Are you sure you want to clear all documents? This cannot be undone!')) {
        return;
    }

    setLoading(true, 'Clearing knowledge base...');

    try {
        const response = await fetch(`${CONFIG.apiUrl}/ingest/clear`, {
            method: 'DELETE',
        });

        const data = await response.json();

        if (data.success) {
            showToast('Knowledge base cleared', 'success');
            loadStats();
        } else {
            showToast('Failed to clear knowledge base', 'error');
        }
    } catch (error) {
        console.error('Error:', error);
        showToast('Failed to clear knowledge base', 'error');
    } finally {
        setLoading(false);
    }
}

// ==================== Stats ====================
async function loadStats() {
    try {
        const response = await fetch(`${CONFIG.apiUrl}/query/stats`);
        const data = await response.json();

        state.documentCount = data.document_count || 0;
        elements.docCount.textContent = state.documentCount;
        elements.totalDocs.textContent = state.documentCount;

    } catch (error) {
        // Silently fail - server might not be running
        console.log('Could not load stats');
    }
}

// ==================== Utilities ====================
function setLoading(isLoading, text = 'Processing...') {
    state.isLoading = isLoading;
    elements.loadingText.textContent = text;

    if (isLoading) {
        elements.loadingOverlay.classList.add('active');
    } else {
        elements.loadingOverlay.classList.remove('active');
    }
}

function showToast(message, type = 'info') {
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    toast.innerHTML = `<p>${message}</p>`;

    elements.toastContainer.appendChild(toast);

    // Auto-remove after 4 seconds
    setTimeout(() => {
        toast.style.animation = 'slideIn 0.3s ease reverse';
        setTimeout(() => toast.remove(), 300);
    }, 4000);
}

// Make removeFile available globally
window.removeFile = removeFile;

// ==================== Knowledge Graph ====================
let kgState = {
    simulation: null,
    graphData: null,
    selectedNode: null,
    zoom: null,
    highlightTimer: null
};

// Initialize Knowledge Graph when tab is clicked
document.addEventListener('DOMContentLoaded', () => {
    const kgTab = document.querySelector('[data-tab="knowledge-graph"]');
    if (kgTab) {
        kgTab.addEventListener('click', () => {
            setTimeout(initKnowledgeGraph, 100);
        });
    }

    // Refresh button
    const refreshBtn = document.getElementById('refresh-graph-btn');
    if (refreshBtn) {
        refreshBtn.addEventListener('click', loadKnowledgeGraph);
    }

    // Search functionality
    const searchInput = document.getElementById('kg-search');
    if (searchInput) {
        searchInput.addEventListener('input', debounce(searchKnowledgeGraph, 300));
    }

    // Filter functionality
    const filterSelect = document.getElementById('kg-filter');
    if (filterSelect) {
        filterSelect.addEventListener('change', searchKnowledgeGraph);
    }

    // Zoom controls
    document.getElementById('zoom-in')?.addEventListener('click', () => zoomGraph(1.3));
    document.getElementById('zoom-out')?.addEventListener('click', () => zoomGraph(0.7));
    document.getElementById('zoom-reset')?.addEventListener('click', resetZoom);
    document.getElementById('zoom-fit')?.addEventListener('click', fitGraph);

    // Close details button
    document.getElementById('close-details')?.addEventListener('click', hideEntityDetails);

    // Legend item clicks for filtering
    document.querySelectorAll('.legend-item').forEach(item => {
        item.addEventListener('click', () => {
            const type = item.dataset.type;
            document.getElementById('kg-filter').value = type || '';
            searchKnowledgeGraph();
        });
    });

    // Quick stat clicks
    document.querySelectorAll('.quick-stat').forEach(stat => {
        stat.addEventListener('click', () => {
            const type = stat.id.replace('stat-', '').replace('-', '_');
            if (type === 'products') document.getElementById('kg-filter').value = 'product';
            else if (type === 'brands') document.getElementById('kg-filter').value = 'brand';
            else if (type === 'categories') document.getElementById('kg-filter').value = 'category';
            else if (type === 'price_ranges') document.getElementById('kg-filter').value = 'price_range';
            searchKnowledgeGraph();
        });
    });
});

function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

async function initKnowledgeGraph() {
    console.log('Initializing Knowledge Graph...');
    await loadKGStats();
    await loadKnowledgeGraph();
}

async function loadKGStats() {
    try {
        const response = await fetch(`${CONFIG.apiUrl}/query/knowledge-graph/stats`);
        const data = await response.json();
        console.log('KG Stats:', data);

        document.getElementById('kg-entities').textContent = data.total_entities || 0;
        document.getElementById('kg-relationships').textContent = data.total_relationships || 0;

        // Update quick stats
        const entitiesByType = data.entities_by_type || {};
        const productCount = entitiesByType.product || 0;
        const brandCount = entitiesByType.brand || 0;
        const categoryCount = entitiesByType.category || 0;
        const priceRangeCount = entitiesByType.price_range || 0;

        document.getElementById('count-products').textContent = productCount;
        document.getElementById('count-brands').textContent = brandCount;
        document.getElementById('count-categories').textContent = categoryCount;
        document.getElementById('count-price-ranges').textContent = priceRangeCount;

        console.log(`Quick stats: ${productCount} products, ${brandCount} brands, ${categoryCount} categories, ${priceRangeCount} price ranges`);
    } catch (error) {
        console.error('Could not load KG stats:', error);
    }
}

async function loadKnowledgeGraph() {
    const graphContainer = document.getElementById('kg-graph');
    const visibleCounter = document.getElementById('kg-visible-nodes');

    try {
        graphContainer.innerHTML = '<div style="display: flex; align-items: center; justify-content: center; height: 100%;"><div class="spinner"></div><span style="margin-left: 10px; color: var(--text-secondary);">Loading graph...</span></div>';

        console.log('Fetching knowledge graph...');
        const response = await fetch(`${CONFIG.apiUrl}/query/knowledge-graph?max_nodes=150`);
        kgState.graphData = await response.json();

        console.log('Graph data received:', kgState.graphData.nodes?.length || 0, 'nodes,', kgState.graphData.links?.length || 0, 'links');

        if (kgState.graphData.nodes && kgState.graphData.nodes.length > 0) {
            renderGraph(kgState.graphData);
            // Update visible nodes count after render
            const nodeCount = kgState.graphData.nodes.length;
            visibleCounter.textContent = nodeCount;
            console.log('Visible nodes set to:', nodeCount);
        } else {
            visibleCounter.textContent = 0;
            graphContainer.innerHTML = `
                <div style="display: flex; flex-direction: column; align-items: center; justify-content: center; height: 100%; color: var(--text-secondary);">
                    <span style="font-size: 3rem; margin-bottom: 1rem;">📊</span>
                    <p>No graph data available.</p>
                    <p style="font-size: 0.9rem; color: var(--text-muted);">Load products using Settings > Load E-commerce Data</p>
                </div>
            `;
        }
    } catch (error) {
        console.error('Could not load knowledge graph:', error);
        visibleCounter.textContent = 0;
        graphContainer.innerHTML = `
            <div style="display: flex; flex-direction: column; align-items: center; justify-content: center; height: 100%; color: var(--text-secondary);">
                <span style="font-size: 3rem; margin-bottom: 1rem;">⚠️</span>
                <p>Error loading graph.</p>
                <p style="font-size: 0.9rem; color: var(--text-muted);">Make sure the server is running.</p>
            </div>
        `;
    }
}

function renderGraph(data) {
    const container = document.getElementById('kg-graph');
    container.innerHTML = '<svg id="graph-svg"></svg>';

    const svg = d3.select('#graph-svg');
    const width = container.clientWidth;
    const height = container.clientHeight || 400;

    svg.attr('width', width).attr('height', height);

    // Color scheme for different entity types
    const colorScale = d3.scaleOrdinal()
        .domain(['product', 'brand', 'category', 'price_range', 'feature'])
        .range(['#6366f1', '#22c55e', '#f59e0b', '#ef4444', '#8b5cf6']);

    // Node size based on type
    const sizeScale = d3.scaleOrdinal()
        .domain(['product', 'brand', 'category', 'price_range', 'feature'])
        .range([6, 12, 14, 10, 5]);

    // Create force simulation
    kgState.simulation = d3.forceSimulation(data.nodes)
        .force('link', d3.forceLink(data.links).id(d => d.id).distance(80))
        .force('charge', d3.forceManyBody().strength(-100))
        .force('center', d3.forceCenter(width / 2, height / 2))
        .force('collision', d3.forceCollide().radius(d => sizeScale(d.type) + 5));

    // Create container for zoom/pan
    const g = svg.append('g');

    // Add zoom behavior and store it
    kgState.zoom = d3.zoom()
        .scaleExtent([0.1, 4])
        .on('zoom', (event) => {
            g.attr('transform', event.transform);
        });

    svg.call(kgState.zoom);

    // Create links
    const link = g.append('g')
        .selectAll('line')
        .data(data.links)
        .enter()
        .append('line')
        .attr('class', 'link')
        .attr('stroke', 'rgba(255, 255, 255, 0.15)')
        .attr('stroke-width', 1);

    // Create nodes
    const node = g.append('g')
        .selectAll('g')
        .data(data.nodes)
        .enter()
        .append('g')
        .attr('class', 'node')
        .call(d3.drag()
            .on('start', dragstarted)
            .on('drag', dragged)
            .on('end', dragended))
        .on('click', (event, d) => {
            console.log('Node clicked in graph:', d.name, d.type);
            event.stopPropagation();
            showNodeDetails(d);
        });

    // Add circles to nodes
    node.append('circle')
        .attr('r', d => sizeScale(d.type))
        .attr('fill', d => colorScale(d.type))
        .attr('stroke', 'rgba(255, 255, 255, 0.3)')
        .attr('stroke-width', 2);

    // Add labels to larger nodes (brands, categories)
    node.filter(d => ['brand', 'category', 'price_range'].includes(d.type))
        .append('text')
        .text(d => d.name.length > 15 ? d.name.substring(0, 12) + '...' : d.name)
        .attr('x', d => sizeScale(d.type) + 5)
        .attr('y', 4)
        .attr('fill', '#94a3b8')
        .attr('font-size', '10px');

    // Add tooltips
    node.append('title')
        .text(d => `${d.name} (${d.type})`);

    // Update positions on tick
    kgState.simulation.on('tick', () => {
        link
            .attr('x1', d => d.source.x)
            .attr('y1', d => d.source.y)
            .attr('x2', d => d.target.x)
            .attr('y2', d => d.target.y);

        node.attr('transform', d => `translate(${d.x}, ${d.y})`);
    });

    // Drag functions
    function dragstarted(event, d) {
        if (!event.active) kgState.simulation.alphaTarget(0.3).restart();
        d.fx = d.x;
        d.fy = d.y;
    }

    function dragged(event, d) {
        d.fx = event.x;
        d.fy = event.y;
    }

    function dragended(event, d) {
        if (!event.active) kgState.simulation.alphaTarget(0);
        d.fx = null;
        d.fy = null;
    }

    // Click on svg background to deselect
    svg.on('click', () => {
        resetGraphHighlight();
        hideEntityDetails();
    });
}

// Zoom control functions
function zoomGraph(factor) {
    const svg = d3.select('#graph-svg');
    if (kgState.zoom) {
        svg.transition().duration(300).call(kgState.zoom.scaleBy, factor);
    }
}

function resetZoom() {
    const svg = d3.select('#graph-svg');
    if (kgState.zoom) {
        svg.transition().duration(500).call(kgState.zoom.transform, d3.zoomIdentity);
    }
}

function fitGraph() {
    const svg = d3.select('#graph-svg');
    const container = document.getElementById('kg-graph');
    if (kgState.zoom && kgState.graphData) {
        const width = container.clientWidth;
        const height = container.clientHeight;
        svg.transition().duration(500).call(
            kgState.zoom.transform,
            d3.zoomIdentity.translate(width / 2, height / 2).scale(0.8)
        );
    }
}

async function showNodeDetails(nodeData) {
    console.log('Node clicked:', nodeData);
    kgState.selectedNode = nodeData;

    // Show the entity info panel and hide placeholder
    const placeholder = document.getElementById('details-placeholder');
    const entityInfo = document.getElementById('entity-info');

    if (placeholder) placeholder.style.display = 'none';
    if (entityInfo) entityInfo.style.display = 'block';

    // Update badge and name
    const badge = document.getElementById('entity-type-badge');
    if (badge) {
        badge.className = `entity-type-badge ${nodeData.type}`;
        badge.textContent = nodeData.type.replace('_', ' ').toUpperCase();
    }

    const nameEl = document.getElementById('entity-name');
    if (nameEl) nameEl.textContent = nodeData.name;

    // Highlight the selected node
    highlightSelectedNode(nodeData.id);

    try {
        console.log('Fetching entity details for:', nodeData.name);
        const response = await fetch(`${CONFIG.apiUrl}/query/knowledge-graph/entity/${encodeURIComponent(nodeData.name)}`);
        const data = await response.json();

        if (data.entity) {
            const entity = data.entity;
            const related = data.related || [];

            // Update properties with rich display for products
            const propsContainer = document.getElementById('entity-properties');
            if (entity.properties && Object.keys(entity.properties).length > 0) {
                let propsHtml = '';

                // Special handling for Product entities
                if (entity.type === 'product') {
                    const price = entity.properties.price;
                    const rating = entity.properties.rating;
                    const reviews = entity.properties.reviews_count;
                    const desc = entity.properties.description;
                    const features = entity.properties.features;
                    const brand = entity.properties.brand;
                    const category = entity.properties.category;

                    // Header stats (Price, Rating)
                    propsHtml += `<div style="display: flex; gap: 12px; margin-bottom: 12px;">`;

                    if (price) {
                        propsHtml += `
                            <div style="background: rgba(99, 102, 241, 0.1); padding: 8px 12px; border-radius: 8px; flex: 1; border: 1px solid rgba(99, 102, 241, 0.2);">
                                <div style="font-size: 0.75rem; color: var(--text-secondary); margin-bottom: 2px;">Price</div>
                                <div style="font-size: 1.1rem; font-weight: 700; color: #818cf8;">₹${price}</div>
                            </div>
                        `;
                    }

                    if (rating) {
                        const stars = '⭐'.repeat(Math.round(rating));
                        propsHtml += `
                            <div style="background: rgba(34, 197, 94, 0.1); padding: 8px 12px; border-radius: 8px; flex: 1; border: 1px solid rgba(34, 197, 94, 0.2);">
                                <div style="font-size: 0.75rem; color: var(--text-secondary); margin-bottom: 2px;">Rating</div>
                                <div style="font-size: 1.1rem; font-weight: 700; color: #4ade80;">${rating} <span style="font-size: 0.8rem;">${stars}</span></div>
                                ${reviews ? `<div style="font-size: 0.7rem; color: var(--text-muted);">${reviews} reviews</div>` : ''}
                            </div>
                        `;
                    }
                    propsHtml += `</div>`;

                    // Description
                    if (desc) {
                        propsHtml += `
                            <div style="margin-bottom: 12px;">
                                <div style="font-size: 0.8rem; font-weight: 600; color: var(--text-secondary); margin-bottom: 4px;">Description</div>
                                <div style="font-size: 0.9rem; line-height: 1.5; color: var(--text-primary); background: var(--bg-tertiary); padding: 10px; border-radius: 8px;">${desc}</div>
                            </div>
                        `;
                    }

                    // Features
                    if (features && Array.isArray(features) && features.length > 0) {
                        propsHtml += `
                            <div style="margin-bottom: 12px;">
                                <div style="font-size: 0.8rem; font-weight: 600; color: var(--text-secondary); margin-bottom: 4px;">Key Features</div>
                                <ul style="margin: 0; padding-left: 20px; color: var(--text-primary); font-size: 0.9rem;">
                                    ${features.map(f => `<li style="margin-bottom: 4px;">${f}</li>`).join('')}
                                </ul>
                            </div>
                        `;
                    }

                    // Other properties (Brand, Category) if not main display
                    if (brand || category) {
                        propsHtml += `<div style="display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 8px;">`;
                        if (brand) propsHtml += `<span style="background: var(--bg-tertiary); padding: 4px 8px; border-radius: 4px; font-size: 0.8rem; color: var(--text-secondary);">🏷️ ${brand}</span>`;
                        if (category) propsHtml += `<span style="background: var(--bg-tertiary); padding: 4px 8px; border-radius: 4px; font-size: 0.8rem; color: var(--text-secondary);">📁 ${category}</span>`;
                        propsHtml += `</div>`;
                    }

                } else {
                    // Standard display for non-product entities
                    for (const [key, value] of Object.entries(entity.properties)) {
                        if (value !== null && value !== undefined && value !== '') {
                            const displayKey = key.charAt(0).toUpperCase() + key.slice(1).replace(/_/g, ' ');
                            const displayValue = typeof value === 'number' && key.includes('price') ? `₹${value}` : value;
                            propsHtml += `
                                <div class="entity-property">
                                    <span class="property-label">${displayKey}</span>
                                    <span class="property-value">${displayValue}</span>
                                </div>
                            `;
                        }
                    }
                }

                propsContainer.innerHTML = propsHtml || '<div style="color: var(--text-muted);">No properties</div>';
                propsContainer.style.display = 'block';
            } else {
                propsContainer.style.display = 'none';
            }

            // Update relationships
            const relList = document.getElementById('relationships-list');
            if (related.length > 0) {
                let relHtml = '';
                related.slice(0, 15).forEach(rel => {
                    const direction = rel.direction === 'outgoing' ? '→' : '←';
                    relHtml += `
                        <div class="relationship-item" onclick="focusOnNode('${rel.entity.id}')">
                            <span class="relationship-type">${rel.relationship}</span>
                            <span class="relationship-name">${rel.entity.name}</span>
                            <span class="relationship-direction">${direction}</span>
                        </div>
                    `;
                });
                if (related.length > 15) {
                    relHtml += `<div style="color: var(--text-muted); padding: 8px; text-align: center;">...and ${related.length - 15} more</div>`;
                }
                relList.innerHTML = relHtml;
            } else {
                relList.innerHTML = '<div style="color: var(--text-muted); padding: 8px;">No relationships found</div>';
            }
        }
    } catch (error) {
        document.getElementById('entity-properties').innerHTML = '<div style="color: var(--text-muted);">Could not load properties</div>';
        document.getElementById('relationships-list').innerHTML = '<div style="color: var(--text-muted);">Could not load relationships</div>';
    }
}

function hideEntityDetails() {
    document.getElementById('details-placeholder').style.display = 'flex';
    document.getElementById('entity-info').style.display = 'none';
    kgState.selectedNode = null;
}

function highlightSelectedNode(nodeId) {
    const svg = d3.select('#graph-svg');

    // Remove previous selection
    svg.selectAll('.node').classed('selected', false);

    // Highlight selected node
    svg.selectAll('.node')
        .filter(d => d.id === nodeId)
        .classed('selected', true);

    // Dim other nodes slightly
    svg.selectAll('.node circle')
        .transition()
        .duration(300)
        .attr('opacity', d => d.id === nodeId ? 1 : 0.5);

    // Highlight connected links
    svg.selectAll('.link')
        .transition()
        .duration(300)
        .attr('opacity', d => {
            const sourceId = d.source.id || d.source;
            const targetId = d.target.id || d.target;
            return (sourceId === nodeId || targetId === nodeId) ? 0.8 : 0.1;
        })
        .attr('stroke', d => {
            const sourceId = d.source.id || d.source;
            const targetId = d.target.id || d.target;
            return (sourceId === nodeId || targetId === nodeId) ? '#6366f1' : 'rgba(255, 255, 255, 0.15)';
        });
}

const searchDebounce = debounce(async () => {
    await searchKnowledgeGraph();
}, 500);

async function searchKnowledgeGraph() {
    const query = document.getElementById('kg-search').value.toLowerCase().trim();
    const filter = document.getElementById('kg-filter').value;

    // If no search query and no filter, reset to normal view
    if (!query && !filter) {
        resetGraphHighlight();
        hideEntityDetails();
        return;
    }

    // If we don't have graph data, return
    if (!kgState.graphData || !kgState.graphData.nodes) {
        return;
    }

    // 1. Local Search
    let matchingNodes = kgState.graphData.nodes.filter(node => {
        if (filter && node.type !== filter) return false;
        if (query && !node.name.toLowerCase().includes(query)) return false;
        return true;
    });

    // 2. Server Search & Expansion (if query is substantial and local results are few)
    if (query && query.length >= 3 && matchingNodes.length < 5) {
        try {
            console.log('Searching backend for:', query);
            const response = await fetch(`${CONFIG.apiUrl}/query/knowledge-graph/search?q=${encodeURIComponent(query)}`);
            const data = await response.json();

            if (data.results && data.results.length > 0) {
                // Filter out nodes we already have
                const newNodes = data.results.filter(r => !kgState.graphData.nodes.find(n => n.id === r.id));

                if (newNodes.length > 0) {
                    console.log(`Found ${newNodes.length} new nodes from backend. Expanding graph...`);

                    // Add new nodes
                    newNodes.forEach(n => {
                        n.x = 0; n.y = 0; // Initialize pos
                        kgState.graphData.nodes.push(n);
                    });

                    // Fetch connections for top result to anchor it
                    const topResult = newNodes[0];
                    const detailsResp = await fetch(`${CONFIG.apiUrl}/query/knowledge-graph/entity/${encodeURIComponent(topResult.name)}`);
                    const details = await detailsResp.json();

                    if (details.related) {
                        details.related.forEach(rel => {
                            const targetId = rel.entity.id;
                            // Add related node if missing
                            if (!kgState.graphData.nodes.find(n => n.id === targetId)) {
                                rel.entity.x = 0; rel.entity.y = 0;
                                kgState.graphData.nodes.push(rel.entity);
                            }
                            // Add link
                            const sourceId = topResult.id;
                            // Check uniqueness
                            const linkExists = kgState.graphData.links.find(l =>
                                (l.source.id === sourceId && l.target.id === targetId) ||
                                (l.source.id === targetId && l.target.id === sourceId)
                            );
                            if (!linkExists) {
                                kgState.graphData.links.push({
                                    source: sourceId,
                                    target: targetId,
                                    type: rel.relationship
                                });
                            }
                        });
                    }

                    // Re-render graph to show new nodes
                    renderGraph(kgState.graphData);

                    // Re-calculate matching nodes after expansion
                    matchingNodes = kgState.graphData.nodes.filter(node => {
                        if (filter && node.type !== filter) return false;
                        if (query && !node.name.toLowerCase().includes(query)) return false;
                        return true;
                    });
                }
            }
        } catch (e) {
            console.warn('Backend search failed:', e);
        }
    }

    console.log(`Found ${matchingNodes.length} matching nodes for: "${query}"`);
    document.getElementById('kg-visible-nodes').textContent = matchingNodes.length;

    if (matchingNodes.length > 0) {
        highlightNodes(matchingNodes.map(n => n.id));
        showSearchResults(matchingNodes);
    } else {
        // No matches - dim everything
        const svg = d3.select('#graph-svg');
        svg.selectAll('.node circle').attr('opacity', 0.2);
        svg.selectAll('.link').attr('opacity', 0.1);

        document.getElementById('details-placeholder').style.display = 'none';
        document.getElementById('entity-info').style.display = 'block';
        document.getElementById('entity-type-badge').className = 'entity-type-badge';
        document.getElementById('entity-type-badge').textContent = 'No Results';
        document.getElementById('entity-name').textContent = 'No Matches Found';
        document.getElementById('entity-properties').innerHTML = `<div style="padding: 20px; text-align: center; color: var(--text-muted);">Could not find "${query}" in the knowledge graph.</div>`;
        document.getElementById('relationships-list').innerHTML = '';
    }
}

function showSearchResults(nodes) {
    document.getElementById('details-placeholder').style.display = 'none';
    document.getElementById('entity-info').style.display = 'block';

    document.getElementById('entity-type-badge').className = 'entity-type-badge';
    document.getElementById('entity-type-badge').textContent = 'Search Results';
    document.getElementById('entity-name').textContent = `${nodes.length} matches found`;

    // Show matching nodes as clickable items
    let html = '';
    nodes.slice(0, 15).forEach(node => {
        html += `
            <div class="relationship-item" onclick="focusOnNode('${node.id}')">
                <span class="relationship-type">${node.type}</span>
                <span class="relationship-name">${node.name}</span>
            </div>
        `;
    });

    if (nodes.length > 15) {
        html += `<div style="color: var(--text-muted); padding: 8px; text-align: center;">...and ${nodes.length - 15} more</div>`;
    }

    document.getElementById('entity-properties').style.display = 'none';
    document.getElementById('relationships-list').innerHTML = html;
}

function getTypeColor(type) {
    const colors = {
        'product': '#6366f1',
        'brand': '#22c55e',
        'category': '#f59e0b',
        'price_range': '#ef4444',
        'feature': '#8b5cf6'
    };
    return colors[type] || '#94a3b8';
}

function focusOnNode(nodeId) {
    if (!kgState.graphData) return;

    const node = kgState.graphData.nodes.find(n => n.id === nodeId);
    if (node) {
        // Highlight just this node
        highlightNodes([nodeId]);

        // Show node details
        showNodeDetails(node);
    }
}

function highlightNodes(nodeIds) {
    const svg = d3.select('#graph-svg');

    // Clear the auto-reset timer during active search
    if (kgState.highlightTimer) {
        clearTimeout(kgState.highlightTimer);
        kgState.highlightTimer = null;
    }

    // Add/remove classes for CSS-based animations
    svg.selectAll('.node')
        .classed('search-match', d => nodeIds.includes(d.id))
        .classed('search-dimmed', d => !nodeIds.includes(d.id));

    // Animate matching nodes - make them larger and fully visible
    svg.selectAll('.node circle')
        .transition()
        .duration(400)
        .attr('opacity', d => nodeIds.includes(d.id) ? 1 : 0.1)
        .attr('r', d => {
            const baseSize = getNodeSize(d.type);
            return nodeIds.includes(d.id) ? baseSize * 2 : baseSize * 0.7;
        })
        .attr('stroke-width', d => nodeIds.includes(d.id) ? 4 : 1)
        .attr('stroke', d => nodeIds.includes(d.id) ? '#fff' : 'rgba(255, 255, 255, 0.1)');

    // Add glow effect to matching nodes using filter
    svg.selectAll('.node')
        .filter(d => nodeIds.includes(d.id))
        .select('circle')
        .style('filter', 'drop-shadow(0 0 8px currentColor)');

    // Remove glow from non-matching nodes
    svg.selectAll('.node')
        .filter(d => !nodeIds.includes(d.id))
        .select('circle')
        .style('filter', 'none');

    // Dim node labels for non-matching nodes
    svg.selectAll('.node text')
        .transition()
        .duration(400)
        .attr('opacity', d => nodeIds.includes(d.id) ? 1 : 0.2);

    // Dim non-matching links
    svg.selectAll('.link')
        .transition()
        .duration(400)
        .attr('opacity', d => {
            const sourceMatch = nodeIds.includes(d.source.id || d.source);
            const targetMatch = nodeIds.includes(d.target.id || d.target);
            return (sourceMatch && targetMatch) ? 0.8 : (sourceMatch || targetMatch) ? 0.3 : 0.02;
        })
        .attr('stroke', d => {
            const sourceMatch = nodeIds.includes(d.source.id || d.source);
            const targetMatch = nodeIds.includes(d.target.id || d.target);
            return (sourceMatch && targetMatch) ? '#6366f1' : 'rgba(255, 255, 255, 0.15)';
        })
        .attr('stroke-width', d => {
            const sourceMatch = nodeIds.includes(d.source.id || d.source);
            const targetMatch = nodeIds.includes(d.target.id || d.target);
            return (sourceMatch && targetMatch) ? 2 : 1;
        });

    // Auto zoom to fit matching nodes if there are a reasonable number
    if (nodeIds.length > 0 && nodeIds.length <= 50 && kgState.zoom && kgState.graphData) {
        zoomToNodes(nodeIds);
    }
}

// Zoom to fit specific nodes in view
function zoomToNodes(nodeIds) {
    if (!kgState.graphData || !kgState.zoom) return;

    const matchingNodes = kgState.graphData.nodes.filter(n => nodeIds.includes(n.id));
    if (matchingNodes.length === 0) return;

    // Calculate bounding box of matching nodes
    const xs = matchingNodes.map(n => n.x);
    const ys = matchingNodes.map(n => n.y);

    const minX = Math.min(...xs);
    const maxX = Math.max(...xs);
    const minY = Math.min(...ys);
    const maxY = Math.max(...ys);

    const container = document.getElementById('kg-graph');
    const width = container.clientWidth;
    const height = container.clientHeight;

    // Calculate center and scale
    const centerX = (minX + maxX) / 2;
    const centerY = (minY + maxY) / 2;
    const dx = maxX - minX + 100; // Add padding
    const dy = maxY - minY + 100;
    const scale = Math.min(width / dx, height / dy, 2); // Max scale of 2

    // Apply zoom transform
    const svg = d3.select('#graph-svg');
    svg.transition()
        .duration(750)
        .call(
            kgState.zoom.transform,
            d3.zoomIdentity
                .translate(width / 2, height / 2)
                .scale(Math.max(0.5, Math.min(scale, 1.5)))
                .translate(-centerX, -centerY)
        );
}

function getNodeSize(type) {
    const sizes = {
        'product': 6,
        'brand': 12,
        'category': 14,
        'price_range': 10,
        'feature': 5
    };
    return sizes[type] || 6;
}

function resetGraphHighlight() {
    const svg = d3.select('#graph-svg');

    // Remove all state classes
    svg.selectAll('.node')
        .classed('selected', false)
        .classed('highlighted', false)
        .classed('dimmed', false)
        .classed('search-match', false)
        .classed('search-dimmed', false);

    // Reset node circles
    svg.selectAll('.node circle')
        .transition()
        .duration(400)
        .attr('opacity', 1)
        .attr('r', d => getNodeSize(d.type))
        .attr('stroke-width', 2)
        .attr('stroke', 'rgba(255, 255, 255, 0.3)')
        .style('filter', null);

    // Reset node text
    svg.selectAll('.node text')
        .transition()
        .duration(400)
        .attr('opacity', 1);

    // Reset links
    svg.selectAll('.link')
        .classed('highlighted', false)
        .classed('dimmed', false)
        .transition()
        .duration(400)
        .attr('opacity', 1)
        .attr('stroke', 'rgba(255, 255, 255, 0.15)')
        .attr('stroke-width', 1);

    // Update visible count to total
    if (kgState.graphData) {
        document.getElementById('kg-visible-nodes').textContent = kgState.graphData.nodes.length;
    }

    // Clear search inputs
    document.getElementById('kg-search').value = '';
    document.getElementById('kg-filter').value = '';

    // Reset zoom to default
    if (kgState.zoom) {
        const container = document.getElementById('kg-graph');
        const width = container.clientWidth;
        const height = container.clientHeight;
        svg.transition()
            .duration(500)
            .call(kgState.zoom.transform, d3.zoomIdentity);
    }
}

// Make functions globally available
window.focusOnNode = focusOnNode;
window.showNodeDetails = showNodeDetails;


// ==================== Evaluation Tab Functionality ====================
let evaluationState = {
    running: false,
    results: null,
    pollInterval: null
};

// Initialize Evaluation Tab
document.addEventListener('DOMContentLoaded', () => {
    initEvaluation();
});

function initEvaluation() {
    // Run Evaluation button
    const runBtn = document.getElementById('run-evaluation-btn');
    if (runBtn) {
        runBtn.addEventListener('click', runEvaluation);
    }

    // Test Single Question button
    const testSingleBtn = document.getElementById('test-single-btn');
    if (testSingleBtn) {
        testSingleBtn.addEventListener('click', testSingleQuestion);
    }

    // Export buttons
    const exportBtn = document.getElementById('export-results-btn');
    if (exportBtn) {
        exportBtn.addEventListener('click', exportResultsCSV);
    }

    const copyMdBtn = document.getElementById('copy-markdown-btn');
    if (copyMdBtn) {
        copyMdBtn.addEventListener('click', copyMarkdownReport);
    }

    // Load benchmark stats
    loadBenchmarkStats();
}

async function loadBenchmarkStats() {
    try {
        const response = await fetch(`${CONFIG.apiUrl}/evaluation/benchmark/stats`);
        if (response.ok) {
            const stats = await response.json();
            document.getElementById('eval-total-questions').textContent = stats.total_questions;
            document.getElementById('eval-categories').textContent = Object.keys(stats.by_category).length;
            document.getElementById('eval-graph-required').textContent = stats.graph_required;
        }
    } catch (error) {
        console.log('Benchmark stats not available:', error);
    }
}

async function runEvaluation() {
    const runBtn = document.getElementById('run-evaluation-btn');
    const progressDiv = document.getElementById('eval-progress');
    const progressFill = document.getElementById('eval-progress-fill');
    const progressText = document.getElementById('eval-progress-text');
    const resultsCard = document.getElementById('eval-results-card');

    // Get filters
    const category = document.getElementById('eval-category-filter').value;
    const limit = document.getElementById('eval-limit').value;

    // Disable button and show progress
    runBtn.disabled = true;
    runBtn.innerHTML = '<span class="btn-icon">⏳</span> Running...';
    progressDiv.style.display = 'block';
    progressFill.style.width = '0%';
    progressText.textContent = 'Starting evaluation...';
    resultsCard.style.display = 'none';

    try {
        // Start evaluation
        const startResponse = await fetch(`${CONFIG.apiUrl}/evaluation/run`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                modes: ['vector_only', 'graph_only', 'hybrid'],
                category: category || null,
                limit: limit ? parseInt(limit) : null
            })
        });

        if (!startResponse.ok) {
            const error = await startResponse.json();
            throw new Error(error.detail || 'Failed to start evaluation');
        }

        const startData = await startResponse.json();
        showToast(`Evaluation started: ${startData.total_evaluations} evaluations`, 'success');

        // Poll for progress
        evaluationState.running = true;
        evaluationState.pollInterval = setInterval(async () => {
            try {
                const progressResponse = await fetch(`${CONFIG.apiUrl}/evaluation/progress`);
                const progress = await progressResponse.json();

                const percentage = progress.percentage.toFixed(0);
                progressFill.style.width = `${percentage}%`;
                progressText.textContent = `${percentage}% - ${progress.current_question || 'Processing...'}`;

                if (!progress.running) {
                    clearInterval(evaluationState.pollInterval);
                    evaluationState.running = false;

                    // Fetch results
                    const resultsResponse = await fetch(`${CONFIG.apiUrl}/evaluation/results`);
                    const resultsData = await resultsResponse.json();

                    if (resultsData.status === 'complete') {
                        evaluationState.results = resultsData.results;
                        displayEvaluationResults(resultsData.results);
                        showToast('Evaluation complete!', 'success');
                    }

                    runBtn.disabled = false;
                    runBtn.innerHTML = '<span class="btn-icon">▶️</span> Run Evaluation';
                }
            } catch (e) {
                console.error('Progress poll error:', e);
            }
        }, 2000);

    } catch (error) {
        console.error('Evaluation error:', error);
        showToast(error.message, 'error');
        runBtn.disabled = false;
        runBtn.innerHTML = '<span class="btn-icon">▶️</span> Run Evaluation';
        progressDiv.style.display = 'none';
    }
}

function displayEvaluationResults(results) {
    const resultsCard = document.getElementById('eval-results-card');
    const resultsBody = document.getElementById('eval-results-body');

    if (!results || !results.aggregated || !results.aggregated.by_mode) {
        showToast('No results to display', 'warning');
        return;
    }

    const byMode = results.aggregated.by_mode;
    const modes = ['vector_only', 'graph_only', 'hybrid'];

    // Define metrics to display
    const metrics = [
        { key: 'avg_relevance_score', label: 'Avg. Relevance (1-5)', format: v => v.toFixed(2), higherBetter: true },
        { key: 'avg_accuracy_score', label: 'Avg. Accuracy', format: v => (v * 100).toFixed(1) + '%', higherBetter: true },
        { key: 'avg_keyword_coverage', label: 'Keyword Coverage', format: v => (v * 100).toFixed(1) + '%', higherBetter: true },
        { key: 'avg_entity_coverage', label: 'Entity Coverage', format: v => (v * 100).toFixed(1) + '%', higherBetter: true },
        { key: 'avg_response_time_ms', label: 'Avg. Response Time', format: v => v.toFixed(0) + 'ms', higherBetter: false },
        { key: 'avg_source_count', label: 'Avg. Sources Used', format: v => v.toFixed(1), higherBetter: true },
        { key: 'hallucination_rate', label: 'Hallucination Rate', format: v => (v * 100).toFixed(1) + '%', higherBetter: false }
    ];

    // Build table rows
    let html = '';
    metrics.forEach(metric => {
        const values = {};
        let bestMode = null;
        let bestValue = null;

        modes.forEach(mode => {
            if (byMode[mode]) {
                values[mode] = byMode[mode][metric.key] || 0;
                if (bestValue === null) {
                    bestValue = values[mode];
                    bestMode = mode;
                } else if (metric.higherBetter && values[mode] > bestValue) {
                    bestValue = values[mode];
                    bestMode = mode;
                } else if (!metric.higherBetter && values[mode] < bestValue) {
                    bestValue = values[mode];
                    bestMode = mode;
                }
            }
        });

        html += `<tr>
            <td>${metric.label}</td>
            ${modes.map(mode => {
            const value = values[mode];
            const isBest = mode === bestMode;
            const className = isBest ? 'best-value' : '';
            return `<td class="${className}">${metric.format(value)}</td>`;
        }).join('')}
            <td><span class="winner-badge">${bestMode === 'hybrid' ? '🏆 Hybrid' : bestMode}</span></td>
        </tr>`;
    });

    resultsBody.innerHTML = html;
    resultsCard.style.display = 'block';
}

async function testSingleQuestion() {
    const questionInput = document.getElementById('single-question-input');
    const groundTruthInput = document.getElementById('single-ground-truth');
    const resultsDiv = document.getElementById('single-test-results');
    const testBtn = document.getElementById('test-single-btn');

    const question = questionInput.value.trim();
    if (!question) {
        showToast('Please enter a question', 'warning');
        return;
    }

    // Show loading state
    testBtn.disabled = true;
    testBtn.innerHTML = '<span class="btn-icon">⏳</span> Testing...';
    resultsDiv.innerHTML = '<p style="color: var(--text-secondary); padding: 20px; text-align: center;">🔄 Running query across 3 modes... This may take 30-60 seconds.</p>';
    resultsDiv.style.display = 'block';

    try {
        const response = await fetch(`${CONFIG.apiUrl}/evaluation/single`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                question: question,
                ground_truth: groundTruthInput.value.trim() || null,
                modes: ['vector_only', 'graph_only', 'hybrid']
            })
        });

        if (!response.ok) {
            throw new Error('Failed to test question');
        }

        const data = await response.json();
        displaySingleTestResults(data);
        showToast('Test complete!', 'success');

    } catch (error) {
        console.error('Single test error:', error);
        showToast('Failed to test question', 'error');
        resultsDiv.innerHTML = '<p style="color: #ef4444; padding: 20px;">❌ Error: Could not complete test. Please try again.</p>';
    } finally {
        testBtn.disabled = false;
        testBtn.innerHTML = '<span class="btn-icon">🧪</span> Test Now';
    }
}

function displaySingleTestResults(data) {
    const resultsDiv = document.getElementById('single-test-results');
    const modes = ['vector_only', 'graph_only', 'hybrid'];
    const modeNames = {
        'vector_only': '📄 Vector-Only (Traditional RAG)',
        'graph_only': '🕸️ Graph-Only',
        'hybrid': '🔀 Hybrid (GraphRAG)'
    };

    let html = '';
    modes.forEach(mode => {
        const result = data.results[mode];
        if (result) {
            html += `
                <div class="single-result-mode">
                    <h4>${modeNames[mode]}</h4>
                    <div class="answer-text">${result.answer}</div>
                    <div class="result-meta">
                        <span>⏱️ <span class="value">${result.response_time_ms}ms</span></span>
                        <span>📚 <span class="value">${result.sources?.length || 0} sources</span></span>
                        ${result.graph_entities_found !== undefined ?
                    `<span>🕸️ <span class="value">${result.graph_entities_found} graph entities</span></span>` : ''}
                        ${result.relevance_score !== undefined ?
                    `<span>⭐ <span class="value">Relevance: ${result.relevance_score}/5</span></span>` : ''}
                    </div>
                </div>
            `;
        }
    });

    resultsDiv.innerHTML = html;
    resultsDiv.style.display = 'block';
}

function exportResultsCSV() {
    if (!evaluationState.results) {
        showToast('No results to export', 'warning');
        return;
    }

    const byMode = evaluationState.results.aggregated?.by_mode;
    if (!byMode) return;

    // Build CSV
    let csv = 'Metric,Vector-Only,Graph-Only,Hybrid\n';
    const metrics = [
        'avg_relevance_score',
        'avg_accuracy_score',
        'avg_keyword_coverage',
        'avg_entity_coverage',
        'avg_response_time_ms',
        'avg_source_count',
        'hallucination_rate'
    ];

    metrics.forEach(metric => {
        csv += `${metric},${byMode.vector_only?.[metric] || 0},${byMode.graph_only?.[metric] || 0},${byMode.hybrid?.[metric] || 0}\n`;
    });

    // Download
    const blob = new Blob([csv], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `ablation_results_${new Date().toISOString().slice(0, 10)}.csv`;
    a.click();
    URL.revokeObjectURL(url);

    showToast('Results exported to CSV', 'success');
}

async function copyMarkdownReport() {
    try {
        const response = await fetch(`${CONFIG.apiUrl}/evaluation/report`);
        if (!response.ok) {
            throw new Error('Report not available');
        }

        const data = await response.json();
        await navigator.clipboard.writeText(data.report);
        showToast('Markdown report copied to clipboard!', 'success');
    } catch (error) {
        console.error('Copy report error:', error);
        showToast('Failed to copy report', 'error');
    }
}
