const queryForm = document.getElementById('query-form');
const questionInput = document.getElementById('question');
const resultContainer = document.getElementById('result');
const historyPanel = document.getElementById('history-panel');
const healthStatus = document.getElementById('health-status');
const queryCount = document.getElementById('query-count');
const clearButton = document.getElementById('clear-button');
const useSavedButton = document.getElementById('use-saved-button');
const rebuildButton = document.getElementById('rebuild-button');
const indexMessage = document.getElementById('index-message');

let historyEntries = [];
let indexReady = false;

const spinnerMarkup = () => `
    <div class="response-card loading-card">
        <div class="loader"></div>
        <div>Looking up the best answer from GST documents...</div>
    </div>
`;

const updateHealth = (data) => {
    if (!data || data.status !== 'ok') {
        healthStatus.textContent = 'Health check failed';
        healthStatus.classList.add('badge-error');
        return;
    }
    healthStatus.textContent = `Ready · ${data.documents_in_store} chunks indexed`;
    healthStatus.classList.remove('badge-error');
};

const updateQueryCount = () => {
    queryCount.textContent = `${historyEntries.length} query${historyEntries.length === 1 ? '' : 'ies'} completed`;
};

const setQueryEnabled = (enabled) => {
    questionInput.disabled = !enabled;
    queryForm.querySelector('button[type="submit"]').disabled = !enabled;
};

const updateIndexControls = (status) => {
    if (!status || status.status !== 'ok') {
        indexMessage.textContent = 'Unable to determine saved index status.';
        useSavedButton.disabled = true;
        rebuildButton.disabled = true;
        setQueryEnabled(false);
        return;
    }

    if (status.saved_index_exists) {
        useSavedButton.disabled = false;
        rebuildButton.disabled = false;

        if (status.new_data_detected) {
            indexMessage.textContent = `New or changed PDF files detected. Use the saved index or rebuild from current data.`;
            indexReady = true;
        } else {
            indexMessage.textContent = `Saved index loaded. ${status.documents_in_store} chunks are available. Rebuild only if you added or updated PDF documents.`;
            indexReady = true;
        }
    } else if (status.pdf_count > 0) {
        indexMessage.textContent = `No saved vector store available yet. Rebuild the index from your current PDF files to start querying.`;
        useSavedButton.disabled = true;
        rebuildButton.disabled = false;
        indexReady = false;
    } else {
        indexMessage.textContent = `No PDF documents detected in the data folder. Add PDFs and rebuild the index first.`;
        useSavedButton.disabled = true;
        rebuildButton.disabled = true;
        indexReady = false;
    }

    setQueryEnabled(indexReady);
};

const renderSources = (sources = []) => {
    if (!sources.length) {
        return '<div class="source-card"><p>No supporting sources were returned.</p></div>';
    }
    return sources.map(src => `
        <div class="source-card">
            <div class="source-row">
                <span class="source-title">${src.source}</span>
                <span class="badge">Score ${src.score.toFixed(3)}</span>
            </div>
            <div class="source-metadata">Page: ${src.page}</div>
            <p>${src.preview}</p>
        </div>
    `).join('');
};

const renderResult = (payload, query) => {
    const answer = payload.Answer || 'No answer returned.';
    const confidence = payload.Confidence_Score ?? 0;
    const sourcesHtml = renderSources(payload.Sources);

    resultContainer.innerHTML = `
        <div class="response-card">
            <div class="response-header">
                <div>
                    <h2>Answer</h2>
                    <p class="query-summary">${query}</p>
                </div>
                <span class="badge">Confidence ${confidence.toFixed(3)}</span>
            </div>
            <p class="answer-body">${answer}</p>
        </div>
        <div class="source-grid">${sourcesHtml}</div>
    `;
};

const addHistoryEntry = (query, payload) => {
    historyEntries.unshift({ query, payload, timestamp: new Date().toLocaleString() });
    historyPanel.innerHTML = historyEntries.map((entry, index) => `
        <button class="history-item" type="button" data-index="${index}">
            <span>${entry.query}</span>
            <small>${entry.timestamp}</small>
        </button>
    `).join('');
    updateQueryCount();
};

const handleHistoryClick = (event) => {
    const button = event.target.closest('button.history-item');
    if (!button) return;
    const index = Number(button.dataset.index);
    const entry = historyEntries[index];
    if (!entry) return;
    renderResult(entry.payload, entry.query);
};

const showError = (message) => {
    resultContainer.innerHTML = `
        <div class="response-card error-card">
            <h2>Something went wrong</h2>
            <p>${message}</p>
        </div>
    `;
};

const sendQuery = async (question) => {
    resultContainer.innerHTML = spinnerMarkup();

    const data = new FormData();
    data.append('question', question);

    try {
        const response = await fetch('/api/query', {
            method: 'POST',
            body: data,
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.error || 'Unable to fetch answer.');
        }

        const payload = await response.json();
        renderResult(payload, question);
        addHistoryEntry(question, payload);
    } catch (error) {
        showError(error.message || 'Request failed.');
    }
};

const manageIndexAction = async (action) => {
    indexMessage.textContent = 'Processing index action...';
    useSavedButton.disabled = true;
    rebuildButton.disabled = true;

    const data = new FormData();
    data.append('action', action);

    try {
        const response = await fetch('/api/index', {
            method: 'POST',
            body: data,
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.error || 'Index action failed.');
        }

        const payload = await response.json();
        indexMessage.textContent = payload.message || 'Index action completed successfully.';
        await loadStatus();
        return payload;
    } catch (error) {
        indexMessage.textContent = error.message || 'Index action failed.';
        useSavedButton.disabled = action !== 'use_saved';
        rebuildButton.disabled = false;
    }
};

const loadStatus = async () => {
    try {
        const response = await fetch('/api/status');
        const data = await response.json();
        updateHealth(data);
        updateIndexControls(data);
    } catch (error) {
        healthStatus.textContent = 'Health unavailable';
        healthStatus.classList.add('badge-error');
        indexMessage.textContent = 'Unable to load index status.';
        useSavedButton.disabled = true;
        rebuildButton.disabled = true;
        setQueryEnabled(false);
    }
};

queryForm.addEventListener('submit', async (event) => {
    event.preventDefault();
    const question = questionInput.value.trim();

    if (!question) {
        showError('Please enter a GST query.');
        return;
    }

    await sendQuery(question);
});

useSavedButton.addEventListener('click', () => manageIndexAction('use_saved'));
rebuildButton.addEventListener('click', () => manageIndexAction('rebuild'));
historyPanel.addEventListener('click', handleHistoryClick);
clearButton.addEventListener('click', () => {
    questionInput.value = '';
    resultContainer.innerHTML = '';
});

window.addEventListener('load', () => {
    loadStatus();
    updateQueryCount();
});
