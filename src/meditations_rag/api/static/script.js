document.addEventListener('DOMContentLoaded', () => {
    const queryInput = document.getElementById('query-input');
    const askBtn = document.getElementById('ask-btn');
    const statusContainer = document.getElementById('status-container');
    const resultContainer = document.getElementById('result-container');
    const answerText = document.getElementById('answer-text');
    const sourcesList = document.getElementById('sources-list');
    const agenticToggle = document.getElementById('agentic-toggle');
    const modeLabel = document.getElementById('mode-label');
    
    // Steps
    const stepRetrieval = document.getElementById('step-retrieval');
    const stepGeneration = document.getElementById('step-generation');
    const stepController = document.getElementById('step-controller');
    const stepEvaluator = document.getElementById('step-evaluator');

    // Toggle Mode
    agenticToggle.addEventListener('change', () => {
        const isAgentic = agenticToggle.checked;
        modeLabel.textContent = isAgentic ? "Agentic RAG" : "Standard RAG";
        modeLabel.style.color = isAgentic ? "#8e44ad" : "var(--primary)";
        
        // Show/Hide relevant steps
        if (isAgentic) {
            stepController.classList.remove('hidden');
            stepEvaluator.classList.remove('hidden');
        } else {
            stepController.classList.add('hidden');
            stepEvaluator.classList.add('hidden');
        }
    });

    askBtn.addEventListener('click', handleQuery);
    queryInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') handleQuery();
    });

    async function handleQuery() {
        const query = queryInput.value.trim();
        if (!query) return;

        const isAgentic = agenticToggle.checked;
        const endpoint = isAgentic ? '/agentic/stream-query' : '/rag/stream-query';

        // Reset UI
        askBtn.disabled = true;
        statusContainer.classList.remove('hidden');
        resultContainer.classList.add('hidden');
        
        resetSteps();
        
        // Initial Status
        if (isAgentic) {
            setStepStatus(stepController, 'active');
        } else {
            setStepStatus(stepRetrieval, 'active');
        }

        try {
            const encodedQuery = encodeURIComponent(query);
            const eventSource = new EventSource(`${endpoint}?query=${encodedQuery}`);

            eventSource.onmessage = (event) => {
                const data = JSON.parse(event.data);
                
                if (data.type === 'status') {
                    handleStatusUpdate(data, isAgentic);
                } else if (data.type === 'result') {
                    completeAllSteps(isAgentic);
                    displayResult(data.data);
                    eventSource.close();
                    askBtn.disabled = false;
                } else if (data.type === 'error') {
                    alert('Error: ' + data.message);
                    eventSource.close();
                    askBtn.disabled = false;
                }
            };

            eventSource.onerror = (err) => {
                console.error('EventSource failed:', err);
                eventSource.close();
                askBtn.disabled = false;
            };

        } catch (error) {
            console.error('Error:', error);
            alert('Failed to send query');
            askBtn.disabled = false;
        }
    }

    function handleStatusUpdate(data, isAgentic) {
        // Standard RAG
        if (!isAgentic) {
            if (data.step === 'retrieval_start') {
                setStepStatus(stepRetrieval, 'active');
            } else if (data.step === 'retrieval_done') {
                setStepStatus(stepRetrieval, 'completed');
                setStepStatus(stepGeneration, 'active');
            } else if (data.step === 'generation_start') {
                setStepStatus(stepGeneration, 'active');
            }
            return;
        }

        // Agentic RAG
        // Steps: controller -> retriever -> generator -> evaluator
        const step = data.step;
        
        if (step === 'controller') {
            setStepStatus(stepController, 'active');
            // If we go back to controller, previous steps might be done
        } else if (step === 'retriever') {
            setStepStatus(stepController, 'completed');
            setStepStatus(stepRetrieval, 'active');
        } else if (step === 'generator') {
            setStepStatus(stepRetrieval, 'completed');
            setStepStatus(stepController, 'completed'); // Ensure controller is marked done
            setStepStatus(stepGeneration, 'active');
        } else if (step === 'evaluator') {
            setStepStatus(stepGeneration, 'completed');
            setStepStatus(stepEvaluator, 'active');
        }
    }

    function resetSteps() {
        [stepRetrieval, stepGeneration, stepController, stepEvaluator].forEach(step => {
            if (step) {
                step.classList.remove('active', 'completed');
                step.querySelector('.step-icon').textContent = 'Wait';
            }
        });
    }

    function setStepStatus(element, status) {
        if (!element) return;
        element.classList.remove('active', 'completed');
        element.classList.add(status);
        
        const icon = element.querySelector('.step-icon');
        if (status === 'active') icon.textContent = '...';
        if (status === 'completed') icon.textContent = '✓';
    }

    function completeAllSteps(isAgentic) {
        if (isAgentic) {
            setStepStatus(stepController, 'completed');
            setStepStatus(stepRetrieval, 'completed');
            setStepStatus(stepGeneration, 'completed');
            setStepStatus(stepEvaluator, 'completed');
        } else {
            setStepStatus(stepRetrieval, 'completed');
            setStepStatus(stepGeneration, 'completed');
        }
    }

    function displayResult(data) {
        resultContainer.classList.remove('hidden');
        answerText.textContent = data.answer;
        
        sourcesList.innerHTML = '';
        if (data.sources && data.sources.length > 0) {
            data.sources.forEach(source => {
                const div = document.createElement('div');
                div.className = 'source-item';
                
                const meta = [];
                if (source.book) meta.push(`Book ${source.book}`);
                if (source.chapter) meta.push(`Chapter ${source.chapter}`);
                if (source.score) meta.push(`Score: ${source.score}`);
                
                div.innerHTML = `
                    <div class="source-meta">${meta.join(' • ')}</div>
                    <div class="source-text">"${source.text}"</div>
                `;
                sourcesList.appendChild(div);
            });
        } else {
            sourcesList.innerHTML = '<p>No sources found.</p>';
        }
    }
});