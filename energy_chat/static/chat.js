/**
 * Chat Interface JavaScript
 * Handles UI interactions and API communication
 */

const messagesContainer = document.getElementById('messages-container');
const promptInput = document.getElementById('prompt-input');
const sendButton = document.getElementById('send-button');
const modelSelect = document.getElementById('model-select');

let history = [];
let isLoading = false;

// Load available models
async function loadModels() {
    const statusDiv = document.querySelector('.header-status');
    statusDiv.textContent = 'Loading...';
    
    try {
        // Try root endpoint first, then api
        let response = await fetch('/models');
        if (!response.ok) {
            console.log("Root /models failed, trying /api/models");
            response = await fetch('/api/models');
        }

        if (response.ok) {
            const data = await response.json();
            if (data.models && data.models.length > 0) {
                modelSelect.innerHTML = '';
                data.models.forEach(model => {
                    const option = document.createElement('option');
                    option.value = model.id;
                    option.textContent = model.name;
                    modelSelect.appendChild(option);
                });
                statusDiv.textContent = 'Online';
            } else {
                modelSelect.innerHTML = '<option>No models found</option>';
                statusDiv.textContent = 'Error';
            }
        } else {
            modelSelect.innerHTML = '<option>Load failed</option>';
            statusDiv.textContent = 'API Error';
            appendMessage('assistant', 'Failed to load models. Please refresh.', true);
        }
    } catch (error) {
        console.error('Failed to load models:', error);
        modelSelect.innerHTML = '<option>Connection Error</option>';
        statusDiv.textContent = 'Offline';
        appendMessage('assistant', `Connection error: ${error.message}. Is the server running?`, true);
    }
}

// Initialize
loadModels();

// Auto-resize input
promptInput.addEventListener('input', () => {
    promptInput.style.height = 'auto';
    promptInput.style.height = Math.min(promptInput.scrollHeight, 100) + 'px';
});

/**
 * Clear empty state from messages container
 */
function clearEmptyState() {
    const emptyState = messagesContainer.querySelector('.empty-state');
    if (emptyState) {
        emptyState.remove();
    }
}

/**
 * Append a message to the chat
 * @param {string} role - 'user' or 'assistant'
 * @param {string} content - Message content
 * @param {boolean} isError - Whether this is an error message
 */
function appendMessage(role, content, isError = false) {
    clearEmptyState();
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${role}`;

    const bubble = document.createElement('div');
    if (isError) {
        bubble.className = 'error-message';
    } else {
        bubble.className = 'message-bubble';
    }
    bubble.innerHTML = content.replace(/\n/g, '<br>');
    messageDiv.appendChild(bubble);
    messagesContainer.appendChild(messageDiv);
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
}

/**
 * Show typing indicator animation
 */
function showTypingIndicator() {
    clearEmptyState();
    const messageDiv = document.createElement('div');
    messageDiv.className = 'message assistant';
    messageDiv.id = 'typing-indicator';
    
    const typingDiv = document.createElement('div');
    typingDiv.className = 'typing-indicator';
    typingDiv.innerHTML = '<span></span><span></span><span></span>';
    messageDiv.appendChild(typingDiv);
    messagesContainer.appendChild(messageDiv);
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
}

/**
 * Remove typing indicator animation
 */
function removeTypingIndicator() {
    const typing = document.getElementById('typing-indicator');
    if (typing) {
        typing.remove();
    }
}

/**
 * Send message to the API
 */
async function sendMessage() {
    const prompt = promptInput.value.trim();
    if (!prompt || isLoading) return;

    isLoading = true;
    sendButton.disabled = true;

    appendMessage('user', prompt);
    promptInput.value = '';
    promptInput.style.height = 'auto';
    showTypingIndicator();

    try {
        const response = await fetch('/api/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ 
                prompt: prompt, 
                history: history,
                model: modelSelect.value
            })
        });

        removeTypingIndicator();
        const data = await response.json();

        if (response.ok && data.status === 'success') {
            const answer = data.answer;
            appendMessage('assistant', answer);
            history.push({"role": "user", "content": prompt});
            history.push({"role": "assistant", "content": answer});
        } else {
            const errorMsg = data.detail || 'An error occurred';
            appendMessage('assistant', errorMsg, true);
        }
    } catch (error) {
        removeTypingIndicator();
        appendMessage('assistant', `Connection error: ${error.message}`, true);
    } finally {
        isLoading = false;
        sendButton.disabled = false;
        promptInput.focus();
    }
}

// Event listeners
sendButton.addEventListener('click', sendMessage);
promptInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
    }
});

// Focus input on load
promptInput.focus();
