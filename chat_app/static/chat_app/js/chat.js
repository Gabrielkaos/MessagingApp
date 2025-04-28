function sendMessage() {
    var messageInput = document.getElementById('messageInput');
    var chatMessages = document.getElementById('chatMessages');

    var message = messageInput.value;
    if (message.trim() === '') {
        return;
    }

    var userMessage = document.createElement('div');
    userMessage.className = 'user-message chat-bubble';
    userMessage.innerText = message;
    chatMessages.appendChild(userMessage);


    fetch('/response', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/x-www-form-urlencoded',
            'X-CSRFToken': getCookie('csrftoken'),
        },
        body: JSON.stringify({ user_message: message }),
    })
    .then(response => response.json())
    .then(data => {
        var chatbotMessage = document.createElement('div');
        chatbotMessage.className = 'response-message chat-bubble';
        chatbotMessage.innerText = data.response;
        // alert(data.response);
        chatMessages.appendChild(chatbotMessage);

        messageInput.value = '';

        chatMessages.scrollTop = chatMessages.scrollHeight;
    })
    .catch(error => console.error('Error:', error));
}

function getCookie(name) {
    var cookieValue = null;
    if (document.cookie && document.cookie !== '') {
        var cookies = document.cookie.split(';');
        for (var i = 0; i < cookies.length; i++) {
            var cookie = cookies[i].trim();
            if (cookie.substring(0, name.length + 1) === (name + '=')) {
                cookieValue = decodeURIComponent(cookie.substring(name.length + 1));
                break;
            }
        }
    }
    return cookieValue;
}