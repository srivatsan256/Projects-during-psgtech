fetch('/query', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ username: 'user', query: 'test' })
})
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            console.log('Response:', data.data.response);
            console.log('Timestamp:', data.data.timestamp); // Use this numeric value
            // Convert to readable format if needed
            const date = new Date(data.data.timestamp * 1000); // Multiply by 1000 for milliseconds
            console.log('Formatted Date:', date.toLocaleString());
        } else {
            console.error('Error:', data.error);
        }
    })
    .catch(error => console.error('Fetch error:', error));