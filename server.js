const express = require('express');
const path = require('path');
const { createProxyMiddleware } = require('http-proxy-middleware');
const app = express();

// Proxy requests to the Flask backend (running on http://localhost:5000)
app.use('/predict', createProxyMiddleware({ target: 'http://localhost:5000', changeOrigin: true }));
app.use('/login', createProxyMiddleware({ target: 'http://localhost:5000', changeOrigin: true }));
app.use('/unmask_predictions', createProxyMiddleware({ target: 'http://localhost:5000', changeOrigin: true }));
app.use('/logout', createProxyMiddleware({ target: 'http://localhost:5000', changeOrigin: true }));

// Serve static files (e.g., your HTML, CSS, JS)
app.use(express.static(path.join(__dirname, 'public')));

// Handle all other routes (i.e., root, about, etc.)
app.get('/', (req, res) => {
  res.send('Welcome to the Node.js server, connected to Flask!');
});

app.listen(5000, () => {
  console.log('Server is running at http://localhost:5000');
});
