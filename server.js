const express = require('express');
const path = require('path');
const { createProxyMiddleware } = require('http-proxy-middleware');
const app = express();

// Proxy requests to the Flask backend (running on http://localhost:5000)
app.use('/predict', createProxyMiddleware({ 
  target: 'http://localhost:5000', 
  changeOrigin: true, 
  onProxyReq: (proxyReq, req, res) => {
    console.log(`Proxying request: ${req.method} ${req.url}`);
  }
}));
app.use('/login', createProxyMiddleware({ 
  target: 'http://localhost:5000', 
  changeOrigin: true, 
  onProxyReq: (proxyReq, req, res) => {
    console.log(`Proxying request: ${req.method} ${req.url}`);
  }
}));
app.use('/unmask_predictions', createProxyMiddleware({ 
  target: 'http://localhost:5000', 
  changeOrigin: true, 
  onProxyReq: (proxyReq, req, res) => {
    console.log(`Proxying request: ${req.method} ${req.url}`);
  }
}));
app.use('/logout', createProxyMiddleware({ 
  target: 'http://localhost:5000', 
  changeOrigin: true, 
  onProxyReq: (proxyReq, req, res) => {
    console.log(`Proxying request: ${req.method} ${req.url}`);
  }
}));

// Serve static files (e.g., your HTML, CSS, JS) from the 'public' directory
app.use(express.static(path.join(__dirname, 'public')));

// Handle the root route (index) and serve a simple message
app.get('/', (req, res) => {
  res.send('Welcome to the Node.js server, connected to Flask!');
});

// Start the server
app.listen(3000, () => {
  console.log('Server is running at http://localhost:5000');
});
