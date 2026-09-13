import React from 'react';
import ReactDOM from 'react-dom/client';
import { BrowserRouter } from 'react-router-dom';
import App from './App';
import './styles.css';

ReactDOM.createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    <BrowserRouter>
      <App />
    </BrowserRouter>
  </React.StrictMode>
);
<div className="login-left">
    <img
        src="/image.png"
        alt="InsurePulse"
        className="login-image"
    />

    <h1>InsurePulse XAI</h1>
    <p>AI-powered insurance underwriting.</p>
</div>