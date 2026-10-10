import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { BrowserRouter } from 'react-router-dom'
import App from './App.jsx'
import AccessGate from './components/AccessGate.jsx'
import './index.css'

createRoot(document.getElementById('root')).render(
  <StrictMode>
    <AccessGate>
      <BrowserRouter>
        <App />
      </BrowserRouter>
    </AccessGate>
  </StrictMode>,
)
