import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { BrowserRouter } from 'react-router-dom'
import App from './App.jsx'
import AccessGate from './components/AccessGate.jsx'
import { DEMO_MODE } from './demo/demoMode'
import './index.css'

function render() {
  createRoot(document.getElementById('root')).render(
    <StrictMode>
      <AccessGate>
        <BrowserRouter>
          <App />
        </BrowserRouter>
      </AccessGate>
    </StrictMode>,
  )
}

// 데모 모드일 때만 가짜 응답 코드를 불러온다(평소 번들에는 포함되지 않음).
if (DEMO_MODE) {
  import('./demo/installDemo').then(render)
} else {
  render()
}
