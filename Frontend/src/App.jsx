import { Routes, Route, useLocation } from 'react-router-dom'
import Layout from './components/Layout'
import ErrorBoundary from './components/ErrorBoundary'
import MainPage from './pages/MainPage'
import ViolationsPage from './pages/ViolationsPage'
import StatsPage from './pages/StatsPage'
import ZonePage from './pages/ZonePage'
import './App.css'

function App() {
  const location = useLocation()

  return (
    <Layout>
      {/* key로 pathname을 줘서 페이지 이동 시 에러 상태가 자동으로 리셋되게 함 */}
      <ErrorBoundary key={location.pathname}>
        <Routes>
          <Route path="/" element={<MainPage />} />
          <Route path="/violations" element={<ViolationsPage />} />
          <Route path="/stats" element={<StatsPage />} />
          <Route path="/zones" element={<ZonePage />} />
        </Routes>
      </ErrorBoundary>
    </Layout>
  )
}

export default App
