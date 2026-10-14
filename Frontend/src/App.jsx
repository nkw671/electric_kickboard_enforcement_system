import { lazy, Suspense } from 'react'
import { Routes, Route, useLocation } from 'react-router-dom'
import Layout from './components/Layout'
import ErrorBoundary from './components/ErrorBoundary'
import MainPage from './pages/MainPage'
import ViolationsPage from './pages/ViolationsPage'
import ZonePage from './pages/ZonePage'
import './App.css'

// 통계 페이지는 차트 라이브러리(recharts)가 커서 첫 화면 번들에서 분리해 필요할 때 불러온다.
const StatsPage = lazy(() => import('./pages/StatsPage'))

function App() {
  const location = useLocation()

  return (
    <Layout>
      {/* key로 pathname을 줘서 페이지 이동 시 에러 상태가 자동으로 리셋되게 함 */}
      <ErrorBoundary key={location.pathname}>
        <Suspense fallback={null}>
          <Routes>
            <Route path="/" element={<MainPage />} />
            <Route path="/violations" element={<ViolationsPage />} />
            <Route path="/stats" element={<StatsPage />} />
            <Route path="/zones" element={<ZonePage />} />
          </Routes>
        </Suspense>
      </ErrorBoundary>
    </Layout>
  )
}

export default App
