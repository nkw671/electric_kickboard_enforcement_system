import { Routes, Route } from 'react-router-dom'
import Layout from './components/Layout'
import MainPage from './pages/MainPage'
import ViolationsPage from './pages/ViolationsPage'
import StatsPage from './pages/StatsPage'
import ZonePage from './pages/ZonePage'
import './App.css'

function App() {
  return (
    <Layout>
      <Routes>
        <Route path="/" element={<MainPage />} />
        <Route path="/violations" element={<ViolationsPage />} />
        <Route path="/stats" element={<StatsPage />} />
        <Route path="/zones" element={<ZonePage />} />
      </Routes>
    </Layout>
  )
}

export default App
