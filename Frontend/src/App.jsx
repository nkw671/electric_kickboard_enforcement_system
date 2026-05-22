import { Routes, Route } from 'react-router-dom'
import Layout from './components/Layout'
import MainPage from './pages/MainPage'
import ViolationsPage from './pages/ViolationsPage'
import './App.css'

function App() {
  return (
    <Layout>
      <Routes>
        <Route path="/" element={<MainPage />} />
        <Route path="/violations" element={<ViolationsPage />} />
      </Routes>
    </Layout>
  )
}

export default App
