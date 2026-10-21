/**
 * 백엔드/AI 서버 주소.
 * - 로컬 개발: 환경 변수를 비워두면 Vite 프록시(/api → :8080, /ai → :8000)를 탄다.
 * - 배포(Vercel): VITE_API_BASE=https://api.kickdan.com, VITE_AI_BASE=https://ai.kickdan.com
 */
const API_BASE = (import.meta.env.VITE_API_BASE || '').replace(/\/+$/, '')
const AI_BASE = (import.meta.env.VITE_AI_BASE || '/ai').replace(/\/+$/, '')

/** 백엔드 경로('/api/...')를 실제 요청 주소로 변환 */
export const apiUrl = (path) => `${API_BASE}${path}`

/** AI 서버 경로('/zones', '/video/stream' 등)를 실제 요청 주소로 변환 */
export const aiUrl = (path) => `${AI_BASE}${path}`
