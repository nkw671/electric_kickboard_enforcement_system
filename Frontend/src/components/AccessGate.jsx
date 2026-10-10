import { useState } from 'react'
import styles from './AccessGate.module.css'

const SESSION_KEY = 'kb_access_granted'
const ACCESS_CODE = import.meta.env.VITE_ACCESS_CODE

// 발표/데모용 최소 접근 제어. 정식 로그인(JWT 등)이 아니라 env 변수 하나로
// 정한 접근 코드를 세션스토리지에 저장해두는 수준의 "출입 코드" 게이트.
// VITE_ACCESS_CODE가 설정 안 돼 있으면(로컬 개발 등) 게이트를 건너뛴다.
function AccessGate({ children }) {
  const [granted, setGranted] = useState(
    () => !ACCESS_CODE || sessionStorage.getItem(SESSION_KEY) === '1'
  )
  const [code, setCode] = useState('')
  const [error, setError] = useState(false)

  if (granted) return children

  const handleSubmit = (e) => {
    e.preventDefault()
    if (code === ACCESS_CODE) {
      sessionStorage.setItem(SESSION_KEY, '1')
      setGranted(true)
    } else {
      setError(true)
    }
  }

  return (
    <div className={styles.wrap}>
      <form className={styles.card} onSubmit={handleSubmit}>
        <svg className={styles.icon} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.75" strokeLinecap="round" strokeLinejoin="round">
          <rect x="3" y="11" width="18" height="10" rx="2" />
          <path d="M7 11V7a5 5 0 0 1 10 0v4" />
        </svg>
        <p className={styles.title}>전동킥보드 단속 시스템</p>
        <p className={styles.desc}>접근 코드를 입력해주세요</p>
        <input
          className={styles.input}
          type="password"
          value={code}
          autoFocus
          onChange={(e) => { setCode(e.target.value); setError(false) }}
          placeholder="접근 코드"
        />
        {error && <p className={styles.error}>접근 코드가 올바르지 않습니다.</p>}
        <button className={styles.submitBtn} type="submit">입장</button>
      </form>
    </div>
  )
}

export default AccessGate
