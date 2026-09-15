import { useEffect, useRef, useState } from 'react'

const BASE_RETRY_DELAY = 1000
const MAX_RETRY_DELAY = 30000

/**
 * SSE(EventSource) 구독 훅. 연결이 끊기면 지수 백오프로 자동 재연결한다.
 * @param {string} url - 구독할 SSE 엔드포인트
 * @param {(violation: unknown) => void} onViolation - 'violation' 이벤트 수신 시 콜백
 * @returns {{ connected: boolean }} - 현재 연결 여부(재연결 대기 중이면 false)
 */
function useSSE(url, onViolation) {
  const callbackRef = useRef(onViolation)
  const [connected, setConnected] = useState(false)

  useEffect(() => {
    callbackRef.current = onViolation
  })

  useEffect(() => {
    let es
    let retryTimer
    let attempt = 0
    let stopped = false

    const connect = () => {
      es = new EventSource(url)

      es.onopen = () => {
        attempt = 0
        setConnected(true)
      }

      es.addEventListener('violation', (e) => {
        callbackRef.current(JSON.parse(e.data))
      })

      es.onerror = () => {
        es.close()
        setConnected(false)
        if (stopped) return
        const delay = Math.min(BASE_RETRY_DELAY * 2 ** attempt, MAX_RETRY_DELAY)
        attempt += 1
        retryTimer = setTimeout(connect, delay)
      }
    }

    connect()

    return () => {
      stopped = true
      clearTimeout(retryTimer)
      es.close()
    }
  }, [url])

  return { connected }
}

export default useSSE
