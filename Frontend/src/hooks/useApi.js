import { useState, useEffect, useRef } from 'react'

/**
 * 주기적으로 API를 폴링하는 훅
 * @param {string} url - 폴링할 엔드포인트
 * @param {number} interval - 폴링 간격(ms), 기본 3000
 * @returns {{ data, loading, error, connected }}
 */
function useApi(url, interval = 3000) {
  const [data, setData] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [connected, setConnected] = useState(false)
  const abortRef = useRef(null)

  useEffect(() => {
    let timerId

    async function fetchData() {
      abortRef.current = new AbortController()
      try {
        const res = await fetch(url, { signal: abortRef.current.signal })
        if (!res.ok) throw new Error(`HTTP ${res.status}`)
        const json = await res.json()
        setData(json)
        setConnected(true)
        setError(null)
      } catch (err) {
        if (err.name === 'AbortError') return
        setConnected(false)
        setError(err.message)
      } finally {
        setLoading(false)
      }
    }

    fetchData()
    timerId = setInterval(fetchData, interval)

    return () => {
      clearInterval(timerId)
      abortRef.current?.abort()
    }
  }, [url, interval])

  return { data, loading, error, connected }
}

export default useApi
