import { useEffect, useRef } from 'react'

function useSSE(url, onViolation) {
  const callbackRef = useRef(onViolation)

  useEffect(() => {
    callbackRef.current = onViolation
  })

  useEffect(() => {
    const es = new EventSource(url)
    es.addEventListener('violation', (e) => {
      callbackRef.current(JSON.parse(e.data))
    })
    es.onerror = () => es.close()
    return () => es.close()
  }, [url])
}

export default useSSE
