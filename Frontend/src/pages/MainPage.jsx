import { useState, useRef, useEffect } from 'react'
import { TYPE_COLOR, FEED_MAX_COUNT } from '../constants'
import { extractTime } from '../utils'
import useApi from '../hooks/useApi'
import useSSE from '../hooks/useSSE'
import styles from './MainPage.module.css'

function MainPage() {
  const { data: violations, loading, error, connected } = useApi('/api/violations?limit=10')
  const { data: stats } = useApi('/api/stats')

  const [toast, setToast] = useState(null)
  const toastTimerRef = useRef(null)
  const [currentTime, setCurrentTime] = useState('')
  const [streamError, setStreamError] = useState(false)
  const [streamKey, setStreamKey] = useState(0)

  const retryStream = () => {
    setStreamError(false)
    setStreamKey(k => k + 1)
  }

  useEffect(() => {
    const tick = () => {
      const now = new Date()
      const y = now.getFullYear()
      const m = now.getMonth() + 1
      const d = now.getDate()
      const h = now.getHours()
      const min = String(now.getMinutes()).padStart(2, '0')
      const sec = String(now.getSeconds()).padStart(2, '0')
      const ampm = h < 12 ? '오전' : '오후'
      const h12 = h % 12 || 12
      setCurrentTime(`${y}. ${m}. ${d}. ${ampm} ${h12}:${min}:${sec}`)
    }
    tick()
    const id = setInterval(tick, 1000)
    return () => clearInterval(id)
  }, [])

  useSSE('/api/stream', (violation) => {
    if (toastTimerRef.current) clearTimeout(toastTimerRef.current)
    setToast(violation)
    toastTimerRef.current = setTimeout(() => setToast(null), 4000)
  })

  if (loading || !stats) return <div className={styles.status}>불러오는 중...</div>
  if (error) return <div className={styles.statusError}>서버에 연결할 수 없습니다.</div>

  return (
    <div className={styles.page}>
      {toast && (
        <div className={styles.toast}>
          <span className={styles.toastLabel}>새 위반 감지</span>
          <div className={styles.toastContent}>
            <span
              className={styles.toastBadge}
              style={{ backgroundColor: TYPE_COLOR[toast.type] || '#64748b' }}
            >
              {toast.type}
            </span>
            <span className={styles.toastCamera}>{toast.camera}</span>
            <span className={styles.toastConf}>{toast.confidence}%</span>
          </div>
        </div>
      )}

      <h2 className={styles.pageTitle}>실시간 단속 모니터링</h2>
      <div className={styles.contentGrid}>
        {/* 영상 스트림 */}
        <div className={styles.streamBox}>
          <div className={styles.streamHeader}>
            <div className={styles.streamHeaderLeft}>
              <svg className={styles.camIcon} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <polygon points="23 7 16 12 23 17 23 7"/>
                <rect x="1" y="5" width="15" height="14" rx="2" ry="2"/>
              </svg>
              <div className={styles.streamInfo}>
                <div className={styles.streamCamName}>CAM-01</div>
                <div className={styles.streamCamSub}>실시간 단속 카메라</div>
              </div>
            </div>
            <div className={styles.streamBadges}>
              {connected && (
                <span className={styles.recordBadge}>
                  <span className={styles.recordDot} /> 녹화 중
                </span>
              )}
              <span className={styles.liveBadge}>실시간</span>
            </div>
          </div>
          <div className={styles.streamBody}>
            {!streamError ? (
              <img
                src={`/ai/video/stream?t=${streamKey}`}
                className={styles.stream}
                alt="stream"
                onError={() => setStreamError(true)}
              />
            ) : (
              <button className={styles.streamPlaceholder} onClick={retryStream}>
                영상 스트림 영역<br />
                <small>AI 서버 연결 후 활성화 · 클릭하여 재연결</small>
              </button>
            )}
            {currentTime && (
              <span className={styles.streamTimestamp}>{currentTime}</span>
            )}
          </div>
        </div>

        {/* 우측 사이드바 */}
        <div className={styles.sidebar}>
          {/* 시스템 상태 */}
          <div className={styles.sideCard}>
            <div className={styles.sideCardHeader}>
              <div className={styles.sideCardTitleRow}>
                <svg className={styles.shieldIcon} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>
                </svg>
                <span className={styles.sideCardTitle}>시스템 상태</span>
              </div>
              <span className={styles.statusBadge}>정상 작동</span>
            </div>
            <div className={styles.miniStats}>
              <div className={styles.miniStat}>
                <span className={styles.miniVal} style={{ color: '#002855' }}>{stats.total}</span>
                <span className={styles.miniLbl}>오늘 총 위반</span>
              </div>
              <div className={styles.miniStat}>
                <span className={styles.miniVal} style={{ color: '#2563eb' }}>{stats.helmet}</span>
                <span className={styles.miniLbl}>헬멧 미착용</span>
              </div>
              <div className={styles.miniStat}>
                <span className={styles.miniVal} style={{ color: '#2563eb' }}>{stats.sidewalk}</span>
                <span className={styles.miniLbl}>인도 주행</span>
              </div>
              <div className={styles.miniStat}>
                <span className={styles.miniVal} style={{ color: '#2563eb' }}>{stats.multiRider}</span>
                <span className={styles.miniLbl}>다인 탑승</span>
              </div>
            </div>
          </div>

          {/* 실시간 알림 */}
          <div className={styles.sideCard}>
            <div className={styles.sideCardHeader}>
              <span className={styles.sideCardTitle}>실시간 위반 알림</span>
            </div>
            <ul className={styles.feedList}>
              {violations.slice(0, FEED_MAX_COUNT).map((v) => (
                <li key={v.id} className={styles.feedItem}>
                  <span
                    className={styles.feedBadge}
                    style={{ backgroundColor: TYPE_COLOR[v.type] || '#64748b' }}
                  >
                    {v.type}
                  </span>
                  <span className={styles.feedCamera}>{v.camera}</span>
                  <span className={styles.feedTime}>{extractTime(v.timestamp)}</span>
                </li>
              ))}
            </ul>
          </div>
        </div>
      </div>
    </div>
  )
}

export default MainPage
