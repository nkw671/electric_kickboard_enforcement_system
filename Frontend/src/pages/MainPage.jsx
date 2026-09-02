import { useState, useRef, useEffect } from 'react'
import { TYPE_COLOR, FEED_MAX_COUNT } from '../constants'
import { extractTime } from '../utils'
import useApi from '../hooks/useApi'
import useSSE from '../hooks/useSSE'
import styles from './MainPage.module.css'

// TODO(backend): /api/stats가 전일 비교값(prevDay) 반환하도록 확장 필요 — miniDelta 동적 표시 가능해짐
const FALLBACK_TYPE_COLOR = 'var(--ink-400)'

function MainPage() {
  const { data: violations, loading, error, connected } = useApi('/api/violations?limit=10')
  const { data: stats } = useApi('/api/stats')

  const [toast, setToast] = useState(null)
  const toastTimerRef = useRef(null)
  const [currentTime, setCurrentTime] = useState('')
  const [streamTime, setStreamTime] = useState('')
  const [streamError, setStreamError] = useState(false)
  const [streamKey, setStreamKey] = useState(0)

  const retryStream = () => {
    setStreamError(false)
    setStreamKey(k => k + 1)
  }

  useEffect(() => {
    const tick = () => {
      const now = new Date()
      const h = String(now.getHours()).padStart(2, '0')
      const min = String(now.getMinutes()).padStart(2, '0')
      const sec = String(now.getSeconds()).padStart(2, '0')
      setCurrentTime(`${h}:${min}:${sec}`)
      setStreamTime(`${h}:${min}:${sec}`)
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

  const miniItems = [
    { value: stats.total,      label: '오늘 총 위반', color: 'var(--navy)' },
    { value: stats.helmet,     label: '헬멧 미착용',  color: 'var(--accent-blue)' },
    { value: stats.sidewalk,   label: '인도 주행',    color: 'var(--accent-rose)' },
    { value: stats.multiRider, label: '다인 탑승',    color: 'var(--accent-amber)' },
  ]

  return (
    <div className={styles.page}>
      {toast && (
        <div
          className={styles.toast}
          style={{ borderLeftColor: TYPE_COLOR[toast.type] || FALLBACK_TYPE_COLOR }}
        >
          <div className={styles.toastHeader}>
            <span className={styles.toastLabel}>
              NEW DETECTION · {extractTime(toast.timestamp)}
            </span>
            <span
              className={styles.toastConf}
              style={{ color: TYPE_COLOR[toast.type] || FALLBACK_TYPE_COLOR }}
            >
              CONF {toast.confidence}%
            </span>
          </div>
          <div className={styles.toastTitle}>{toast.type}</div>
          <div className={styles.toastMeta}>
            {toast.camera}{toast.location ? ` · ${toast.location}` : ''}
          </div>
        </div>
      )}

      <div className={styles.topbar}>
        <div>
          <div className={styles.crumb}>OPERATIONS / MONITORING</div>
          <h2 className={styles.pageTitle}>실시간 모니터링</h2>
        </div>
        <div className={styles.timestamp}>
          <span className={styles.timestampLbl}>LAST UPDATE</span>
          <b data-numeric>{currentTime}</b>
          <span className={styles.timestampZone}>KST</span>
        </div>
      </div>

      <div className={styles.contentGrid}>
        {/* 영상 스트림 */}
        <div className={styles.streamBox}>
          <div className={styles.streamHeader}>
            <div className={styles.streamHeaderLeft}>
              <span className={styles.camChip}>CAM-01</span>
              <div className={styles.streamInfo}>
                <div className={styles.streamCamName}>단속 카메라 1</div>
                <div className={styles.streamCamSub}>37.5665° N · 126.9780° E</div>
              </div>
            </div>
            <div className={styles.streamBadges}>
              {connected && (
                <span className={styles.recordBadge}>
                  <span className={styles.recordDot} /> REC
                </span>
              )}
              <span className={styles.liveBadge}>LIVE · 1080p</span>
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
                영상 스트림 영역
                <small>AI 서버 연결 후 활성화 · 클릭하여 재연결</small>
              </button>
            )}
            {streamTime && (
              <span className={styles.streamTimestamp}>{streamTime}</span>
            )}
          </div>
        </div>

        {/* 우측 사이드바 */}
        <div className={styles.sidebar}>
          {/* 시스템 상태 */}
          <div className={styles.sideCard}>
            <div className={styles.sideCardHeader}>
              <div className={styles.sideCardTitleRow}>
                <svg className={styles.shieldIcon} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round">
                  <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>
                </svg>
                <span className={styles.sideCardTitle}>시스템 상태</span>
              </div>
              <span className={styles.statusBadge}>정상 작동</span>
            </div>
            <div className={styles.miniStats}>
              {miniItems.map(({ value, label, color }) => {
                const dim = value === 0
                return (
                  <div key={label} className={styles.miniStat}>
                    <span
                      className={styles.miniLed}
                      style={{ backgroundColor: dim ? 'var(--ink-200)' : color }}
                    />
                    <span
                      className={styles.miniVal}
                      data-numeric
                      style={{ color: dim ? 'var(--ink-300)' : color }}
                    >
                      {value}
                    </span>
                    <span className={styles.miniLbl}>{label}</span>
                    <span className={styles.miniDelta}>— vs 어제</span>
                  </div>
                )
              })}
            </div>
          </div>

          {/* 실시간 알림 */}
          <div className={styles.sideCard}>
            <div className={styles.sideCardHeader}>
              <span className={styles.sideCardTitle}>실시간 위반 알림</span>
              <span className={styles.streamingBadge}>
                <span className={styles.streamingDot} /> STREAMING
              </span>
            </div>
            {violations.length === 0 ? (
              <div className={styles.feedEmpty}>
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
                  <circle cx="12" cy="12" r="9"/>
                  <line x1="12" y1="8" x2="12" y2="12"/>
                  <line x1="12" y1="16" x2="12.01" y2="16"/>
                </svg>
                <span>대기 중 · 새 위반이 감지되면 표시됩니다</span>
              </div>
            ) : (
              <ul className={styles.feedList}>
                {violations.slice(0, FEED_MAX_COUNT).map((v) => (
                  <li key={v.id} className={styles.feedItem}>
                    <span
                      className={styles.feedBar}
                      style={{ backgroundColor: TYPE_COLOR[v.type] || FALLBACK_TYPE_COLOR }}
                    />
                    <span className={styles.feedTime}>{extractTime(v.timestamp)}</span>
                    <span className={styles.feedMeta}>
                      <span className={styles.feedType}>{v.type}</span>
                      <span className={styles.feedCam}>{v.camera}</span>
                    </span>
                    <span className={styles.feedConf}>{v.confidence}%</span>
                  </li>
                ))}
              </ul>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}

export default MainPage
