import { useState, useRef, useEffect } from 'react'
import { BADGE_COLOR, FEED_MAX_COUNT } from '../constants'
import { extractTime, toViolationList } from '../utils'
import { apiUrl, aiUrl } from '../config'
import useApi from '../hooks/useApi'
import useSSE from '../hooks/useSSE'
import styles from './MainPage.module.css'

function MainPage() {
  // limit(기존)과 size(페이지네이션 적용 후)를 같이 보냄 — 백엔드 머지 후 limit 제거
  const { data, loading, error, connected } = useApi(apiUrl('/api/violations?limit=10&size=10'))
  const violations = toViolationList(data)

  const [toast, setToast] = useState(null)
  const [toastLeaving, setToastLeaving] = useState(false)
  const toastTimerRef = useRef(null)

  // 토스트 닫기: 아래로 사라지는 애니메이션(0.25초)이 끝난 뒤 제거
  const dismissToast = () => {
    clearTimeout(toastTimerRef.current)
    setToastLeaving(true)
    toastTimerRef.current = setTimeout(() => setToast(null), 250)
  }
  const [currentTime, setCurrentTime] = useState('')
  const [streamError, setStreamError] = useState(false)
  const [streamKey, setStreamKey] = useState(0)
  const streamAttemptRef = useRef(0)
  const streamRetryTimerRef = useRef(null)

  // 수동 재연결: 즉시 다시 시도하고 백오프 카운트를 리셋한다.
  const retryStream = () => {
    clearTimeout(streamRetryTimerRef.current)
    streamAttemptRef.current = 0
    setStreamError(false)
    setStreamKey(k => k + 1)
  }

  // 스트림이 끊기면 지수 백오프(1s -> 2s -> ... 최대 30s)로 자동 재연결한다.
  const handleStreamError = () => {
    setStreamError(true)
    const delay = Math.min(1000 * 2 ** streamAttemptRef.current, 30000)
    streamAttemptRef.current += 1
    streamRetryTimerRef.current = setTimeout(() => {
      setStreamError(false)
      setStreamKey(k => k + 1)
    }, delay)
  }

  const handleStreamLoad = () => {
    streamAttemptRef.current = 0
  }

  useEffect(() => {
    return () => clearTimeout(streamRetryTimerRef.current)
  }, [])

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

  const { connected: sseConnected } = useSSE(apiUrl('/api/stream'), (violation) => {
    clearTimeout(toastTimerRef.current)
    setToast(violation)
    setToastLeaving(false)
    // 4초 뒤 자동으로 사라짐 (X 버튼으로 바로 닫을 수도 있음)
    toastTimerRef.current = setTimeout(dismissToast, 4000)
  })

  // Backend(위반 목록 폴링), AI 영상 스트림, 실시간 알림(SSE) 셋 다 정상일 때만 "정상 작동"으로 표시한다.
  const systemOk = connected && !streamError && sseConnected

  return (
    <div className={styles.page}>
      {toast && (
        // 화면 우측 하단 고정. 아이콘 색은 위반 유형 색을 따름
        <div className={`${styles.toast} ${toastLeaving ? styles.toastLeaving : ''}`} role="status">
          <div
            className={styles.toastIcon}
            style={{ backgroundColor: BADGE_COLOR[toast.type]?.bg || '#f1f5f9', color: BADGE_COLOR[toast.type]?.text || '#475569' }}
          >
            {/* Lucide: triangle-alert */}
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.25" strokeLinecap="round" strokeLinejoin="round">
              <path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3" />
              <path d="M12 9v4" />
              <path d="M12 17h.01" />
            </svg>
          </div>
          <div className={styles.toastBody}>
            <div className={styles.toastTop}>
              <span className={styles.toastLabel}>새 위반 감지</span>
              <span className={styles.toastTime}>방금 전</span>
            </div>
            <div className={styles.toastContent}>
              <span
                className={styles.toastBadge}
                style={{ backgroundColor: BADGE_COLOR[toast.type]?.bg || '#f1f5f9', color: BADGE_COLOR[toast.type]?.text || '#475569' }}
              >
                {toast.type}
              </span>
              <span className={styles.toastCamera}>{toast.camera}</span>
              <span className={styles.toastConf}>신뢰도 {toast.confidence}%</span>
            </div>
          </div>
          <button className={styles.toastClose} onClick={dismissToast} aria-label="알림 닫기">
            {/* Lucide: x */}
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.25" strokeLinecap="round" strokeLinejoin="round">
              <path d="M18 6 6 18" />
              <path d="m6 6 12 12" />
            </svg>
          </button>
        </div>
      )}

      <div className={styles.header}>
        <h2 className={styles.pageTitle}>실시간 위반 감지 모니터링</h2>
        <p className={styles.desc}>
          AI가 영상을 실시간으로 분석해 헬멧 미착용·다인 탑승·인도 주행을 감지합니다.
        </p>
      </div>
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
                <div className={styles.streamCamSub}>분석 영상</div>
              </div>
            </div>
          </div>
          <div className={styles.streamBody}>
            {!streamError ? (
              <img
                src={aiUrl(`/video/stream?t=${streamKey}`)}
                className={styles.stream}
                alt="stream"
                onError={handleStreamError}
                onLoad={handleStreamLoad}
              />
            ) : (
              <button className={styles.streamPlaceholder} onClick={retryStream}>
                영상 스트림 영역<br />
                <small>AI 서버 연결 후 활성화 · 자동 재연결 중 · 클릭하여 즉시 재연결</small>
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
              <span className={`${styles.statusBadge} ${systemOk ? '' : styles.statusBadgeWarn}`}>
                {systemOk ? '정상 작동' : '연결 확인 필요'}
              </span>
            </div>
          </div>

          {/* 최근 위반 목록 */}
          <div className={styles.sideCard}>
            <div className={styles.sideCardHeader}>
              <span className={styles.sideCardTitle}>최근 위반</span>
              {!sseConnected && (
                <span className={`${styles.statusBadge} ${styles.statusBadgeWarn}`}>재연결 중</span>
              )}
            </div>
            {/* 목록 조회 실패는 이 카드 안에서만 표시한다. 페이지 전체를 에러 화면으로
                바꾸면 정상인 AI 영상까지 가려지고, 폴링이 한 번만 실패해도 영상이 끊긴다. */}
            {loading ? (
              <div className={styles.feedStatus}>불러오는 중...</div>
            ) : error && violations.length === 0 ? (
              <div className={styles.feedStatusError}>단속 기록 서버에 연결할 수 없습니다.</div>
            ) : (
              <ul className={styles.feedList}>
                {violations.slice(0, FEED_MAX_COUNT).map((v) => (
                  <li key={v.id} className={styles.feedItem}>
                    <span
                      className={styles.feedBadge}
                      style={{ backgroundColor: BADGE_COLOR[v.type]?.bg || '#f1f5f9', color: BADGE_COLOR[v.type]?.text || '#475569' }}
                    >
                      {v.type}
                    </span>
                    <span className={styles.feedCamera}>{v.camera}</span>
                    <span className={styles.feedTime}>{extractTime(v.timestamp)}</span>
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
