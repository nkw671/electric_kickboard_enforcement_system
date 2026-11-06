import { useEffect, useState } from 'react'
import { TYPE_ICON } from '../typeIcons'
import styles from './ViolationDetailModal.module.css'

const MODAL_BADGE = {
  '헬멧 미착용': { bg: '#fef3c7', text: '#b45309', border: '#fde68a' },
  '다인 탑승':   { bg: '#ede9fe', text: '#6d28d9', border: '#ddd6fe' },
  '인도 주행':   { bg: '#fee2e2', text: '#b91c1c', border: '#fecaca' },
}

function formatTimestamp(ts) {
  if (!ts) return ''
  const [date, time] = ts.split(' ')
  const [y, m, d] = date.split('-')
  const [h, min, sec] = time.split(':')
  const hNum = parseInt(h)
  const ampm = hNum < 12 ? '오전' : '오후'
  const h12 = hNum % 12 || 12
  return `${y}년 ${parseInt(m)}월 ${parseInt(d)}일 ${ampm} ${h12}:${min}:${sec}`
}

/**
 * 단속 기록 상세 창 (캡처 이미지 + 신뢰도 + 위치·시각).
 * 단속 기록 페이지의 표, 모니터링 페이지의 최근 위반 목록·토스트에서 함께 쓴다.
 * Esc 또는 바깥 클릭으로 닫힌다.
 */
function ViolationDetailModal({ violation, onClose }) {
  // 불러오지 못한 이미지 주소. 이미지는 AI 서버(vast.ai)에 있어서 서버가 꺼져 있으면 깨진다.
  const [brokenImageUrl, setBrokenImageUrl] = useState(null)

  useEffect(() => {
    const onKeyDown = (e) => {
      if (e.key === 'Escape') onClose()
    }
    window.addEventListener('keydown', onKeyDown)
    return () => window.removeEventListener('keydown', onKeyDown)
  }, [onClose])

  return (
    <div className={styles.modalOverlay} onClick={onClose}>
      <div className={styles.modal} role="dialog" aria-modal="true" aria-label="단속 기록 상세" onClick={(e) => e.stopPropagation()}>
        <div className={styles.modalHeader}>
          <span>단속 기록 상세</span>
          <button className={styles.closeBtn} onClick={onClose} aria-label="닫기">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/>
            </svg>
          </button>
        </div>

        <div className={styles.modalImageWrap}>
          {violation.image_url && violation.image_url !== brokenImageUrl ? (
            <img
              src={violation.image_url}
              alt="위반 단속 이미지"
              className={styles.modalImage}
              onError={() => setBrokenImageUrl(violation.image_url)}
            />
          ) : (
            <div className={styles.modalImagePlaceholder}>
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" style={{ width: 40, height: 40 }}>
                <rect x="3" y="3" width="18" height="18" rx="2"/><circle cx="8.5" cy="8.5" r="1.5"/>
                <polyline points="21 15 16 10 5 21"/>
              </svg>
              {violation.image_url ? '이미지를 불러올 수 없습니다 (AI 서버 연결 확인)' : '캡처 이미지 없음'}
            </div>
          )}
          {MODAL_BADGE[violation.type] && (
            <span
              className={styles.modalTypeBadge}
              style={{
                backgroundColor: MODAL_BADGE[violation.type].bg,
                color: MODAL_BADGE[violation.type].text,
                borderColor: MODAL_BADGE[violation.type].border,
              }}
            >
              {TYPE_ICON[violation.type]}
              {violation.type}
            </span>
          )}
        </div>

        <div className={styles.modalDetails}>
          <div className={styles.confidenceBox}>
            <div className={styles.confidenceHeader}>
              <div className={styles.confidenceTitle}>
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>
                </svg>
                AI 분석 신뢰도
              </div>
              <span className={styles.confidenceVal}>{violation.confidence}%</span>
            </div>
            <div className={styles.progressTrack}>
              <div className={styles.progressFill} style={{ width: `${violation.confidence}%` }} />
            </div>
          </div>

          <div className={styles.infoList}>
            <div className={styles.infoRow}>
              <div className={styles.infoIconBox}>
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <path d="M21 10c0 7-9 13-9 13s-9-6-9-13a9 9 0 0 1 18 0z"/><circle cx="12" cy="10" r="3"/>
                </svg>
              </div>
              <div>
                <p className={styles.infoSubLabel}>단속 위치</p>
                <p className={styles.infoVal}>{violation.location || '—'}</p>
              </div>
            </div>
            <div className={styles.infoRow}>
              <div className={styles.infoIconBox}>
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/>
                </svg>
              </div>
              <div>
                <p className={styles.infoSubLabel}>단속 시각</p>
                <p className={styles.infoVal}>{formatTimestamp(violation.timestamp)}</p>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

export default ViolationDetailModal
