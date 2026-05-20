import { useState, useEffect, useMemo } from 'react'
import { TYPE_COLOR, BADGE_COLOR, HOVER_COLOR } from '../constants'
import { DUMMY_VIOLATIONS } from '../data/dummyData'
import useApi from '../hooks/useApi'
import styles from './ViolationsPage.module.css'

const MODAL_BADGE = {
  '헬멧 미착용': { bg: '#e0f2fe', text: '#0369a1', border: '#bae6fd' },
  '다인 탑승':   { bg: '#fef3c7', text: '#b45309', border: '#fde68a' },
  '인도 주행':   { bg: '#fce7f3', text: '#be185d', border: '#fbcfe8' },
}

const TYPE_ICON = {
  '헬멧 미착용': (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M12 2a9 9 0 0 0-9 9h18a9 9 0 0 0-9-9z"/>
      <path d="M3 11h18v2a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-2z"/>
      <path d="M8 15v1a4 4 0 0 0 8 0v-1"/>
    </svg>
  ),
  '다인 탑승': (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M16 21v-2a4 4 0 0 0-4-4H6a4 4 0 0 0-4 4v2"/>
      <circle cx="9" cy="7" r="4"/>
      <path d="M22 21v-2a4 4 0 0 0-3-3.87"/>
      <path d="M16 3.13a4 4 0 0 1 0 7.75"/>
    </svg>
  ),
  '인도 주행': (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <circle cx="12" cy="4" r="2"/>
      <path d="M9 22v-7l-2-5h10l-2 5v7"/>
      <path d="M9 12h6"/>
    </svg>
  ),
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

const ITEMS_PER_PAGE = 10

function ViolationsPage() {
  const [filter, setFilter] = useState('전체')
  const [startDate, setStartDate] = useState('')
  const [endDate, setEndDate] = useState('')
  const [page, setPage] = useState(1)
  const [selected, setSelected] = useState(null)

  const { data: apiData, loading } = useApi('/api/violations')
  const violations = (apiData && apiData.length > 0) ? apiData : DUMMY_VIOLATIONS

  useEffect(() => {
    setPage(1)
  }, [filter, startDate, endDate])

  useEffect(() => {
    if (!selected) return
    const onKeyDown = (e) => {
      if (e.key === 'Escape') setSelected(null)
    }
    window.addEventListener('keydown', onKeyDown)
    return () => window.removeEventListener('keydown', onKeyDown)
  }, [selected])

  const filtered = useMemo(() => {
    return violations.filter((v) => {
      if (filter !== '전체' && v.type !== filter) return false
      if (startDate && v.timestamp.slice(0, 10) < startDate) return false
      if (endDate && v.timestamp.slice(0, 10) > endDate) return false
      return true
    })
  }, [violations, filter, startDate, endDate])

  const isFiltered = filter !== '전체' || startDate !== '' || endDate !== ''
  const handleReset = () => { setFilter('전체'); setStartDate(''); setEndDate('') }

  const totalPages = Math.ceil(filtered.length / ITEMS_PER_PAGE)
  const paginated = filtered.slice((page - 1) * ITEMS_PER_PAGE, page * ITEMS_PER_PAGE)

  if (loading) return <div className={styles.status}>불러오는 중...</div>

  return (
    <div className={styles.page}>
      <div className={styles.header}>
        <div className={styles.titleRow}>
          <h2 className={styles.title}>단속 기록</h2>
        </div>

        <div className={styles.filterPanel}>
          {/* 패널 헤더 */}
          <div className={styles.filterPanelHeader}>
            <div className={styles.filterPanelTitle}>
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <line x1="21" y1="4" x2="14" y2="4"/><line x1="10" y1="4" x2="3" y2="4"/>
                <line x1="21" y1="12" x2="12" y2="12"/><line x1="8" y1="12" x2="3" y2="12"/>
                <line x1="21" y1="20" x2="16" y2="20"/><line x1="12" y1="20" x2="3" y2="20"/>
                <line x1="14" y1="2" x2="14" y2="6"/>
                <line x1="8" y1="10" x2="8" y2="14"/>
                <line x1="16" y1="18" x2="16" y2="22"/>
              </svg>
              필터
            </div>
            {isFiltered && (
              <button className={styles.resetBtn} onClick={handleReset}>
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/>
                </svg>
                초기화
              </button>
            )}
          </div>

          {/* 날짜 필터 */}
          <div className={styles.filterRow}>
            <span className={styles.filterLabel}>날짜</span>
            <div className={styles.dateFilterRow}>
              <input
                type="date"
                className={styles.dateInput}
                value={startDate}
                onChange={e => setStartDate(e.target.value)}
              />
              <span className={styles.dateSep}>~</span>
              <input
                type="date"
                className={styles.dateInput}
                value={endDate}
                min={startDate}
                onChange={e => setEndDate(e.target.value)}
              />
            </div>
          </div>

          {/* 위반 유형 필터 */}
          <div className={styles.filterRow}>
            <span className={styles.filterLabel}>위반 유형</span>
            <div className={styles.filterGroup}>
              <button
                className={`${styles.filterBtn} ${filter === '헬멧 미착용' ? styles.active : ''}`}
                style={filter === '헬멧 미착용'
                  ? { backgroundColor: TYPE_COLOR['헬멧 미착용'], borderColor: TYPE_COLOR['헬멧 미착용'], color: '#fff' }
                  : { color: TYPE_COLOR['헬멧 미착용'], '--hover-bg': HOVER_COLOR['헬멧 미착용'].bg, '--hover-border': HOVER_COLOR['헬멧 미착용'].border, '--hover-text': TYPE_COLOR['헬멧 미착용'] }
                }
                onClick={() => setFilter('헬멧 미착용')}
              >
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <path d="M12 2a9 9 0 0 0-9 9h18a9 9 0 0 0-9-9z"/>
                  <path d="M3 11h18v2a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-2z"/>
                  <path d="M8 15v1a4 4 0 0 0 8 0v-1"/>
                </svg>
                헬멧 미착용
              </button>
              <button
                className={`${styles.filterBtn} ${filter === '다인 탑승' ? styles.active : ''}`}
                style={filter === '다인 탑승'
                  ? { backgroundColor: TYPE_COLOR['다인 탑승'], borderColor: TYPE_COLOR['다인 탑승'], color: '#fff' }
                  : { color: TYPE_COLOR['다인 탑승'], '--hover-bg': HOVER_COLOR['다인 탑승'].bg, '--hover-border': HOVER_COLOR['다인 탑승'].border, '--hover-text': TYPE_COLOR['다인 탑승'] }
                }
                onClick={() => setFilter('다인 탑승')}
              >
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <path d="M16 21v-2a4 4 0 0 0-4-4H6a4 4 0 0 0-4 4v2"/>
                  <circle cx="9" cy="7" r="4"/>
                  <path d="M22 21v-2a4 4 0 0 0-3-3.87"/>
                  <path d="M16 3.13a4 4 0 0 1 0 7.75"/>
                </svg>
                다인 탑승
              </button>
              <button
                className={`${styles.filterBtn} ${filter === '인도 주행' ? styles.active : ''}`}
                style={filter === '인도 주행'
                  ? { backgroundColor: TYPE_COLOR['인도 주행'], borderColor: TYPE_COLOR['인도 주행'], color: '#fff' }
                  : { color: TYPE_COLOR['인도 주행'], '--hover-bg': HOVER_COLOR['인도 주행'].bg, '--hover-border': HOVER_COLOR['인도 주행'].border, '--hover-text': TYPE_COLOR['인도 주행'] }
                }
                onClick={() => setFilter('인도 주행')}
              >
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <circle cx="12" cy="4" r="2"/>
                  <path d="M9 22v-7l-2-5h10l-2 5v7"/>
                  <path d="M9 12h6"/>
                </svg>
                인도 주행
              </button>
            </div>
            <div className={styles.filterDivider} />
            <button className={`${styles.filterBtn} ${filter === '전체' ? styles.active : ''}`} onClick={() => setFilter('전체')}>
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <polygon points="22 3 2 3 10 12.46 10 19 14 21 14 12.46 22 3"/>
              </svg>
              전체
            </button>
          </div>
        </div>
      </div>

      <span className={styles.countBadge}>{filtered.length}건</span>

      <div className={styles.cardList}>
        {paginated.length === 0 ? (
          <div className={styles.status}>해당 조건의 위반 기록이 없습니다.</div>
        ) : (
          paginated.map((v) => (
            <div key={v.id} className={styles.card} onClick={() => setSelected(v)}>
              <div className={styles.cardTop}>
                <span className={styles.badge} style={{ backgroundColor: BADGE_COLOR[v.type]?.bg || '#f1f5f9', color: BADGE_COLOR[v.type]?.text || '#475569' }}>
                  {v.type}
                </span>
                <span className={styles.cardId}>#{v.id}</span>
              </div>
              <div className={styles.cardBody}>
                <div className={styles.cardRow}>
                  <span className={styles.cardLabel}>카메라</span>
                  <span className={styles.cardValue}>{v.camera}</span>
                </div>
                <div className={styles.cardRow}>
                  <span className={styles.cardLabel}>일시</span>
                  <span className={styles.cardValue}>{v.timestamp}</span>
                </div>
              </div>
            </div>
          ))
        )}
      </div>

      {totalPages > 1 && (
        <div className={styles.pagination}>
          <button className={styles.pageBtn} onClick={() => setPage(p => p - 1)} disabled={page === 1}>←</button>
          {Array.from({ length: totalPages }, (_, i) => i + 1).map(p => (
            <button
              key={p}
              className={`${styles.pageBtn} ${p === page ? styles.pageActive : ''}`}
              onClick={() => setPage(p)}
            >
              {p}
            </button>
          ))}
          <button className={styles.pageBtn} onClick={() => setPage(p => p + 1)} disabled={page === totalPages}>→</button>
        </div>
      )}

      {selected && (
        <div className={styles.modalOverlay} onClick={() => setSelected(null)}>
          <div className={styles.modal} onClick={(e) => e.stopPropagation()}>
            <div className={styles.modalHeader}>
              <span>단속 기록 상세</span>
              <button className={styles.closeBtn} onClick={() => setSelected(null)}>
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/>
                </svg>
              </button>
            </div>

            <div className={styles.modalImageWrap}>
              {selected.image_url ? (
                <img src={selected.image_url} alt="위반 단속 이미지" className={styles.modalImage} />
              ) : (
                <div className={styles.modalImagePlaceholder}>
                  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" style={{ width: 40, height: 40 }}>
                    <rect x="3" y="3" width="18" height="18" rx="2"/><circle cx="8.5" cy="8.5" r="1.5"/>
                    <polyline points="21 15 16 10 5 21"/>
                  </svg>
                  캡처 이미지 없음
                </div>
              )}
              {MODAL_BADGE[selected.type] && (
                <span
                  className={styles.modalTypeBadge}
                  style={{
                    backgroundColor: MODAL_BADGE[selected.type].bg,
                    color: MODAL_BADGE[selected.type].text,
                    borderColor: MODAL_BADGE[selected.type].border,
                  }}
                >
                  {TYPE_ICON[selected.type]}
                  {selected.type}
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
                  <span className={styles.confidenceVal}>{selected.confidence}%</span>
                </div>
                <div className={styles.progressTrack}>
                  <div className={styles.progressFill} style={{ width: `${selected.confidence}%` }} />
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
                    <p className={styles.infoVal}>{selected.location || '—'}</p>
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
                    <p className={styles.infoVal}>{formatTimestamp(selected.timestamp)}</p>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

export default ViolationsPage
