import { useState, useEffect, useMemo } from 'react'
import { TYPE_COLOR, BADGE_COLOR } from '../constants'
import { DUMMY_VIOLATIONS } from '../data/dummyData'
import useApi from '../hooks/useApi'
import styles from './ViolationsPage.module.css'

const FALLBACK_TYPE_COLOR = 'var(--ink-400)'

function downloadCsv(rows) {
  const header = ['ID', '유형', '카메라', '일시', '위치', '신뢰도(%)']
  const lines = [header.join(',')]
  rows.forEach(v => {
    const cells = [
      v.id,
      v.type,
      v.camera,
      v.timestamp,
      (v.location || '').replace(/,/g, ' '),
      v.confidence,
    ]
    lines.push(cells.map(c => `"${String(c).replace(/"/g, '""')}"`).join(','))
  })
  // BOM 포함 — 엑셀에서 한글 깨지지 않음
  const blob = new Blob(['﻿' + lines.join('\n')], { type: 'text/csv;charset=utf-8' })
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = `단속기록_${new Date().toISOString().slice(0, 10)}.csv`
  document.body.appendChild(a)
  a.click()
  document.body.removeChild(a)
  URL.revokeObjectURL(url)
}

const TYPE_ICON = {
  '헬멧 미착용': (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round">
      <path d="M12 2a9 9 0 0 0-9 9h18a9 9 0 0 0-9-9z"/>
      <path d="M3 11h18v2a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-2z"/>
      <path d="M8 15v1a4 4 0 0 0 8 0v-1"/>
    </svg>
  ),
  '다인 탑승': (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round">
      <path d="M16 21v-2a4 4 0 0 0-4-4H6a4 4 0 0 0-4 4v2"/>
      <circle cx="9" cy="7" r="4"/>
      <path d="M22 21v-2a4 4 0 0 0-3-3.87"/>
      <path d="M16 3.13a4 4 0 0 1 0 7.75"/>
    </svg>
  ),
  '인도 주행': (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round">
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
const FILTER_TYPES = ['전체', '헬멧 미착용', '다인 탑승', '인도 주행']

function ViolationsPage() {
  const [filter, setFilter]       = useState('전체')
  const [startDate, setStartDate] = useState('')
  const [endDate, setEndDate]     = useState('')
  const [page, setPage]           = useState(1)
  const [selected, setSelected]   = useState(null)

  const { data: apiData, loading } = useApi('/api/violations?limit=9999')
  const violations = (apiData && apiData.length > 0) ? apiData : DUMMY_VIOLATIONS

  useEffect(() => { setPage(1) }, [filter, startDate, endDate])

  useEffect(() => {
    if (!selected) return
    const onKeyDown = (e) => { if (e.key === 'Escape') setSelected(null) }
    window.addEventListener('keydown', onKeyDown)
    return () => window.removeEventListener('keydown', onKeyDown)
  }, [selected])

  const filtered = useMemo(() => violations.filter((v) => {
    if (filter !== '전체' && v.type !== filter) return false
    if (startDate && v.timestamp.slice(0, 10) < startDate) return false
    if (endDate   && v.timestamp.slice(0, 10) > endDate)   return false
    return true
  }), [violations, filter, startDate, endDate])

  const isFiltered   = filter !== '전체' || startDate !== '' || endDate !== ''
  const handleReset  = () => { setFilter('전체'); setStartDate(''); setEndDate('') }
  const totalPages   = Math.ceil(filtered.length / ITEMS_PER_PAGE)
  const paginated    = filtered.slice((page - 1) * ITEMS_PER_PAGE, page * ITEMS_PER_PAGE)

  const visiblePages = useMemo(() => {
    if (totalPages <= 5) return Array.from({ length: totalPages }, (_, i) => i + 1)
    let start = Math.max(1, page - 2)
    let end   = Math.min(totalPages, page + 2)
    if (end - start < 4) {
      if (start === 1) end = Math.min(totalPages, 5)
      else start = Math.max(1, end - 4)
    }
    return Array.from({ length: end - start + 1 }, (_, i) => start + i)
  }, [page, totalPages])

  if (loading) return <div className={styles.status}>불러오는 중...</div>

  return (
    <div className={styles.page}>

      {/* 상단 헤더 */}
      <div className={styles.topbar}>
        <div>
          <div className={styles.crumb}>OPERATIONS / VIOLATIONS</div>
          <h2 className={styles.pageTitle}>단속 기록</h2>
        </div>
      </div>

      {/* 필터 툴바 */}
      <div className={styles.toolbar}>
        <div className={styles.toolCount}>
          <span className={styles.countNum} data-numeric>{filtered.length.toLocaleString()}</span>
          <span className={styles.countLbl}>건{isFiltered ? ' · FILTERED' : ''}</span>
        </div>

        <div className={styles.segment}>
          {FILTER_TYPES.map(type => (
            <button
              key={type}
              className={`${styles.segBtn} ${filter === type ? styles.segActive : ''}`}
              onClick={() => setFilter(type)}
            >
              {type !== '전체' && (
                <span className={styles.segDot} style={{ backgroundColor: TYPE_COLOR[type] }} />
              )}
              {type}
            </button>
          ))}
        </div>

        <div className={styles.dateChip}>
          <input
            type="date"
            className={styles.dateInput}
            value={startDate}
            onChange={e => setStartDate(e.target.value)}
          />
          <span className={styles.dateSep}>—</span>
          <input
            type="date"
            className={styles.dateInput}
            value={endDate}
            min={startDate}
            onChange={e => setEndDate(e.target.value)}
          />
        </div>

        <div className={styles.toolRight}>
          {isFiltered && (
            <button className={styles.resetBtn} onClick={handleReset} title="필터 초기화">
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round">
                <line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/>
              </svg>
              초기화
            </button>
          )}
          <button className={styles.csvBtn} onClick={() => downloadCsv(filtered)}>
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round">
              <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
              <polyline points="7 10 12 15 17 10"/>
              <line x1="12" y1="15" x2="12" y2="3"/>
            </svg>
            CSV 내보내기
          </button>
        </div>
      </div>

      {/* 테이블 */}
      <div className={styles.tableWrap}>
        <table className={styles.table}>
          <thead>
            <tr>
              <th>ID</th>
              <th>유형</th>
              <th>카메라</th>
              <th>일시</th>
              <th>위치</th>
              <th>신뢰도</th>
              <th></th>
            </tr>
          </thead>
          <tbody>
            {paginated.length === 0 ? (
              <tr>
                <td colSpan={7}>
                  <div className={styles.emptyState}>
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
                      <circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/>
                    </svg>
                    데이터 없음
                  </div>
                </td>
              </tr>
            ) : paginated.map((v) => (
              <tr key={v.id} className={styles.row} onClick={() => setSelected(v)}>
                <td className={styles.cellId} data-numeric>#{v.id}</td>
                <td>
                  <span
                    className={styles.typeBadge}
                    style={{ background: BADGE_COLOR[v.type]?.bg, color: BADGE_COLOR[v.type]?.text }}
                  >
                    <span className={styles.typeDot} style={{ backgroundColor: TYPE_COLOR[v.type] }} />
                    {v.type}
                  </span>
                </td>
                <td className={styles.cellCamera} data-numeric>{v.camera}</td>
                <td className={styles.cellTime}   data-numeric>{v.timestamp}</td>
                <td className={styles.cellLoc}>
                  {v.location || <span className={styles.cellMute}>미상</span>}
                </td>
                <td className={styles.cellConf}>
                  <span className={styles.confNum} data-numeric>{v.confidence}%</span>
                  <div className={styles.confTrack}>
                    <div className={styles.confFill} style={{ width: `${v.confidence}%` }} />
                  </div>
                </td>
                <td className={styles.cellArrow}>
                  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round">
                    <line x1="5" y1="12" x2="19" y2="12"/><polyline points="12 5 19 12 12 19"/>
                  </svg>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* 페이지네이션 */}
      {totalPages > 1 && (
        <div className={styles.pagination}>
          <span className={styles.pageInfo} data-numeric>
            {(page - 1) * ITEMS_PER_PAGE + 1}–{Math.min(page * ITEMS_PER_PAGE, filtered.length)} / {filtered.length.toLocaleString()}
          </span>
          <div className={styles.pageBtns}>
            <button className={styles.pageBtn} onClick={() => setPage(p => p - 1)} disabled={page === 1}>
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round">
                <polyline points="15 18 9 12 15 6"/>
              </svg>
            </button>
            {visiblePages.map(p => (
              <button
                key={p}
                className={`${styles.pageBtn} ${p === page ? styles.pageActive : ''}`}
                onClick={() => setPage(p)}
              >
                {p}
              </button>
            ))}
            <button className={styles.pageBtn} onClick={() => setPage(p => p + 1)} disabled={page === totalPages}>
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round">
                <polyline points="9 18 15 12 9 6"/>
              </svg>
            </button>
          </div>
        </div>
      )}

      {/* 모달 */}
      {selected && (
        <div className={styles.modalOverlay} onClick={() => setSelected(null)}>
          <div className={styles.modal} onClick={(e) => e.stopPropagation()}>
            <div className={styles.modalHeader}>
              <span>단속 기록 상세</span>
              <button className={styles.closeBtn} onClick={() => setSelected(null)}>
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round">
                  <line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/>
                </svg>
              </button>
            </div>

            <div className={styles.modalImageWrap}>
              {selected.image_url ? (
                <img src={selected.image_url} alt="위반 단속 이미지" className={styles.modalImage} />
              ) : (
                <div className={styles.modalImagePlaceholder}>
                  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
                    <rect x="3" y="3" width="18" height="18" rx="2"/>
                    <circle cx="8.5" cy="8.5" r="1.5"/>
                    <polyline points="21 15 16 10 5 21"/>
                  </svg>
                  캡처 이미지 없음
                </div>
              )}
              {BADGE_COLOR[selected.type] && (
                <span
                  className={styles.modalTypeBadge}
                  style={{
                    background: BADGE_COLOR[selected.type].bg,
                    color: BADGE_COLOR[selected.type].text,
                  }}
                >
                  <span
                    className={styles.modalTypeDot}
                    style={{ backgroundColor: TYPE_COLOR[selected.type] }}
                  />
                  {TYPE_ICON[selected.type]}
                  {selected.type}
                </span>
              )}
            </div>

            <div className={styles.modalDetails}>
              <div className={styles.confidenceBox}>
                <div className={styles.confidenceHeader}>
                  <div className={styles.confidenceTitle}>
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round">
                      <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>
                    </svg>
                    AI 분석 신뢰도
                  </div>
                  <span className={styles.confidenceVal} data-numeric>{selected.confidence}%</span>
                </div>
                <div className={styles.progressTrack}>
                  <div className={styles.progressFill} style={{ width: `${selected.confidence}%` }} />
                </div>
              </div>

              <div className={styles.infoList}>
                <div className={styles.infoRow}>
                  <div className={styles.infoIconBox}>
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round">
                      <path d="M21 10c0 7-9 13-9 13s-9-6-9-13a9 9 0 0 1 18 0z"/>
                      <circle cx="12" cy="10" r="3"/>
                    </svg>
                  </div>
                  <div>
                    <p className={styles.infoSubLabel}>단속 위치</p>
                    <p className={styles.infoVal}>{selected.location || '—'}</p>
                  </div>
                </div>
                <div className={styles.infoRow}>
                  <div className={styles.infoIconBox}>
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round">
                      <circle cx="12" cy="12" r="10"/>
                      <polyline points="12 6 12 12 16 14"/>
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
