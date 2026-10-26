import { useState, useEffect, useMemo, useRef } from 'react'
import { TYPE_COLOR, BADGE_COLOR } from '../constants'
import useApi from '../hooks/useApi'
import { apiUrl } from '../config'
import { toViolationList } from '../utils'
import styles from './ViolationsPage.module.css'

// 필터 탭 순서 (통계 페이지와 같은 순서, 전체를 맨 앞)
const FILTER_TYPES = ['전체', '헬멧 미착용', '다인 탑승', '인도 주행']

// 로컬 기준 n일 전 날짜를 "YYYY-MM-DD"로 (toISOString은 UTC라 자정 근처에 날짜가 밀림)
function daysAgo(n) {
  const d = new Date()
  d.setDate(d.getDate() - n)
  return `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, '0')}-${String(d.getDate()).padStart(2, '0')}`
}

// 빠른 기간 선택: [시작 n일 전, 끝 0일 전]. 전체는 날짜 조건 없음
const DATE_PRESETS = [
  { label: '전체 기간', range: null },
  { label: '오늘', range: [0, 0] },
  { label: '7일', range: [6, 0] },
  { label: '1개월', range: [29, 0] },
]

// 브라우저 달력 팝업 열기. 입력칸에 포커스를 먼저 줘서, 달력이 열린 동안을 포커스로 알 수 있게 한다.
// 지원하지 않거나 막히면(사용자 클릭이 아닌 경우 등) 포커스만 남는다.
function openPicker(input) {
  if (!input) return
  input.focus()
  try {
    input.showPicker()
  } catch {
    // 포커스만 유지
  }
}

// 날짜 한 칸: "2026.10.20"처럼 글자로 보여주고, 누르면 숨겨둔 date 입력의 달력을 연다.
// 고른 날짜가 없으면 fallback(실제 데이터의 첫/마지막 기록 날짜)을 연하게 보여준다.
// 평소엔 글자만, 마우스를 올리면 연한 배경, 이 칸의 달력이 열려 있는 동안은 진한 배경.
// 브라우저 달력은 닫힘 신호가 없어서, 날짜를 고르거나 Esc·다른 곳 클릭(포커스 해제) 때 표시를 끈다.
function DateField({ value, fallback, min, max, onChange, label, buttonRef }) {
  const inputRef = useRef(null)
  const [open, setOpen] = useState(false)
  const shown = value || fallback
  return (
    <span className={styles.dateField}>
      <button
        ref={buttonRef}
        type="button"
        className={`${styles.dateText} ${open ? styles.dateTextOpen : ''}`}
        onClick={() => {
          setOpen(true)
          openPicker(inputRef.current)
        }}
        aria-label={`${label}: ${shown || '선택 안 함'}`}
        aria-expanded={open}
      >
        {shown ? shown.replaceAll('-', '.') : '날짜 선택'}
      </button>
      <input
        ref={inputRef}
        type="date"
        className={styles.dateHidden}
        value={value}
        min={min}
        max={max}
        onChange={e => {
          onChange(e.target.value)
          e.target.blur()
        }}
        onBlur={() => setOpen(false)}
        onKeyDown={e => { if (e.key === 'Escape') e.currentTarget.blur() }}
        tabIndex={-1}
        aria-hidden="true"
      />
    </span>
  )
}

const MODAL_BADGE = {
  '헬멧 미착용': { bg: '#fef3c7', text: '#b45309', border: '#fde68a' },
  '다인 탑승':   { bg: '#dbeafe', text: '#1d4ed8', border: '#bfdbfe' },
  '인도 주행':   { bg: '#fee2e2', text: '#b91c1c', border: '#fecaca' },
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
// 페이지 버튼을 5개씩 묶어서 보여준다. 예: « ‹ 11 12 [13] 14 15 › »
const PAGE_BLOCK = 5

function ViolationsPage() {
  const [filter, setFilter] = useState('전체')
  const [startDate, setStartDate] = useState('')
  const [endDate, setEndDate] = useState('')
  const [page, setPage] = useState(1)
  const [selected, setSelected] = useState(null)
  // 불러오지 못한 이미지 주소. 이미지는 AI 서버(vast.ai)에 있어서 서버가 꺼져 있으면 깨진다.
  const [brokenImageUrl, setBrokenImageUrl] = useState(null)

  // limit(기존)과 size(페이지네이션 적용 후)를 같이 보냄 — 백엔드 머지 후 limit 제거
  const { data: apiData, loading, error } = useApi(apiUrl('/api/violations?limit=9999&size=9999'))
  const violations = useMemo(() => toViolationList(apiData), [apiData])
  // 달력 아이콘을 눌러도 시작 날짜 달력이 열리도록 (시작 날짜 버튼을 대신 눌러줌)
  const startButtonRef = useRef(null)
  // 날짜를 고르지 않았을 때 날짜 칸에 보여줄 실제 데이터 범위(첫 기록 ~ 마지막 기록)
  const dataRange = useMemo(() => {
    const days = violations.map(v => v.timestamp.slice(0, 10)).sort()
    return { first: days[0] ?? '', last: days[days.length - 1] ?? '' }
  }, [violations])

  // 필터가 바뀌면 페이지를 1로 되돌린다. 렌더 중 조건부 setState로 처리해
  // effect 안에서 setState를 호출하지 않도록 함(cascading render 방지).
  const filterKey = `${filter}|${startDate}|${endDate}`
  const [prevFilterKey, setPrevFilterKey] = useState(filterKey)
  if (filterKey !== prevFilterKey) {
    setPrevFilterKey(filterKey)
    setPage(1)
  }

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

  const applyPreset = (range) => {
    setStartDate(range ? daysAgo(range[0]) : '')
    setEndDate(range ? daysAgo(range[1]) : '')
  }
  // 지금 날짜 조건이 어느 프리셋과 같은지 (직접 고른 기간이면 없음)
  const activePreset = DATE_PRESETS.find(({ range }) =>
    range ? startDate === daysAgo(range[0]) && endDate === daysAgo(range[1]) : !startDate && !endDate
  )?.label
  const handleReset = () => { setFilter('전체'); setStartDate(''); setEndDate('') }

  const totalPages = Math.ceil(filtered.length / ITEMS_PER_PAGE)
  const blockStart = Math.floor((page - 1) / PAGE_BLOCK) * PAGE_BLOCK + 1
  const blockEnd = Math.min(blockStart + PAGE_BLOCK - 1, totalPages)
  const blockPages = Array.from({ length: blockEnd - blockStart + 1 }, (_, i) => blockStart + i)
  const paginated = filtered.slice((page - 1) * ITEMS_PER_PAGE, page * ITEMS_PER_PAGE)

  if (loading) return <div className={styles.status}>불러오는 중...</div>
  // 이전 데이터가 있으면 폴링이 한 번 실패해도 목록을 그대로 보여준다.
  if (error && !apiData) return <div className={styles.statusError}>서버에 연결할 수 없습니다.</div>

  return (
    <div className={styles.page}>
      <div className={styles.header}>
        <div className={styles.titleRow}>
          <h2 className={styles.title}>단속 기록</h2>
          <p className={styles.desc}>
            AI가 감지한 위반 기록을 날짜·유형별로 조회하고, 기록을 누르면 캡처 이미지를 확인할 수 있습니다.
          </p>
        </div>

        {/* 한 줄 툴바: 왼쪽 날짜·위반 유형 / 오른쪽 초기화·건수 */}
        <div className={styles.toolbar}>
          <div className={styles.toolbarLeft}>
            <div className={styles.dateRange}>
              <button
                type="button"
                className={styles.dateIconBtn}
                onClick={() => startButtonRef.current?.click()}
                aria-label="시작 날짜 달력 열기"
              >
                {/* Lucide: calendar */}
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <path d="M8 2v4" /><path d="M16 2v4" />
                  <rect width="18" height="18" x="3" y="4" rx="2" />
                  <path d="M3 10h18" />
                </svg>
              </button>
              <DateField
                buttonRef={startButtonRef}
                value={startDate}
                fallback={dataRange.first}
                max={endDate || undefined}
                onChange={setStartDate}
                label="시작 날짜"
              />
              <span className={styles.dateSep}>~</span>
              <DateField
                value={endDate}
                fallback={dataRange.last}
                min={startDate || undefined}
                onChange={setEndDate}
                label="끝 날짜"
              />
            </div>

            {/* 빠른 기간 선택 */}
            <div className={styles.segment} role="group" aria-label="기간">
              {DATE_PRESETS.map(({ label, range }) => (
                <button
                  key={label}
                  className={`${styles.segBtn} ${activePreset === label ? styles.segActive : ''}`}
                  aria-pressed={activePreset === label}
                  onClick={() => applyPreset(range)}
                >
                  {label}
                </button>
              ))}
            </div>

            {/* 위반 유형: 전체를 맨 앞에 둔 세그먼트 탭 */}
            <div className={styles.segment} role="group" aria-label="위반 유형">
              {FILTER_TYPES.map((t) => (
                <button
                  key={t}
                  className={`${styles.segBtn} ${filter === t ? styles.segActive : ''}`}
                  aria-pressed={filter === t}
                  onClick={() => setFilter(t)}
                >
                  {t !== '전체' && <span className={styles.segDot} style={{ backgroundColor: TYPE_COLOR[t] }} />}
                  {t}
                </button>
              ))}
            </div>
          </div>

          <div className={styles.toolbarRight}>
            {isFiltered && (
              <button className={styles.resetBtn} onClick={handleReset}>
                {/* Lucide: rotate-ccw */}
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <path d="M3 12a9 9 0 1 0 9-9 9.75 9.75 0 0 0-6.74 2.74L3 8" />
                  <path d="M3 3v5h5" />
                </svg>
                초기화
              </button>
            )}
            <span className={styles.countText}>총 <strong>{filtered.length}</strong>건</span>
          </div>
        </div>
      </div>

      {/* 한 화면에 10건이 다 보이도록 표로 표시. 행을 누르면(또는 Enter) 상세 팝업 */}
      <div className={styles.tableWrap}>
        {paginated.length === 0 ? (
          <div className={styles.status}>해당 조건의 위반 기록이 없습니다.</div>
        ) : (
          <table className={styles.table}>
            <thead>
              <tr>
                <th className={styles.colId}>ID</th>
                <th>위반 유형</th>
                <th>카메라</th>
                <th>위치</th>
                <th>단속 일시</th>
                <th className={styles.colNum}>신뢰도</th>
                <th className={styles.colAction} aria-label="상세" />
              </tr>
            </thead>
            <tbody>
              {paginated.map((v) => (
                <tr
                  key={v.id}
                  className={styles.row}
                  onClick={() => setSelected(v)}
                  onKeyDown={(e) => { if (e.key === 'Enter') setSelected(v) }}
                  tabIndex={0}
                >
                  <td className={styles.cellId}>#{v.id}</td>
                  <td>
                    <span className={styles.badge} style={{ backgroundColor: BADGE_COLOR[v.type]?.bg || '#f1f5f9', color: BADGE_COLOR[v.type]?.text || '#475569' }}>
                      {v.type}
                    </span>
                  </td>
                  <td className={styles.cellStrong}>{v.camera}</td>
                  <td className={styles.cellMuted}>{v.location || '—'}</td>
                  <td className={styles.cellTime}>{v.timestamp}</td>
                  <td className={`${styles.cellConf} ${styles.colNum}`}>{v.confidence}%</td>
                  <td className={styles.colAction}>
                    <span className={styles.viewLink}>
                      캡처 보기
                      {/* Lucide: chevron-right */}
                      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.25" strokeLinecap="round" strokeLinejoin="round">
                        <path d="m9 18 6-6-6-6" />
                      </svg>
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}

        {/* 표 카드 아래쪽: 페이지 버튼 */}
        {totalPages > 1 && (
          <div className={styles.tableFooter}>
            <div className={styles.pagination}>
              <button className={styles.pageIconBtn} onClick={() => setPage(1)} disabled={page === 1} aria-label="첫 페이지">
                {/* Lucide: chevrons-left */}
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="m11 17-5-5 5-5" /><path d="m18 17-5-5 5-5" /></svg>
              </button>
              <button className={styles.pageIconBtn} onClick={() => setPage(blockStart - PAGE_BLOCK)} disabled={blockStart === 1} aria-label="이전 묶음">
                {/* Lucide: chevron-left */}
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="m15 18-6-6 6-6" /></svg>
              </button>
              {blockPages.map(p => (
                <button
                  key={p}
                  className={`${styles.pageBtn} ${p === page ? styles.pageActive : ''}`}
                  onClick={() => setPage(p)}
                  aria-current={p === page ? 'page' : undefined}
                >
                  {p}
                </button>
              ))}
              <button className={styles.pageIconBtn} onClick={() => setPage(blockStart + PAGE_BLOCK)} disabled={blockEnd === totalPages} aria-label="다음 묶음">
                {/* Lucide: chevron-right */}
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="m9 18 6-6-6-6" /></svg>
              </button>
              <button className={styles.pageIconBtn} onClick={() => setPage(totalPages)} disabled={page === totalPages} aria-label="마지막 페이지">
                {/* Lucide: chevrons-right */}
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="m6 17 5-5-5-5" /><path d="m13 17 5-5-5-5" /></svg>
              </button>
            </div>
          </div>
        )}
      </div>

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
              {selected.image_url && selected.image_url !== brokenImageUrl ? (
                <img
                  src={selected.image_url}
                  alt="위반 단속 이미지"
                  className={styles.modalImage}
                  onError={() => setBrokenImageUrl(selected.image_url)}
                />
              ) : (
                <div className={styles.modalImagePlaceholder}>
                  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" style={{ width: 40, height: 40 }}>
                    <rect x="3" y="3" width="18" height="18" rx="2"/><circle cx="8.5" cy="8.5" r="1.5"/>
                    <polyline points="21 15 16 10 5 21"/>
                  </svg>
                  {selected.image_url ? '이미지를 불러올 수 없습니다 (AI 서버 연결 확인)' : '캡처 이미지 없음'}
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
