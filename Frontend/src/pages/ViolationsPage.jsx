import { useState, useMemo, useRef, useCallback } from 'react'
import { TYPE_COLOR, BADGE_COLOR } from '../constants'
import useApi from '../hooks/useApi'
import { apiUrl } from '../config'
import { toViolationList } from '../utils'
import ViolationDetailModal from '../components/ViolationDetailModal'
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

const ITEMS_PER_PAGE = 10
// 페이지 버튼을 5개씩 묶어서 보여준다. 예: « ‹ 11 12 [13] 14 15 › »
const PAGE_BLOCK = 5

function ViolationsPage() {
  const [filter, setFilter] = useState('전체')
  const [startDate, setStartDate] = useState('')
  const [endDate, setEndDate] = useState('')
  const [page, setPage] = useState(1)
  const [selected, setSelected] = useState(null)
  const closeDetail = useCallback(() => setSelected(null), [])

  // 백엔드 /api/violations에 날짜 조건(startDate/endDate)이 없어서, 기록을 받아 날짜·유형 필터와
  // 페이지 나누기를 브라우저에서 한다. 백엔드에 날짜 조건이 생기면 page/size/type/startDate/endDate를
  // 서버로 넘기도록 바꿀 것.
  const { data: apiData, loading, error } = useApi(apiUrl('/api/violations?size=9999'))
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

      {selected && <ViolationDetailModal violation={selected} onClose={closeDetail} />}
    </div>
  )
}

export default ViolationsPage
