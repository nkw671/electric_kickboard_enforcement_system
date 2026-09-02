import { useState, useEffect, useRef } from 'react'
import styles from './ZonePage.module.css'

const DEFAULT_W = 800
const DEFAULT_H = 450

const ZONE_COLORS_BGR = [
  [0,   0,   255],
  [0,   165, 255],
  [0,   255, 0  ],
  [255, 100, 0  ],
  [255, 0,   200],
]

function bgrToRgb([b, g, r]) {
  return `rgb(${r},${g},${b})`
}

function bgrToHex([b, g, r]) {
  return `#${r.toString(16).padStart(2, '0')}${g.toString(16).padStart(2, '0')}${b.toString(16).padStart(2, '0')}`
}

function ZonePage() {
  const canvasRef = useRef(null)
  const imgRef = useRef(null)
  const [zones, setZones] = useState([])
  const [currentPts, setCurrentPts] = useState([])
  const [isDrawing, setIsDrawing] = useState(false)
  const [status, setStatus] = useState(null)
  const [aiConnected, setAiConnected] = useState(false)
  const [streamError, setStreamError] = useState(false)
  const [videoDims, setVideoDims] = useState({ w: DEFAULT_W, h: DEFAULT_H })
  const [hoverPos, setHoverPos] = useState(null)

  useEffect(() => {
    fetch('/ai/zones')
      .then(r => r.json())
      .then(data => {
        setZones(data.zones || [])
        setAiConnected(true)
      })
      .catch(() => setAiConnected(false))
  }, [])

  useEffect(() => {
    if (!aiConnected || streamError) return
    const img = imgRef.current
    if (!img) return
    const trySync = () => {
      if (img.naturalWidth > 0) {
        setVideoDims({ w: img.naturalWidth, h: img.naturalHeight })
        return true
      }
      return false
    }
    if (trySync()) return
    const timer = setInterval(() => { if (trySync()) clearInterval(timer) }, 200)
    return () => clearInterval(timer)
  }, [aiConnected, streamError])

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    ctx.clearRect(0, 0, videoDims.w, videoDims.h)

    zones.forEach(zone => {
      if (zone.pts.length < 2) return
      const [b, g, r] = zone.color
      const colorRgb = `rgb(${r},${g},${b})`

      ctx.beginPath()
      ctx.moveTo(zone.pts[0][0], zone.pts[0][1])
      zone.pts.slice(1).forEach(([x, y]) => ctx.lineTo(x, y))
      ctx.closePath()
      ctx.fillStyle = `rgba(${r},${g},${b},0.18)`
      ctx.fill()
      ctx.strokeStyle = colorRgb
      ctx.lineWidth = 2
      ctx.stroke()

      zone.pts.forEach(([x, y]) => {
        ctx.beginPath()
        ctx.arc(x, y, 4, 0, Math.PI * 2)
        ctx.fillStyle = colorRgb
        ctx.fill()
      })

      const cx = zone.pts.reduce((s, p) => s + p[0], 0) / zone.pts.length
      const cy = zone.pts.reduce((s, p) => s + p[1], 0) / zone.pts.length
      ctx.font = 'bold 13px "IBM Plex Mono", monospace'
      ctx.fillStyle = 'rgba(0,0,0,0.55)'
      ctx.fillText(zone.name, cx - 20 + 1, cy + 5 + 1)
      ctx.fillStyle = colorRgb
      ctx.fillText(zone.name, cx - 20, cy + 5)
    })

    if (currentPts.length > 0) {
      const color = ZONE_COLORS_BGR[zones.length % ZONE_COLORS_BGR.length]
      const colorRgb = bgrToRgb(color)

      ctx.beginPath()
      ctx.moveTo(currentPts[0][0], currentPts[0][1])
      currentPts.slice(1).forEach(([x, y]) => ctx.lineTo(x, y))
      ctx.strokeStyle = colorRgb
      ctx.lineWidth = 2
      ctx.setLineDash([6, 4])
      ctx.stroke()
      ctx.setLineDash([])

      currentPts.forEach(([x, y], i) => {
        ctx.beginPath()
        ctx.arc(x, y, i === 0 ? 6 : 4, 0, Math.PI * 2)
        ctx.fillStyle = i === 0 ? '#ffffff' : colorRgb
        ctx.fill()
        ctx.strokeStyle = colorRgb
        ctx.lineWidth = 2
        ctx.stroke()
      })
    }
  }, [zones, currentPts, videoDims])

  const finishZone = () => {
    if (currentPts.length < 3) return
    const color = ZONE_COLORS_BGR[zones.length % ZONE_COLORS_BGR.length]
    const name = `Zone-${zones.length + 1}`
    setZones(prev => [...prev, { name, pts: currentPts, color }])
    setCurrentPts([])
    setIsDrawing(false)
  }

  const cancelDraw = () => {
    setCurrentPts([])
    setIsDrawing(false)
  }

  const deleteZone = (idx) => {
    setZones(prev => prev.filter((_, i) => i !== idx).map((z, i) => ({ ...z, name: `Zone-${i + 1}` })))
  }

  const saveZones = async () => {
    try {
      const res = await fetch('/ai/zones', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ zones: zones.map(z => ({ name: z.name, pts: z.pts, color: z.color })) }),
      })
      if (!res.ok) throw new Error()
      setStatus({ ok: true, msg: '저장 완료' })
    } catch {
      setStatus({ ok: false, msg: '저장 실패 — AI 서버 연결 확인' })
    }
    setTimeout(() => setStatus(null), 3000)
  }

  const deleteAllZones = async () => {
    try {
      await fetch('/ai/zones', { method: 'DELETE' })
      setZones([])
      setStatus({ ok: true, msg: '전체 삭제 완료' })
    } catch {
      setStatus({ ok: false, msg: '삭제 실패 — AI 서버 연결 확인' })
    }
    setTimeout(() => setStatus(null), 3000)
  }

  // 키보드 단축키: N = 새 구역, Enter = 완료, ⌘Z = 마지막 점 취소, Esc = 취소
  useEffect(() => {
    const onKey = (e) => {
      if (!isDrawing && e.key.toLowerCase() === 'n') {
        e.preventDefault()
        setIsDrawing(true)
        return
      }
      if (!isDrawing) return
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 'z') {
        e.preventDefault()
        setCurrentPts(prev => prev.slice(0, -1))
        return
      }
      if (e.key === 'Enter')  finishZone()
      if (e.key === 'Escape') cancelDraw()
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [isDrawing, currentPts, zones])

  const handleCanvasClick = (e) => {
    if (!isDrawing) return
    const canvas = canvasRef.current
    const rect = canvas.getBoundingClientRect()
    const x = Math.round((e.clientX - rect.left) * (videoDims.w / rect.width))
    const y = Math.round((e.clientY - rect.top) * (videoDims.h / rect.height))
    setCurrentPts(prev => [...prev, [x, y]])
  }

  const handleMouseMove = (e) => {
    const canvas = canvasRef.current
    const rect = canvas.getBoundingClientRect()
    const x = Math.round((e.clientX - rect.left) * (videoDims.w / rect.width))
    const y = Math.round((e.clientY - rect.top) * (videoDims.h / rect.height))
    setHoverPos({ x, y })
  }

  const handleMouseLeave = () => setHoverPos(null)

  return (
    <div className={styles.page}>
      <div className={styles.topbar}>
        <div>
          <div className={styles.crumb}>OPERATIONS / ZONE CONFIG</div>
          <h2 className={styles.pageTitle}>구역 설정</h2>
        </div>
      </div>

      <div className={styles.layout}>
        {/* 캔버스 영역 */}
        <div className={styles.canvasWrap}>
          <div className={styles.canvasHeader}>
            <div className={styles.canvasHeaderLeft}>
              <span className={`${styles.connDot} ${aiConnected ? styles.connOn : styles.connOff}`} />
              <span className={styles.connLabel}>{aiConnected ? 'AI 서버 연결됨' : 'AI 서버 미연결'}</span>
              {isDrawing && (
                <span className={styles.drawingBadge}>
                  <span className={styles.drawingDot} />
                  DRAWING · {currentPts.length}pt
                </span>
              )}
            </div>
            <div className={styles.canvasHeaderRight}>
              {isDrawing ? (
                <div className={styles.kbHints}>
                  <span className={styles.kbHint}><kbd className={styles.kbd}>Enter</kbd> 완료</span>
                  <span className={styles.kbHint}><kbd className={styles.kbd}>⌘Z</kbd> 마지막 점</span>
                  <span className={styles.kbHint}><kbd className={styles.kbd}>Esc</kbd> 취소</span>
                </div>
              ) : (
                <div className={styles.kbHints}>
                  <span className={styles.kbHint}><kbd className={styles.kbd}>N</kbd> 새 구역</span>
                </div>
              )}
            </div>
          </div>

          <div className={styles.canvasStage}>
            {aiConnected && !streamError ? (
              <img
                ref={imgRef}
                src="/ai/video/stream"
                className={styles.canvasVideo}
                alt="stream"
                onError={() => setStreamError(true)}
              />
            ) : (
              <div className={styles.canvasPlaceholder}>
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
                  <rect x="2" y="2" width="20" height="20" rx="2.18" ry="2.18"/>
                  <line x1="7" y1="2" x2="7" y2="22"/>
                  <line x1="17" y1="2" x2="17" y2="22"/>
                  <line x1="2" y1="12" x2="22" y2="12"/>
                  <line x1="2" y1="7" x2="7" y2="7"/>
                  <line x1="2" y1="17" x2="7" y2="17"/>
                  <line x1="17" y1="17" x2="22" y2="17"/>
                  <line x1="17" y1="7" x2="22" y2="7"/>
                </svg>
                AI 서버 연결 후 활성화
                <small>http://localhost:8000</small>
              </div>
            )}

            <canvas
              ref={canvasRef}
              width={videoDims.w}
              height={videoDims.h}
              className={`${styles.canvas} ${isDrawing ? styles.canvasDrawing : ''}`}
              onClick={handleCanvasClick}
              onMouseMove={handleMouseMove}
              onMouseLeave={handleMouseLeave}
            />

            {/* 좌상단 좌표 오버레이 */}
            <div className={styles.coordOverlay}>
              <span>x <b data-numeric>{hoverPos?.x ?? '—'}</b></span>
              <span>y <b data-numeric>{hoverPos?.y ?? '—'}</b></span>
              <span className={styles.coordSep}>·</span>
              <span>pts <b data-numeric>{currentPts.length}</b></span>
            </div>

            {/* 인캔버스 툴바 — 그리는 중일 때만 표시 */}
            {isDrawing && (
              <div className={styles.canvasToolbar}>
                <button
                  className={styles.toolbarBtnDone}
                  onClick={finishZone}
                  disabled={currentPts.length < 3}
                >
                  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                    <polyline points="20 6 9 17 4 12"/>
                  </svg>
                  완료 ({currentPts.length}pt)
                </button>
                <button className={styles.toolbarBtnCancel} onClick={cancelDraw}>
                  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                    <line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/>
                  </svg>
                  취소
                </button>
                <span className={styles.toolbarHint}>최소 3개 꼭짓점</span>
              </div>
            )}
          </div>
        </div>

        {/* 사이드 패널 */}
        <div className={styles.panel}>
          {!isDrawing && (
            <button
              className={styles.btnDrawStart}
              onClick={() => setIsDrawing(true)}
              disabled={!aiConnected}
              title={!aiConnected ? 'AI 서버 연결 후 사용 가능' : undefined}
            >
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <polygon points="3 11 22 2 13 21 11 13 3 11"/>
              </svg>
              그리기 시작
            </button>
          )}

          <div className={styles.panelHeader}>
            <div className={styles.panelTitleRow}>
              <svg className={styles.panelIcon} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round">
                <polygon points="1 6 1 22 8 18 16 22 23 18 23 2 16 6 8 2 1 6"/>
                <line x1="8" y1="2" x2="8" y2="18"/>
                <line x1="16" y1="6" x2="16" y2="22"/>
              </svg>
              <span className={styles.panelTitle}>등록된 구역</span>
            </div>
            <span className={styles.zoneCnt} data-numeric>{zones.length}</span>
          </div>

          <div className={styles.zoneList}>
            {zones.length === 0 ? (
              <div className={styles.emptyZone}>
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
                  <polygon points="3 11 22 2 13 21 11 13 3 11"/>
                </svg>
                등록된 구역이 없습니다
              </div>
            ) : zones.map((zone, i) => (
              <div key={i} className={styles.zoneItem}>
                <span className={styles.zoneDot} style={{ backgroundColor: bgrToHex(zone.color) }} />
                <span className={styles.zoneName}>{zone.name}</span>
                <span className={styles.zonePts} data-numeric>{zone.pts.length}pt</span>
                <button className={styles.zoneDeleteBtn} onClick={() => deleteZone(i)} title="구역 삭제">
                  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                    <line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/>
                  </svg>
                </button>
              </div>
            ))}
          </div>

          <div className={styles.panelFooter}>
            {status && (
              <div className={`${styles.statusMsg} ${status.ok ? styles.statusOk : styles.statusErr}`}>
                {status.msg}
              </div>
            )}
            <button className={styles.btnSave} onClick={saveZones} disabled={zones.length === 0}>
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round">
                <path d="M19 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h11l5 5v11a2 2 0 0 1-2 2z"/>
                <polyline points="17 21 17 13 7 13 7 21"/>
                <polyline points="7 3 7 8 15 8"/>
              </svg>
              AI 서버에 저장
            </button>
            <button className={styles.btnDeleteAll} onClick={deleteAllZones} disabled={zones.length === 0}>
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round">
                <polyline points="3 6 5 6 21 6"/>
                <path d="M19 6l-1 14a2 2 0 0 1-2 2H8a2 2 0 0 1-2-2L5 6"/>
                <path d="M10 11v6"/><path d="M14 11v6"/>
              </svg>
              전체 삭제
            </button>
          </div>
        </div>
      </div>
    </div>
  )
}

export default ZonePage
