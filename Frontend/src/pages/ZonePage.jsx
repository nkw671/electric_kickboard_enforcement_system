import { useState, useEffect, useRef } from 'react'
import { aiUrl } from '../config'
import styles from './ZonePage.module.css'

// 스트림이 로드되기 전(미연결 등) 사용할 기본 해상도. 연결되면 영상 실제 해상도로 자동 교체된다.
const DEFAULT_W = 800
const DEFAULT_H = 450

// 첫 꼭짓점에서 이 거리(화면 px) 안을 클릭하면 도형을 닫는다.
const CLOSE_DISTANCE_PX = 12

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
  // 삭제 확인 팝업: { title, message, confirmLabel, onConfirm } 또는 null
  const [confirmDialog, setConfirmDialog] = useState(null)

  // AI 서버(vast.ai)는 필요할 때만 켜두므로, 연결될 때까지 5초마다 다시 시도한다.
  // 연결되면 영상 스트림 에러 상태도 풀어서 영상을 다시 불러온다.
  useEffect(() => {
    if (aiConnected) return
    let cancelled = false
    const load = () => {
      fetch(aiUrl('/zones'))
        .then(r => {
          if (!r.ok) throw new Error(`HTTP ${r.status}`)
          return r.json()
        })
        .then(data => {
          if (cancelled) return
          // 연결 전에 그려둔(아직 저장 안 한) 구역이 있으면 덮어쓰지 않는다.
          setZones(prev => (prev.length ? prev : data.zones || []))
          setStreamError(false)
          setAiConnected(true)
        })
        .catch(() => {})
    }
    load()
    const timer = setInterval(load, 5000)
    return () => {
      cancelled = true
      clearInterval(timer)
    }
  }, [aiConnected])

  // 영상 스트림의 실제 해상도(naturalWidth/Height)를 폴링하여 좌표계를 영상에 맞춘다.
  // AI(detection.py INFER_SIZE)가 어떤 해상도를 쓰든 자동으로 일치하므로 하드코딩이 불필요하다.
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

    // 저장된 존 렌더링
    zones.forEach(zone => {
      if (zone.pts.length < 2) return
      const [b, g, r] = zone.color
      const colorRgb = `rgb(${r},${g},${b})`

      ctx.beginPath()
      ctx.moveTo(zone.pts[0][0], zone.pts[0][1])
      zone.pts.slice(1).forEach(([x, y]) => ctx.lineTo(x, y))
      ctx.closePath()
      ctx.fillStyle = `rgba(${r},${g},${b},0.25)`
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
      ctx.fillStyle = colorRgb
      ctx.font = 'bold 14px sans-serif'
      ctx.fillText(zone.name, cx - 20, cy + 5)
    })

    // 현재 그리는 중인 폴리곤
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

  const handleCanvasClick = (e) => {
    if (!isDrawing) return
    const canvas = canvasRef.current
    const rect = canvas.getBoundingClientRect()
    const scale = videoDims.w / rect.width
    const x = Math.round((e.clientX - rect.left) * scale)
    const y = Math.round((e.clientY - rect.top) * (videoDims.h / rect.height))

    // 꼭짓점이 3개 이상일 때 첫 점 근처를 클릭하면 구역을 완성한다.
    if (currentPts.length >= 3) {
      const [fx, fy] = currentPts[0]
      if (Math.hypot(x - fx, y - fy) <= CLOSE_DISTANCE_PX * scale) {
        finishZone()
        return
      }
    }
    setCurrentPts(prev => [...prev, [x, y]])
  }

  const undoPoint = () => {
    setCurrentPts(prev => prev.slice(0, -1))
  }

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

  // 그리는 중 단축키: Enter = 완성, Backspace = 마지막 점 되돌리기, Esc = 그리기 취소
  // (finishZone이 현재 꼭짓점을 쓰므로 currentPts가 바뀔 때마다 리스너를 다시 건다)
  useEffect(() => {
    if (!isDrawing || confirmDialog) return
    const onKeyDown = (e) => {
      if (e.key === 'Enter') {
        e.preventDefault()
        finishZone()
      } else if (e.key === 'Backspace') {
        e.preventDefault()
        setCurrentPts(prev => prev.slice(0, -1))
      } else if (e.key === 'Escape') {
        setCurrentPts([])
        setIsDrawing(false)
      }
    }
    window.addEventListener('keydown', onKeyDown)
    return () => window.removeEventListener('keydown', onKeyDown)
  })

  const deleteZone = (idx) => {
    setZones(prev => prev.filter((_, i) => i !== idx).map((z, i) => ({ ...z, name: `Zone-${i + 1}` })))
  }

  // 확인 팝업은 Esc로도 닫을 수 있다.
  useEffect(() => {
    if (!confirmDialog) return
    const onKeyDown = (e) => {
      if (e.key === 'Escape') setConfirmDialog(null)
    }
    window.addEventListener('keydown', onKeyDown)
    return () => window.removeEventListener('keydown', onKeyDown)
  }, [confirmDialog])

  const askDeleteZone = (idx) => {
    setConfirmDialog({
      title: `${zones[idx].name} 구역을 삭제할까요?`,
      message: '삭제 후 AI 서버에 저장을 눌러야 반영됩니다.',
      confirmLabel: '삭제',
      onConfirm: () => deleteZone(idx),
    })
  }

  const askDeleteAllZones = () => {
    setConfirmDialog({
      title: '등록된 구역을 모두 삭제할까요?',
      message: 'AI 서버에서도 바로 삭제되며 되돌릴 수 없습니다.',
      confirmLabel: '전체 삭제',
      onConfirm: deleteAllZones,
    })
  }

  const handleConfirm = () => {
    confirmDialog.onConfirm()
    setConfirmDialog(null)
  }

  const saveZones = async () => {
    try {
      const res = await fetch(aiUrl('/zones'), {
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
      const res = await fetch(aiUrl('/zones'), { method: 'DELETE' })
      if (!res.ok) throw new Error()
      setZones([])
      setStatus({ ok: true, msg: '전체 삭제 완료' })
    } catch {
      setStatus({ ok: false, msg: '삭제 실패 — AI 서버 연결 확인' })
    }
    setTimeout(() => setStatus(null), 3000)
  }

  return (
    <div className={styles.page}>
      <div className={styles.header}>
        <h2 className={styles.title}>인도 구역 설정</h2>
        <p className={styles.desc}>
          영상에서 인도 영역을 다각형으로 그려 주세요. 이 영역 안에서 주행하는 킥보드는 인도 주행으로 단속됩니다.
          <br />
          구역을 그린 뒤 <strong>AI 서버에 저장</strong>을 눌러야 반영됩니다.
        </p>
      </div>

      <div className={styles.layout}>
        {/* 캔버스 영역 */}
        <div className={styles.canvasWrap}>
          <div className={styles.canvasHeader}>
            <span className={`${styles.connDot} ${aiConnected ? styles.connOn : styles.connOff}`} />
            <span className={styles.connLabel}>{aiConnected ? 'AI 서버 연결됨' : 'AI 서버 미연결'}</span>
            {isDrawing && <span className={styles.drawingBadge}>그리는 중 — 클릭으로 꼭짓점 추가</span>}
          </div>
          <div className={styles.canvasStage}>
            {aiConnected && !streamError ? (
              <img
                ref={imgRef}
                src={aiUrl('/video/stream')}
                className={styles.canvasVideo}
                alt="stream"
                onError={() => {
                  // 스트림이 끊기면 연결 끊김으로 보고 위의 재시도 루프를 다시 돌린다.
                  setStreamError(true)
                  setAiConnected(false)
                }}
              />
            ) : (
              <div className={styles.canvasPlaceholder}>
                AI 서버 연결 후 활성화
                <small>AI 서버가 켜지면 자동으로 다시 연결됩니다</small>
              </div>
            )}
            <canvas
              ref={canvasRef}
              width={videoDims.w}
              height={videoDims.h}
              className={`${styles.canvas} ${isDrawing ? styles.canvasDrawing : ''}`}
              onClick={handleCanvasClick}
            />
          </div>
          <div className={styles.canvasFooter}>
            {!isDrawing ? (
              <button className={styles.btnPrimary} onClick={() => setIsDrawing(true)}>
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <polygon points="3 11 22 2 13 21 11 13 3 11"/>
                </svg>
                그리기 시작
              </button>
            ) : (
              <>
                <button className={styles.btnSuccess} onClick={finishZone} disabled={currentPts.length < 3}>
                  완성 <kbd className={styles.kbdOnDark}>Enter</kbd>
                </button>
                <button className={styles.btnGhost} onClick={undoPoint} disabled={currentPts.length === 0}>
                  되돌리기 <kbd className={styles.kbd}>Backspace</kbd>
                </button>
                <button className={styles.btnGhost} onClick={cancelDraw}>
                  취소 <kbd className={styles.kbd}>Esc</kbd>
                </button>
              </>
            )}
            <span className={styles.hint}>
              {isDrawing
                ? '꼭짓점 3개 이상 찍은 뒤 첫 점을 다시 클릭하면 완성'
                : '꼭짓점을 3개 이상 찍어 구역을 만듭니다'}
            </span>
          </div>
        </div>

        {/* 사이드 패널 */}
        <div className={styles.panel}>
          <div className={styles.panelHeader}>
            <span className={styles.panelTitle}>등록된 구역</span>
            <span className={styles.zoneCnt}>{zones.length}개</span>
          </div>

          <div className={styles.zoneList}>
            {zones.length === 0 ? (
              <div className={styles.emptyZone}>등록된 구역이 없습니다</div>
            ) : (
              zones.map((zone, i) => (
                <div key={i} className={styles.zoneItem}>
                  <span className={styles.zoneDot} style={{ backgroundColor: bgrToHex(zone.color) }} />
                  <span className={styles.zoneName}>{zone.name}</span>
                  <span className={styles.zonePts}>꼭짓점 {zone.pts.length}개</span>
                  <button className={styles.zoneDeleteBtn} onClick={() => askDeleteZone(i)} aria-label={`${zone.name} 삭제`}>✕</button>
                </div>
              ))
            )}
          </div>

          <div className={styles.panelFooter}>
            {status && (
              <div className={`${styles.statusMsg} ${status.ok ? styles.statusOk : styles.statusErr}`}>
                {status.msg}
              </div>
            )}
            <button className={styles.btnSave} onClick={saveZones}>
              AI 서버에 저장
            </button>
            <button className={styles.btnDelete} onClick={askDeleteAllZones} disabled={zones.length === 0}>
              전체 삭제
            </button>
          </div>
        </div>
      </div>

      {confirmDialog && (
        <div className={styles.dialogOverlay} onClick={() => setConfirmDialog(null)}>
          <div className={styles.dialog} role="alertdialog" aria-modal="true" onClick={(e) => e.stopPropagation()}>
            <p className={styles.dialogTitle}>{confirmDialog.title}</p>
            <p className={styles.dialogMessage}>{confirmDialog.message}</p>
            <div className={styles.dialogActions}>
              {/* 실수로 Enter를 눌러도 삭제되지 않도록 취소 버튼에 포커스를 둔다. */}
              <button className={styles.btnGhost} onClick={() => setConfirmDialog(null)} autoFocus>취소</button>
              <button className={styles.btnDanger} onClick={handleConfirm}>{confirmDialog.confirmLabel}</button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

export default ZonePage
