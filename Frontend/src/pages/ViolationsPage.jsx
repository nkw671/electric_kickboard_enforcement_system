import { useState, useEffect } from 'react'
import { TYPE_COLOR, VIOLATION_TYPES } from '../constants'
import useApi from '../hooks/useApi'
import styles from './ViolationsPage.module.css'

function ViolationsPage() {
  const [filter, setFilter] = useState('전체')
  const [selected, setSelected] = useState(null)

  const { data: violations, loading, error } = useApi('/api/violations')

  if (loading) return <div className={styles.status}>불러오는 중...</div>
  if (error) return <div className={styles.statusError}>서버에 연결할 수 없습니다.</div>

  const filtered =
    filter === '전체' ? violations : violations.filter((v) => v.type === filter)

  useEffect(() => {
    if (!selected) return
    const onKeyDown = (e) => {
      if (e.key === 'Escape') setSelected(null)
    }
    window.addEventListener('keydown', onKeyDown)
    return () => window.removeEventListener('keydown', onKeyDown)
  }, [selected])

  return (
    <div className={styles.page}>
      <div className={styles.header}>
        <h2 className={styles.title}>위반 기록</h2>
        <div className={styles.filters}>
          {VIOLATION_TYPES.map((t) => (
            <button
              key={t}
              className={`${styles.filterBtn} ${filter === t ? styles.active : ''}`}
              onClick={() => setFilter(t)}
            >
              {t}
            </button>
          ))}
        </div>
      </div>

      <table className={styles.table}>
        <thead>
          <tr>
            <th>번호</th>
            <th>일시</th>
            <th>위반 유형</th>
            <th>카메라</th>
            <th>신뢰도</th>
            <th>상세</th>
          </tr>
        </thead>
        <tbody>
          {filtered.map((v) => (
            <tr key={v.id}>
              <td>#{v.id}</td>
              <td>{v.timestamp}</td>
              <td>
                <span
                  className={styles.badge}
                  style={{ backgroundColor: TYPE_COLOR[v.type] || '#64748b' }}
                >
                  {v.type}
                </span>
              </td>
              <td>{v.camera}</td>
              <td>{v.confidence}%</td>
              <td>
                <button
                  className={styles.detailBtn}
                  onClick={() => setSelected(v)}
                >
                  보기
                </button>
              </td>
            </tr>
          ))}
        </tbody>
      </table>

      {/* 상세 모달 */}
      {selected && (
        <div className={styles.modalOverlay} onClick={() => setSelected(null)}>
          <div className={styles.modal} onClick={(e) => e.stopPropagation()}>
            <div className={styles.modalHeader}>
              <span>위반 상세 정보</span>
              <button className={styles.closeBtn} onClick={() => setSelected(null)}>
                ✕
              </button>
            </div>
            <div className={styles.modalBody}>
              <div className={styles.imagePlaceholder}>
                캡처 이미지<br />
                <small>백엔드 연결 후 표시</small>
              </div>
              <div className={styles.infoGrid}>
                <div className={styles.infoItem}>
                  <span className={styles.infoLabel}>위반 유형</span>
                  <span
                    className={styles.badge}
                    style={{ backgroundColor: TYPE_COLOR[selected.type] }}
                  >
                    {selected.type}
                  </span>
                </div>
                <div className={styles.infoItem}>
                  <span className={styles.infoLabel}>감지 시각</span>
                  <span>{selected.timestamp}</span>
                </div>
                <div className={styles.infoItem}>
                  <span className={styles.infoLabel}>카메라</span>
                  <span>{selected.camera}</span>
                </div>
                <div className={styles.infoItem}>
                  <span className={styles.infoLabel}>신뢰도</span>
                  <span>{selected.confidence}%</span>
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
