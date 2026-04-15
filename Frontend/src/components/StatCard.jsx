import styles from './StatCard.module.css'

function StatCard({ label, value, color }) {
  return (
    <div className={styles.card} style={{ borderTopColor: color }}>
      <div className={styles.value} style={{ color }}>{value}</div>
      <div className={styles.label}>{label}</div>
    </div>
  )
}

export default StatCard
