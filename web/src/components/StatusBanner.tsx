interface StatusBannerProps {
  kind: 'loading' | 'error' | 'empty';
  title: string;
  detail?: string;
  onRetry?: () => void;
}

export function StatusBanner({ kind, title, detail, onRetry }: StatusBannerProps) {
  return (
    <div
      role={kind === 'error' ? 'alert' : 'status'}
      className="flex flex-col items-center justify-center gap-2 rounded-xl border border-[var(--border)]
        bg-[var(--surface)] px-6 py-10 text-center"
    >
      {kind === 'loading' ? (
        <span
          aria-hidden="true"
          className="h-8 w-8 animate-spin rounded-full border-2 border-[var(--border-strong)] border-t-[var(--accent)]"
        />
      ) : (
        <span
          aria-hidden="true"
          className="flex h-8 w-8 items-center justify-center rounded-full text-sm font-semibold"
          style={{
            backgroundColor: kind === 'error' ? 'var(--critical-solid)' : 'var(--surface-2)',
            color: kind === 'error' ? '#fff' : 'var(--ink-muted-text)',
          }}
        >
          {kind === 'error' ? (
            '!'
          ) : (
            // A small 2x2 board-of-slots glyph (evokes the Connect 4 grid) so
            // the empty state reads as a designed placeholder, not a broken/
            // missing icon (the previous bare en-dash "–" looked like a
            // failed-to-load glyph).
            <svg width="18" height="18" viewBox="0 0 18 18" fill="none" aria-hidden="true">
              <circle cx="5.5" cy="5.5" r="2.4" stroke="currentColor" strokeWidth="1.4" />
              <circle cx="12.5" cy="5.5" r="2.4" stroke="currentColor" strokeWidth="1.4" />
              <circle cx="5.5" cy="12.5" r="2.4" stroke="currentColor" strokeWidth="1.4" />
              <circle cx="12.5" cy="12.5" r="2.4" stroke="currentColor" strokeWidth="1.4" />
            </svg>
          )}
        </span>
      )}
      <p className="font-medium text-[var(--ink)]">{title}</p>
      {detail ? <p className="max-w-md text-sm font-medium text-[var(--ink-2)]">{detail}</p> : null}
      {onRetry ? (
        <button
          type="button"
          onClick={onRetry}
          className="mt-2 rounded-md border border-[var(--border-strong)] bg-[var(--surface-2)] px-3 py-1.5 text-sm
            font-medium text-[var(--ink)] hover:bg-[var(--surface)] cursor-pointer"
        >
          Retry
        </button>
      ) : null}
    </div>
  );
}
