import type { ThemeMode } from '../lib/theme';

export type Screen = 'play' | 'leaderboard';

interface NavBarProps {
  screen: Screen;
  onScreen: (s: Screen) => void;
  theme: ThemeMode;
  onToggleTheme: () => void;
}

export function NavBar({ screen, onScreen, theme, onToggleTheme }: NavBarProps) {
  return (
    <header className="sticky top-0 z-20 border-b border-[var(--border)] bg-[var(--page)]/95 backdrop-blur">
      <div className="mx-auto flex max-w-5xl items-center justify-between gap-3 px-4 py-3">
        <div className="flex items-center gap-2">
          <span
            aria-hidden="true"
            className="grid h-7 w-7 place-items-center rounded-md text-xs font-bold"
            style={{ backgroundColor: 'var(--accent-solid)', color: 'var(--accent-solid-ink)' }}
          >
            N4
          </span>
          <span className="text-sm font-semibold tracking-tight text-[var(--ink)]">NeuroFour</span>
        </div>

        <nav aria-label="Main" className="flex items-center gap-1 rounded-lg bg-[var(--surface-2)] p-1">
          <button
            type="button"
            onClick={() => onScreen('play')}
            aria-current={screen === 'play' ? 'page' : undefined}
            className={`rounded-md px-3 py-1.5 text-sm font-medium cursor-pointer transition-colors ${
              screen === 'play'
                ? 'bg-[var(--surface)] text-[var(--ink)] shadow-sm'
                : 'text-[var(--ink-2)] hover:text-[var(--ink)]'
            }`}
          >
            Play
          </button>
          <button
            type="button"
            onClick={() => onScreen('leaderboard')}
            aria-current={screen === 'leaderboard' ? 'page' : undefined}
            className={`rounded-md px-3 py-1.5 text-sm font-medium cursor-pointer transition-colors ${
              screen === 'leaderboard'
                ? 'bg-[var(--surface)] text-[var(--ink)] shadow-sm'
                : 'text-[var(--ink-2)] hover:text-[var(--ink)]'
            }`}
          >
            Leaderboard
          </button>
        </nav>

        <button
          type="button"
          onClick={onToggleTheme}
          aria-pressed={theme === 'dark'}
          className="rounded-md border border-[var(--border)] px-2.5 py-1.5 text-sm text-[var(--ink-2)]
            hover:text-[var(--ink)] hover:border-[var(--border-strong)] cursor-pointer"
        >
          <span aria-hidden="true">{theme === 'dark' ? '☾' : '☀'}</span>
          <span className="sr-only">Switch to {theme === 'dark' ? 'light' : 'dark'} mode</span>
        </button>
      </div>
    </header>
  );
}
