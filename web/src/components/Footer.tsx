export function Footer() {
  return (
    <footer className="mt-8 border-t border-[var(--border)]">
      <div className="mx-auto flex max-w-5xl flex-col items-center justify-between gap-2 px-4 py-6 text-xs text-[var(--ink-2)] sm:flex-row">
        <p>
          <a
            href="https://github.com/ethan-haas/neurofour"
            target="_blank"
            rel="noreferrer noopener"
            className="font-medium underline underline-offset-2 hover:text-[var(--ink)]"
          >
            View the source on GitHub
          </a>
        </p>
        <p className="text-[var(--ink-muted-text)]">
          The backend runs on a free tier and may cold-start (first request can take ~30s).
        </p>
      </div>
    </footer>
  );
}
