/** Layout-matched loading placeholders (C3). The production backend is a
 * free-tier Render instance that sleeps; a cold first request can take
 * ~30s, during which the old UI showed a single centered spinner inside an
 * otherwise-empty, shrink-wrapped card -- an acre of dead space that reads
 * as broken, not "loading". These skeletons instead approximate the REAL
 * layout that's about to appear (a grid of agent cards, a chart + table
 * frame, a champion stat strip) so the cold-start wait reads as "the real
 * content is arriving", not a blank screen with a spinner glued to it.
 *
 * `animate-pulse` is a Tailwind utility (opacity keyframe); it's already
 * covered by index.css's global `prefers-reduced-motion` override (which
 * clamps every animation's duration to ~0), so no separate reduced-motion
 * branch is needed here. */

function Block({ className }: { className: string }) {
  return <span aria-hidden="true" className={`block animate-pulse rounded-md bg-[var(--surface-2)] ${className}`} />;
}

export function AgentCardSkeleton() {
  return (
    <div className="flex flex-col gap-2.5 rounded-xl border border-[var(--border)] bg-[var(--surface)] p-4">
      <div className="flex items-start justify-between gap-2">
        <div className="flex min-w-0 flex-1 flex-col gap-1.5">
          <Block className="h-4 w-2/3" />
          <Block className="h-3 w-4/5" />
        </div>
        <Block className="h-4 w-12 shrink-0" />
      </div>
      <div className="flex gap-1.5">
        <Block className="h-4 w-14" />
        <Block className="h-4 w-16" />
      </div>
      <div className="grid grid-cols-2 gap-x-3 gap-y-2">
        {Array.from({ length: 6 }, (_, i) => (
          <Block key={i} className="h-7 w-full" />
        ))}
      </div>
      <Block className="mt-1 h-8 w-2/5" />
    </div>
  );
}

export function AgentsGridSkeleton({ count = 6 }: { count?: number }) {
  return (
    <div role="status" aria-label="Loading agents…" className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3">
      {Array.from({ length: count }, (_, i) => (
        <AgentCardSkeleton key={i} />
      ))}
    </div>
  );
}

export function ParetoPlotSkeleton() {
  return (
    <div className="rounded-xl border border-[var(--border)] bg-[var(--surface)] p-4">
      <div className="mb-3 flex items-center justify-between">
        <Block className="h-4 w-56" />
        <Block className="h-7 w-40" />
      </div>
      <Block className="h-[340px] w-full" />
    </div>
  );
}

export function LeaderboardTableSkeleton({ rows = 6 }: { rows?: number }) {
  return (
    <div className="overflow-hidden rounded-xl border border-[var(--border)]">
      <div className="bg-[var(--surface-2)] px-3 py-2.5">
        <Block className="h-3 w-full max-w-md" />
      </div>
      <div className="divide-y divide-[var(--border)]">
        {Array.from({ length: rows }, (_, i) => (
          <div key={i} className="flex items-center gap-4 px-3 py-3">
            <Block className="h-4 w-32" />
            <Block className="h-4 w-16" />
            <Block className="ml-auto h-4 w-12" />
            <Block className="h-4 w-12" />
            <Block className="h-4 w-14" />
          </div>
        ))}
      </div>
    </div>
  );
}

export function LeaderboardSkeleton() {
  return (
    <div role="status" aria-label="Loading leaderboard…" className="flex flex-col gap-6">
      <ParetoPlotSkeleton />
      <LeaderboardTableSkeleton />
    </div>
  );
}

export function ChampionStripSkeleton() {
  return (
    <div
      role="status"
      aria-label="Loading current champion…"
      className="mt-2 grid w-full max-w-md grid-cols-3 gap-x-4 gap-y-3 rounded-xl border border-[var(--border)] bg-[var(--surface)] px-6 py-4"
    >
      {Array.from({ length: 3 }, (_, i) => (
        <div key={i} className="flex flex-col items-center gap-1.5">
          <Block className="h-7 w-16" />
          <Block className="h-3 w-14" />
        </div>
      ))}
    </div>
  );
}
