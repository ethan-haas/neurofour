import { useLeaderboard } from '../hooks/useLeaderboard';
import { ChampionStripSkeleton } from './Skeleton';
import { formatBytes, formatPct } from '../lib/format';

export function AboutScreen() {
  const state = useLeaderboard();
  const champion =
    state.status === 'success' ? state.data.agents.find((a) => a.name === state.data.headline.agent) : undefined;

  return (
    <div className="mx-auto flex max-w-3xl flex-col gap-10 px-4 py-10">
      {/* Hero */}
      <section className="flex flex-col items-center gap-4 text-center">
        <h1 className="text-balance text-3xl font-bold tracking-tight text-[var(--ink)] sm:text-4xl">
          How much Connect Four strength fits in a byte?
        </h1>
        <p className="text-balance max-w-xl text-sm text-[var(--ink-2)] sm:text-base">
          NeuroFour ranks Connect Four agents not by raw strength alone, but by strength <em>per byte and per FLOP</em> --
          under a fixed compute budget, the smartest small agent wins.
        </p>

        {state.status === 'success' && champion ? (
          // Fixed 3-across grid at every width (not `flex-wrap`, which
          // wrapped to a ragged 2+1 on narrow screens -- the third stat
          // orphaned and re-centered on its own row). `min-w-0` on each cell
          // lets the numbers shrink instead of forcing the grid wider than
          // its container.
          <div className="mt-2 grid w-full max-w-md grid-cols-3 gap-x-3 gap-y-3 rounded-xl border border-[var(--border)] bg-[var(--surface)] px-4 py-4 sm:gap-x-8 sm:px-6">
            <div className="flex min-w-0 flex-col items-center">
              <span className="tabular text-xl font-bold text-[var(--ink)] sm:text-2xl">{formatBytes(champion.size_bytes)}</span>
              <span className="text-xs text-[var(--ink-muted-text)]">champion size</span>
            </div>
            <div className="flex min-w-0 flex-col items-center">
              <span className="tabular text-xl font-bold text-[var(--ink)] sm:text-2xl">{formatPct(champion.optimality)}</span>
              <span className="text-xs text-[var(--ink-muted-text)]">optimality</span>
            </div>
            <div className="flex min-w-0 flex-col items-center">
              <span className="tabular text-xl font-bold text-[var(--ink)] sm:text-2xl">{champion.neurogolf_score.toFixed(1)}</span>
              <span className="text-xs text-[var(--ink-muted-text)]">NeuroFour score</span>
            </div>
          </div>
        ) : state.status === 'loading' ? (
          <ChampionStripSkeleton />
        ) : null}
      </section>

      {/* What the score is */}
      <section className="flex flex-col gap-3">
        <h2 className="text-lg font-semibold text-[var(--ink)]">What the NeuroFour Score is</h2>
        <p className="text-sm text-[var(--ink-2)]">
          Every agent plays a fixed set of benchmark positions against the exact solver's ground truth. Two numbers come
          out of that: <strong>optimality</strong> (the fraction of positions where the agent played a game-theoretically
          optimal move) and <strong>soundness</strong> (how rarely it plays a move that outright loses when a
          non-losing one was available). Those combine with the agent's artifact size into one composite:
        </p>
        <pre
          className="overflow-x-auto rounded-lg border border-[var(--border)] bg-[var(--surface-2)] p-3 pr-5 text-sm text-[var(--ink)]"
          tabIndex={0}
          role="region"
          aria-label="NeuroFour Score formula, scrollable on narrow viewports"
        >
          {`NeuroFour Score = 100 * (0.85 * optimality + 0.15 * soundness)
                  / (1 + 0.15 * log2(1 + size_kb))`}
        </pre>
        <p className="text-sm text-[var(--ink-2)]">
          The denominator is a size penalty: it barely dents the score for a tiny agent, but grows (logarithmically) as
          the artifact gets bigger, so two agents with identical strength are ranked apart by how many bytes they
          needed to get there.
        </p>
        <p className="text-xs text-[var(--ink-muted-text)]">
          Honest note: this benchmark is based on the idea of scoring strength-per-byte/compute the way a golf handicap
          scores skill-per-stroke — but everywhere in this app, the number you see is called the <strong>NeuroFour
          Score</strong>, never that other name.
        </p>
      </section>

      {/* Compute budget */}
      <section className="flex flex-col gap-3">
        <h2 className="text-lg font-semibold text-[var(--ink)]">The 5M-FLOP compute budget</h2>
        <p className="text-sm text-[var(--ink-2)]">
          Every agent gets the same ceiling: at most <strong>5,000,000 floating-point operations per move</strong>. An
          agent that blows past that budget is marked <em>over budget</em> and excluded from the cheap Pareto
          frontier — it can still be played and shown on the leaderboard, but it isn't competing for the crown. The
          budget is what makes "strength per byte" a fair fight: without a compute ceiling, an agent could just
          brute-force search deeper and deeper regardless of how small its stored weights are.
        </p>
      </section>

      {/* Why 0-byte wins */}
      <section className="flex flex-col gap-3">
        <h2 className="text-lg font-semibold text-[var(--ink)]">Why the champion is a 0-byte agent</h2>
        <p className="text-sm text-[var(--ink-2)]">
          The current champion ships no learned weights at all — it's a pure bitboard alpha-beta search. Under a
          tight, fixed FLOP budget, a few million node visits of exact tactical search consistently out-plays a small
          learned value network evaluating positions in one shot: search "spends" its whole compute budget looking
          ahead, while a network spends most of its budget on the forward pass itself. That's the headline finding of
          this benchmark: at this specific budget, <strong>search beats learned nets</strong>. A bigger budget, or a
          much smaller/cheaper network architecture, could tip that balance back — the leaderboard and Pareto plot
          are how you'd see that happen if it ever does.
        </p>
      </section>

      {/* Pareto frontier */}
      <section className="flex flex-col gap-3">
        <h2 className="text-lg font-semibold text-[var(--ink)]">What the Pareto frontier means</h2>
        <p className="text-sm text-[var(--ink-2)]">
          An agent is on the frontier if no other agent beats it on strength while costing the same or less on every
          axis (size AND FLOPs). The Leaderboard's scatter plot draws that frontier as a line: agents to its
          lower-right are dominated — something on the line is at least as strong for less cost — while agents on
          the line represent a genuine size/strength trade-off worth knowing about.
        </p>
      </section>

      {/* Exact solver */}
      <section className="flex flex-col gap-3">
        <h2 className="text-lg font-semibold text-[var(--ink)]">The exact solver is ground truth</h2>
        <p className="text-sm text-[var(--ink-2)]">
          Connect Four is a solved game. The <strong>Oracle</strong> agent runs a real, exact game-tree solver — it
          never guesses, and every benchmark position's "correct" move is defined by what the Oracle plays there. It
          is intentionally far over the compute budget (a full search costs vastly more than 5M FLOPs), so it never
          competes for the NeuroFour Score crown; it exists purely to grade every other agent.
        </p>
      </section>
    </div>
  );
}
