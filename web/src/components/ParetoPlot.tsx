import { useMemo, useState } from 'react';
import type { LeaderboardAgent } from '../types';
import { formatBytes, formatFlops, formatPct } from '../lib/format';

export type CostAxis = 'size' | 'flops';

interface ParetoPlotProps {
  agents: LeaderboardAgent[];
  axis: CostAxis;
  onAxisChange: (a: CostAxis) => void;
  /** The headline/flagship agent name, read from `leaderboard.headline.agent`
   * by the caller — never hardcoded here, so the highlighted point always
   * tracks whichever agent the backend currently considers the flagship. */
  flagship?: string | null;
}

const CAT_COLOR: Record<LeaderboardAgent['kind'], string> = {
  table: 'var(--cat-table)',
  nn: 'var(--cat-nn)',
  search: 'var(--cat-search-mark)',
  heuristic: 'var(--cat-heuristic)',
  random: 'var(--cat-random)',
};

const CAT_LABEL: Record<LeaderboardAgent['kind'], string> = {
  table: 'Table',
  nn: 'Neural net',
  search: 'Search',
  heuristic: 'Heuristic',
  random: 'Random',
};

const KIND_ORDER: LeaderboardAgent['kind'][] = ['table', 'nn', 'search', 'heuristic', 'random'];

/** The human-facing name for an agent. The API serves `display_name` ("Zero");
 * `name` is the internal registry id ("neurofour-net14") and must never be the
 * label a user (or a screen reader) is given -- the rest of the app, including
 * this chart's own legend, shows the display name, so the plot's points, labels
 * and detail panel have to agree with it. Falls back to the id if an agent
 * predates the display map. */
const labelOf = (a: Pick<LeaderboardAgent, 'name' | 'display_name'>): string => a.display_name ?? a.name;

// Tier boundaries mirror app/neurogolf/config.py's TIERS caps (nano/micro/
// mini/small size_bytes, excluding the unbounded "open" tier) -- these are
// the size-axis reference lines that actually mean something (a budget-tier
// cutoff), so ticks are always drawn from this list, never an arbitrary
// round number.
const SIZE_TIER_BOUNDARIES = [4096, 32768, 262144, 2097152];
const SIZE_TIER_LABELS = ['4 KB', '32 KB', '256 KB', '2 MB'];

/** Pick the size-axis domain data-driven from the agents actually being
 * plotted: the smallest tier boundary that's comfortably (1.5x) past the
 * largest real agent, so e.g. a ~7.9 KB largest agent draws only through the
 * 32 KB micro-cap tick (dropping the empty 256 KB / 2 MB expanse) instead of
 * always drawing out to 2 MB. Stays correct if a bigger agent is added later
 * -- it just grows to the next boundary that covers it. */
function sizeAxisConfig(agents: LeaderboardAgent[]): {
  ticks: number[];
  tickLabels: string[];
  domainMax: number;
} {
  const maxSize = agents.reduce((m, a) => Math.max(m, a.size_bytes), 0);
  // Domain ends at the SMALLEST tier boundary that actually CONTAINS the
  // largest agent -- not 1.5x past it. The old 1.5x headroom tipped a 24.7 KB
  // largest agent (1.5x = 37 KB) just over the 32 KB micro cap onto the NEXT
  // boundary (256 KB), so the axis ran to 256 KB while every real point sat
  // below 25 KB -- ~85% of the plot width was empty and all agents crammed
  // into the left ~15% (independently flagged by vision review as the plot
  // "failing at its single most important job"). Capping at the containing
  // boundary spreads the points across the full width; the domainMax's own
  // "+0.4" margin below already keeps the boundary tick clear of the right
  // edge, so no multiplicative headroom is needed. `Math.max(_, 1)` keeps a
  // sane domain when every agent ships 0 bytes (target 1 -> the 4 KB nano
  // tick is still drawn for context).
  const target = Math.max(maxSize, 1);
  const idx = Math.max(
    0,
    SIZE_TIER_BOUNDARIES.findIndex((b) => b >= target) === -1
      ? SIZE_TIER_BOUNDARIES.length - 1
      : SIZE_TIER_BOUNDARIES.findIndex((b) => b >= target),
  );
  const lastBoundary = SIZE_TIER_BOUNDARIES[idx];
  return {
    ticks: [0, ...SIZE_TIER_BOUNDARIES.slice(0, idx + 1)],
    tickLabels: ['0', ...SIZE_TIER_LABELS.slice(0, idx + 1)],
    // A small fixed margin (not the old "+2 log2 units", ~4x, headroom) past
    // the last tick -- just enough that the tick label/reference line isn't
    // flush against the right edge. "+2" left ~40% of the chart width empty
    // past the tick even after the tier-boundary tightening above (data tops
    // out well before the tick, and the tick itself then sat well before the
    // domain's right edge too); "+0.4" keeps the tick close to the right
    // edge so the actual plotted data -- which is always comfortably left of
    // the tick, since the tick is chosen to be >=1.5x the largest point --
    // fills most of the width instead of a fixed multiplicative headroom
    // band nothing is ever plotted into.
    domainMax: Math.log2(1 + lastBoundary / 1024) + 0.4,
  };
}

const FLOP_CAP = 5_000_000;
const FLOPS_TICKS = [0, 1e4, 1e5, 1e6, FLOP_CAP];
const FLOPS_TICK_LABELS = ['0', '10K', '100K', '1M', '5M cap'];
const FLOPS_DOMAIN_MAX = Math.log10(1 + FLOP_CAP) + 0.6;

// The y-axis (optimality) used to be a fixed 0-100% every time, regardless
// of what the actually-plotted agents scored. Optimality on this bench is
// bimodal in practice: real search/NN agents cluster at 85-100%, a
// deliberately-bad `random` reference sits far below (~25-30%), and nothing
// plots in between -- so a fixed 0-100% axis spent most of its vertical
// space on empty gridlines the eye has to scan past to reach the one band
// that actually distinguishes competitive agents from each other (vision
// review: "every competitive agent crams into the top-left 90-100% band").
//
// Zooming to the REAL data range (with padding, and clamped to the true
// [0,1] bound of a ratio metric) uses the canvas honestly: the domain always
// reflects what's actually on the chart, drawn with real labelled ticks (not
// a hidden/truncated scale), and an explicit axis-break glyph plus caption
// whenever the domain doesn't start at 0% -- so a zoomed band never reads as
// "the zero baseline" when it isn't one. A currently-tiny agent roster (e.g.
// the mocked 6-agent fixture, or a future roster that's all clustered
// together) still gets a sane minimum span instead of a division-by-near-
// zero degenerate axis.
const Y_MIN_SPAN = 0.12;
const Y_PAD_FRAC = 0.16;

/** The FIRST zoom pass (span = dataMax - dataMin over every plotted agent)
 * fixed the "0-100% fixed axis" problem but immediately hit a second,
 * subtler one: the `random` kind exists purely as a deliberately-bad
 * reference point (an agent that just picks any legal column) -- it's not a
 * competing agent, and its optimality sits far below every real agent's. A
 * domain that still spans all the way down to include it re-created almost
 * the exact same failure one level up (vision review: "the y-axis spans
 * 15-100% but every competing agent sits at 90-100%... one lone Random dot
 * at ~30% is what stretches the axis"). The domain is now computed from the
 * COMPETING agents only; `random` (and anything else, in a future roster,
 * that isn't a real contender) still gets drawn -- `yScale` below already
 * clamps any point below `domainMin` to the plot's bottom edge, so an
 * excluded outlier renders as a real, focusable, correctly-labelled point
 * sitting at the very bottom of the chart instead of vanishing or lying
 * about its value -- it just no longer gets to dictate how much of the
 * canvas the real competitors are allowed to use. */
function yAxisConfig(agents: LeaderboardAgent[]): { domainMin: number; domainMax: number; ticks: number[] } {
  const competing = agents.filter((a) => a.kind !== 'random');
  const pool = competing.length > 0 ? competing : agents;
  if (pool.length === 0) return { domainMin: 0, domainMax: 1, ticks: [0, 0.25, 0.5, 0.75, 1] };
  let dataMin = Infinity;
  let dataMax = -Infinity;
  for (const a of pool) {
    if (a.optimality < dataMin) dataMin = a.optimality;
    if (a.optimality > dataMax) dataMax = a.optimality;
  }
  const span = Math.max(dataMax - dataMin, Y_MIN_SPAN);
  const pad = span * Y_PAD_FRAC;
  const domainMin = Math.max(0, dataMin - pad);
  const domainMax = Math.min(1, dataMax + pad);
  const n = 5;
  const ticks = Array.from({ length: n }, (_, i) => domainMin + (i / (n - 1)) * (domainMax - domainMin));
  return { domainMin, domainMax, ticks };
}

/** Percent labels for the zoomed axis: honest actual values (e.g. "22%",
 * "61%"), not the old fixed 0/25/50/75/100 -- those would misrepresent
 * where each gridline actually falls once the domain is no longer [0,1].
 * One decimal only kicks in when the whole domain span is tight enough that
 * whole-percent ticks would otherwise collide/round to duplicates. */
function formatYTick(v: number, domainSpan: number): string {
  const decimals = domainSpan < 0.08 ? 1 : 0;
  return `${(v * 100).toFixed(decimals)}%`;
}

const W = 640;
const H = 340;
const PAD_L = 46;
const PAD_R = 16;
const PAD_T = 16;
const PAD_B = 34;

// Points at 0 bytes / 0 FLOPs are real (e.g. pure-code search agents ship no
// artifact) but log(0) is undefined and would otherwise land exactly on the
// y-axis stroke, indistinguishable from the axis itself. Reserve a small
// "zero shelf" of the horizontal range so true-zero points sit just clear of
// the axis line instead of stacking on it, and the rest of the (symlog-ish)
// domain still gets the full remaining range.
const ZERO_SHELF = 0.035;

// Points whose x pixel positions land within this many px of each other are
// visually indistinguishable (e.g. several agents all shipping a 0-byte
// artifact) — dodge them apart horizontally so every point stays visible and
// focusable instead of stacking exactly on top of a neighbor.
const DODGE_THRESHOLD_PX = 10;
const DODGE_STEP_PX = 9;

function xValue(agent: LeaderboardAgent, axis: CostAxis): number {
  return axis === 'size' ? agent.size_bytes : agent.flops_per_move;
}

// The flagship marker draws two extra rings around its point (a background
// "surface" halo and an accent ring, see the render loop below) that extend
// its visual footprint well past its own radius `r` -- a label placed just
// `r + 6` away (the spacing every other point's caption uses) lands right on
// top of that halo, reading as glued to the marker. Clear the *actual*
// visual footprint (r + 6 for the halo/ring, + 6px gap) instead.
const FLAGSHIP_HALO_GAP = 12;

// Rough (deliberately slightly generous) average glyph width for the 10px
// weight-700 label font -- good enough to keep the fit checks below
// conservative; agent names are plain ASCII kebab-case, no need for real
// canvas text measurement here.
const FLAGSHIP_CHAR_WIDTH_PX = 6.6;

function estimateLabelWidth(name: string): number {
  return name.length * FLAGSHIP_CHAR_WIDTH_PX;
}

// The SVG itself is fixed at W=640 CSS px regardless of screen size (its
// viewBox and its `width` attribute match 1:1, so it never scales) -- only
// the OUTER wrapper div scrolls (`overflow-x-auto`) on a viewport narrower
// than that. At scrollLeft=0 -- the state a user lands on before ever
// scrolling, which is exactly this defect's repro ("renders as
// 'neurofour-n…' at the right edge before horizontal scroll") -- the
// visible slice of that 640px canvas is only [0, visibleWidth], where
// visibleWidth is the wrapper's actual measured CSS width (e.g. ~300px at a
// 375px viewport; comfortably >=640 on desktop, where nothing here changes
// behavior). The label's own footprint therefore needs to fit inside that
// physically-visible slice, not just inside the SVG's own 640-wide
// viewBox -- fitting the latter but not the former is precisely how the
// label could still get clipped by the container's edge while remaining
// technically inside the drawable plot area.
function flagshipLabelPlacement(
  cx: number,
  r: number,
  name: string,
  visibleWidth: number,
): { anchor: 'start' | 'end'; x: number } {
  const gap = r + FLAGSHIP_HALO_GAP;
  const textWidth = estimateLabelWidth(name);
  const rightBound = Math.min(visibleWidth, W) - PAD_R;
  const leftBound = PAD_L;

  const rightStart = cx + gap;
  const leftEnd = cx - gap;
  const fitsRight = rightStart + textWidth <= rightBound;
  const fitsLeft = leftEnd - textWidth >= leftBound;

  // Prefer growing right (the direction every other point's caption uses)
  // whenever it actually fits; fall back to growing left (inward, back
  // toward the bulk of the plot) when right doesn't fit but left does; and
  // if NEITHER direction has enough room (an extremely narrow visible
  // window), still clamp the anchor itself inside bounds so the glyphs at
  // least START on-screen instead of off past the container edge.
  if (fitsRight || !fitsLeft) {
    return { anchor: 'start', x: Math.max(leftBound, Math.min(rightBound, rightStart)) };
  }
  return { anchor: 'end', x: Math.max(leftBound, Math.min(rightBound, leftEnd)) };
}

function xScale(v: number, axis: CostAxis, sizeDomainMax: number): number {
  const floored = Math.max(v, 0);
  const t = axis === 'size' ? Math.log2(1 + floored / 1024) : Math.log10(1 + floored);
  const domainMax = axis === 'size' ? sizeDomainMax : FLOPS_DOMAIN_MAX;
  const frac = Math.max(0, Math.min(1, t / domainMax));
  const adjFrac = ZERO_SHELF + frac * (1 - ZERO_SHELF);
  return PAD_L + adjFrac * (W - PAD_L - PAD_R);
}

function yScale(optimality: number, domainMin: number, domainMax: number): number {
  const span = Math.max(domainMax - domainMin, 1e-6);
  const frac = Math.max(0, Math.min(1, (optimality - domainMin) / span));
  return H - PAD_B - frac * (H - PAD_T - PAD_B);
}

/** True 2D non-dominated staircase for the CURRENT axis. Candidates are
 * restricted to agents the backend already marked cheap-frontier-eligible
 * (`pareto` and not `over_budget` — see app/neurogolf/score.py) so the drawn
 * line never contradicts the backend's 3D dominance truth or includes an
 * unbounded/over-budget agent (e.g. the exact solver) that would otherwise
 * float above every real "cheap" frontier. That candidate set can still
 * contain points that are 2D-dominated once the OTHER cost axis is projected
 * away (two agents can both be 3D-non-dominated while shipping the same
 * artifact size and differing only in FLOPs) so this reduces it to a true
 * monotone staircase for THIS axis: sorted by cost ascending, keeping only
 * points whose optimality strictly improves on everything cheaper. That
 * guarantees the drawn line is a valid lower-right boundary — no plotted
 * point (in the candidate set) ever sits above it. */
function paretoFrontier(agents: LeaderboardAgent[], axis: CostAxis): LeaderboardAgent[] {
  const candidates = agents
    .filter((a) => a.pareto && !a.over_budget)
    .map((a) => ({ a, x: xValue(a, axis), y: a.optimality }))
    .sort((p, q) => p.x - q.x || q.y - p.y);

  const staircase: LeaderboardAgent[] = [];
  let bestY = -Infinity;
  for (const p of candidates) {
    if (p.y > bestY) {
      staircase.push(p.a);
      bestY = p.y;
    }
  }
  return staircase;
}

/** Dodge points that are within DODGE_THRESHOLD_PX of a neighbor in EITHER
 * dimension (Euclidean, not x-only) so nothing fully occludes anything else
 * — several nano-tier agents commonly ship near-identical size AND
 * near-identical optimality, so after zooming the y-axis to the real data
 * range (see yAxisConfig) two points can still land close together in both
 * x and y even though they were never close on x alone. Grouped via
 * union-find so a whole chain of mutually-close points spreads out
 * together, not just adjacent pairs. */
function dodgedPositions(
  agents: LeaderboardAgent[],
  axis: CostAxis,
  sizeDomainMax: number,
  yDomainMin: number,
  yDomainMax: number,
): Map<string, { cx: number; cy: number }> {
  const base = agents.map((a) => ({
    a,
    cx: xScale(xValue(a, axis), axis, sizeDomainMax),
    cy: yScale(a.optimality, yDomainMin, yDomainMax),
  }));

  const n = base.length;
  const parent = Array.from({ length: n }, (_, i) => i);
  function find(i: number): number {
    while (parent[i] !== i) {
      parent[i] = parent[parent[i]];
      i = parent[i];
    }
    return i;
  }
  function union(i: number, j: number) {
    const pi = find(i);
    const pj = find(j);
    if (pi !== pj) parent[pi] = pj;
  }
  for (let i = 0; i < n; i += 1) {
    for (let j = i + 1; j < n; j += 1) {
      const dx = base[i].cx - base[j].cx;
      const dy = base[i].cy - base[j].cy;
      if (Math.sqrt(dx * dx + dy * dy) <= DODGE_THRESHOLD_PX) union(i, j);
    }
  }

  const groups = new Map<number, typeof base>();
  base.forEach((p, i) => {
    const root = find(i);
    const g = groups.get(root);
    if (g) g.push(p);
    else groups.set(root, [p]);
  });

  const positioned = new Map<string, { cx: number; cy: number }>();
  for (const group of groups.values()) {
    group.sort((p, q) => p.cx - q.cx);
    const gn = group.length;
    group.forEach((p, i) => {
      const offset = gn > 1 ? (i - (gn - 1) / 2) * DODGE_STEP_PX : 0;
      positioned.set(p.a.name, { cx: p.cx + offset, cy: p.cy });
    });
  }
  return positioned;
}

export function ParetoPlot({ agents, axis, onAxisChange, flagship }: ParetoPlotProps) {
  const [active, setActive] = useState<LeaderboardAgent | null>(null);

  const sizeAxis = useMemo(() => sizeAxisConfig(agents), [agents]);
  const yAxis = useMemo(() => yAxisConfig(agents), [agents]);
  const frontier = useMemo(() => paretoFrontier(agents, axis), [agents, axis]);
  const positions = useMemo(
    () => dodgedPositions(agents, axis, sizeAxis.domainMax, yAxis.domainMin, yAxis.domainMax),
    [agents, axis, sizeAxis.domainMax, yAxis.domainMin, yAxis.domainMax],
  );
  const hasOverBudget = agents.some((a) => a.over_budget);
  const ticks = axis === 'size' ? sizeAxis.ticks : FLOPS_TICKS;
  const tickLabels = axis === 'size' ? sizeAxis.tickLabels : FLOPS_TICK_LABELS;
  // Whether the y-axis is honestly zoomed away from a 0% baseline -- drives
  // the axis-break glyph and caption below so a reader never mistakes a
  // zoomed band for the metric's true floor.
  const yZoomed = yAxis.domainMin > 0.005;

  // The frontier line connects the UNDODGED (true) coordinates — dodging is a
  // rendering-only nudge for occluded points, it must never bend the honesty
  // of the drawn staircase. It ALWAYS extends flat to the right edge at the
  // champion's (last = max-optimality) strength: no agent can beat the
  // frontier champion no matter how many bytes/FLOPs it spends, so the
  // achievable-strength ceiling is horizontal past the champion. That
  // extension is also what makes the line VISIBLE when the frontier collapses
  // to a single point (on the size axis the 0-byte flagship dominates every
  // costlier agent, so the staircase is one point) -- a lone point drew no
  // line at all, contradicting the "Frontier line = ..." caption and SPEC
  // sec.6.3's "frontier drawn as a line".
  const RIGHT_EDGE = W - PAD_R;
  const frontierPts = frontier.map((a) => ({
    x: xScale(xValue(a, axis), axis, sizeAxis.domainMax),
    y: yScale(a.optimality, yAxis.domainMin, yAxis.domainMax),
  }));
  let linePath = '';
  if (frontierPts.length >= 1) {
    linePath = `M ${frontierPts[0].x} ${frontierPts[0].y}`;
    for (let i = 1; i < frontierPts.length; i += 1) linePath += ` L ${frontierPts[i].x} ${frontierPts[i].y}`;
    const champ = frontierPts[frontierPts.length - 1];
    if (champ.x < RIGHT_EDGE) linePath += ` L ${RIGHT_EDGE} ${champ.y}`;
  }

  const displayed = active ?? (flagship ? agents.find((a) => a.name === flagship) : undefined) ?? agents[0] ?? null;

  return (
    <div className="flex flex-col gap-3">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <h3 className="text-sm font-semibold text-[var(--ink)]">Pareto frontier — strength vs. cost</h3>
        <div role="radiogroup" aria-label="Cost axis" className="flex items-center gap-1 rounded-lg bg-[var(--surface-2)] p-1 text-xs">
          {(['size', 'flops'] as const).map((a) => (
            <button
              key={a}
              type="button"
              role="radio"
              aria-checked={axis === a}
              onClick={() => onAxisChange(a)}
              className={`rounded-md px-2.5 py-1 font-medium cursor-pointer ${
                axis === a ? 'bg-[var(--surface)] text-[var(--ink)] shadow-sm' : 'text-[var(--ink-2)]'
              }`}
            >
              {a === 'size' ? 'Size (bytes)' : 'FLOPs/move'}
            </button>
          ))}
        </div>
      </div>

      {/* Two legends, not one -- the single combined row used to mix
          taxonomy (what KIND of agent a color means), identity (which
          specific agent the flagship marker points at), and status (over
          budget) all in one visually undifferentiated run. Splitting them
          groups like-with-like: the first row is purely "what does this dot
          COLOR mean", the second is "what does this SPECIFIC marker/point
          mean". */}
      <div className="flex flex-wrap items-center gap-3 text-xs text-[var(--ink-2)]">
        {KIND_ORDER.map((k) => (
          <span key={k} className="inline-flex items-center gap-1">
            <span aria-hidden="true" className="inline-block h-2.5 w-2.5 rounded-full" style={{ backgroundColor: CAT_COLOR[k] }} />
            {CAT_LABEL[k]}
          </span>
        ))}
      </div>
      {flagship || hasOverBudget ? (
        <div className="-mt-1.5 flex flex-wrap items-center gap-3 text-xs text-[var(--ink-2)]">
          {flagship ? (
            <span className="inline-flex items-center gap-1 font-medium text-[var(--ink)]">
              <span
                aria-hidden="true"
                className="inline-block h-3 w-3 rounded-full ring-2"
                style={{ backgroundColor: 'var(--accent-solid)', boxShadow: '0 0 0 2px var(--ink)' }}
              />
              {(() => {
                const f = agents.find((a) => a.name === flagship);
                return f ? labelOf(f) : flagship;
              })()}{' '}
              (flagship)
            </span>
          ) : null}
          {hasOverBudget ? (
            <span className="inline-flex items-center gap-1 opacity-70">
              <span aria-hidden="true" className="inline-block h-2.5 w-2.5 rounded-full" style={{ backgroundColor: 'var(--ink-muted)' }} />
              Over budget (off cheap frontier)
            </span>
          ) : null}
        </div>
      ) : null}

      <div
        className="mx-auto w-full"
        style={{ maxWidth: W }}
        role="group"
        aria-label={`Scatter plot of agent strength versus ${axis === 'size' ? 'artifact size' : 'FLOPs per move'}, with the cheap Pareto frontier drawn as a line. Each point is focusable for details.`}
      >
      {/* Responsive: the SVG scales to the container width (capped at its
          native 640px on desktop) via width=100% + viewBox, so at 375px the
          WHOLE plot is visible scaled-down -- no fixed-canvas + horizontal
          scroll that previously left the two most-expensive agents scrolled
          off-screen (and clipped points at the right edge) until a user
          manually scrolled the chart sideways. */}
      <svg
        viewBox={`0 0 ${W} ${H}`}
        width="100%"
        className="block h-auto w-full"
      >
        {/* gridlines -- y ticks are the DATA-DRIVEN zoomed set (yAxis.ticks),
            not a fixed 0/25/50/75/100: the domain itself now tracks the
            agents actually plotted (see yAxisConfig), so the old fixed set
            would draw gridlines outside the domain (or bunched at one end)
            whenever the zoom kicked in. */}
        {yAxis.ticks.map((g) => (
          <line
            key={g}
            x1={PAD_L}
            x2={W - PAD_R}
            y1={yScale(g, yAxis.domainMin, yAxis.domainMax)}
            y2={yScale(g, yAxis.domainMin, yAxis.domainMax)}
            stroke="var(--gridline)"
            strokeWidth={1}
          />
        ))}
        {ticks.map((t, i) => (
          <line
            key={t}
            x1={xScale(t, axis, sizeAxis.domainMax)}
            x2={xScale(t, axis, sizeAxis.domainMax)}
            y1={PAD_T}
            y2={H - PAD_B}
            stroke={i === 0 ? 'var(--baseline)' : 'var(--gridline)'}
            strokeDasharray={i === 0 ? undefined : '3 3'}
            strokeWidth={1}
          />
        ))}

        {/* axes labels */}
        {yAxis.ticks.map((g) => (
          <text
            key={g}
            x={PAD_L - 8}
            y={yScale(g, yAxis.domainMin, yAxis.domainMax) + 3}
            textAnchor="end"
            fontSize={10}
            fill="var(--ink-muted-text)"
            className="tabular"
          >
            {formatYTick(g, yAxis.domainMax - yAxis.domainMin)}
          </text>
        ))}
        {/* Axis-break glyph (a small double-slash "zigzag") on the y-axis
            baseline whenever the domain doesn't start at 0% -- an explicit,
            standard signal that this is a zoomed band, not the metric's true
            floor, so the honest tick labels above are never mistaken for a
            hidden-zero truncated axis. Optimality is a [0,1] ratio where 0%
            does carry real meaning (a rounding-argmax coin-flip), so this
            glyph is drawn precisely because that zero baseline exists and
            isn't shown. */}
        {yZoomed ? (
          // Two short diagonal <line>s (a "//" break mark), not a <path> --
          // deliberately a different element type than the frontier's own
          // <path> so this decorative glyph can never be mistaken for (or
          // miscounted alongside) the one real frontier line by anything
          // querying the chart's `path` elements.
          <g stroke="var(--ink-muted-text)" strokeWidth={1.25}>
            <line x1={PAD_L - 6} y1={H - PAD_B + 2} x2={PAD_L - 2} y2={H - PAD_B - 4} />
            <line x1={PAD_L - 2} y1={H - PAD_B - 2} x2={PAD_L + 2} y2={H - PAD_B - 8} />
          </g>
        ) : null}
        {ticks.map((t, i) => (
          <text key={t} x={xScale(t, axis, sizeAxis.domainMax)} y={H - PAD_B + 16} textAnchor="middle" fontSize={10} fill="var(--ink-muted-text)">
            {tickLabels[i]}
          </text>
        ))}
        <text
          x={-(H / 2)}
          y={14}
          transform="rotate(-90)"
          textAnchor="middle"
          fontSize={11}
          fill="var(--ink-2)"
        >
          Optimality (strength)
        </text>

        {/* frontier line -- drawn BEFORE the points loop below (so it's
            behind every point in z-order) and now thinner + lower-opacity
            than the old 2.5px/0.85 (vision review: "the heavy frontier line
            runs straight through the point cluster and through the 'Zero'
            text label"). It's a reference boundary, not the headline data
            -- the points are. */}
        {frontier.length >= 1 ? <path d={linePath} fill="none" stroke="var(--ink-2)" strokeWidth={1.5} strokeLinejoin="round" strokeLinecap="round" opacity={0.4} /> : null}

        {/* points */}
        {agents.map((a) => {
          const { cx, cy } = positions.get(a.name) ?? {
            cx: xScale(xValue(a, axis), axis, sizeAxis.domainMax),
            cy: yScale(a.optimality, yAxis.domainMin, yAxis.domainMax),
          };
          const isFlagship = flagship != null && a.name === flagship;
          const isActive = displayed?.name === a.name;
          const r = isFlagship ? 8 : 5;
          const flagshipLabel = isFlagship ? flagshipLabelPlacement(cx, r, labelOf(a), W) : null;
          return (
            <g key={a.name}>
              <circle
                cx={cx}
                cy={cy}
                r={r + (isFlagship ? 5 : 2)}
                fill="var(--surface)"
              />
              {isFlagship ? (
                <circle
                  cx={cx}
                  cy={cy}
                  r={r + 4}
                  fill="none"
                  stroke="var(--accent-solid)"
                  strokeWidth={1.5}
                  opacity={0.55}
                />
              ) : null}
              <circle
                cx={cx}
                cy={cy}
                r={r}
                fill={isFlagship ? 'var(--accent-solid)' : a.over_budget ? 'var(--ink-muted)' : CAT_COLOR[a.kind]}
                fillOpacity={!isFlagship && a.over_budget ? 0.6 : 1}
                stroke={isFlagship ? 'var(--ink)' : isActive ? 'var(--accent)' : 'none'}
                strokeWidth={isFlagship ? 2.5 : 2}
                tabIndex={0}
                role="button"
                aria-label={`${labelOf(a)}${isFlagship ? ', flagship' : ''}, ${CAT_LABEL[a.kind]}, optimality ${formatPct(a.optimality)}, ${formatBytes(a.size_bytes)}, ${formatFlops(a.flops_per_move)} FLOPs per move${a.pareto ? ', on the Pareto frontier' : ''}${a.over_budget ? ', over compute budget, excluded from the cheap frontier' : ''}`}
                onMouseEnter={() => setActive(a)}
                onMouseLeave={() => setActive(null)}
                onFocus={() => setActive(a)}
                onBlur={() => setActive(null)}
                style={{ cursor: 'pointer', outlineOffset: 3 }}
              />
              {isFlagship && flagshipLabel ? (
                <text
                  x={flagshipLabel.x}
                  // Above the point/halo (not level with it): the flagship
                  // is frequently ON the frontier line (it's usually the
                  // best cheap agent), and a same-y label used to sit
                  // directly on top of that line, reading as glued to it
                  // even with the halo stroke below. Lifting the label clear
                  // of both the point's halo ring AND the line's y is a real
                  // callout, not just an in-place color trick.
                  y={Math.max(10, cy - r - FLAGSHIP_HALO_GAP)}
                  textAnchor={flagshipLabel.anchor}
                  fontSize={10}
                  fontWeight={700}
                  fill="var(--ink)"
                  // Still keep the halo stroke as a second line of defense
                  // (a gridline or the axis-break glyph could still cross
                  // behind the label at some domains).
                  paintOrder="stroke"
                  stroke="var(--surface)"
                  strokeWidth={3}
                >
                  {labelOf(a)}
                </text>
              ) : a.over_budget ? (
                <text x={cx + r + 5} y={cy + 3} fontSize={9} fill="var(--ink-muted-text)">
                  off cheap frontier
                </text>
              ) : null}
            </g>
          );
        })}
      </svg>
      </div>
      {/* ONE caption line, not three separate paragraphs (vision review: "it
          needs a 3-paragraph caption to be understood"). Rendered as normal
          HTML (not SVG <text>) so it still wraps naturally on narrow
          screens instead of being clipped by the chart viewport; the
          zoomed-axis honesty note only appears when the zoom is actually
          active, keeping the common case (unzoomed) even shorter. */}
      <p className="-mt-1 text-center text-[11px] text-[var(--ink-2)]">
        {axis === 'size' ? 'Size (log scale, 0 = no artifact)' : 'FLOPs/move (log scale, 0 = no arithmetic)'} · Frontier
        line = cheapest non-dominated agents on this axis
        {hasOverBudget ? ' (grey = over budget, excluded)' : ''}
        {yZoomed ? (
          <>
            {' '}
            · y-axis zoomed to {formatYTick(yAxis.domainMin, yAxis.domainMax - yAxis.domainMin)}–
            {formatYTick(yAxis.domainMax, yAxis.domainMax - yAxis.domainMin)} (⌇ = axis break, not 0%)
          </>
        ) : null}
      </p>

      {displayed ? (
        <div className="rounded-lg border border-[var(--border)] bg-[var(--surface)] p-3 text-sm">
          <div className="flex items-center justify-between gap-2">
            <span className="font-semibold text-[var(--ink)]">
              {labelOf(displayed)}
              {flagship != null && displayed.name === flagship ? (
                <span className="ml-1.5 rounded px-1 py-0.5 text-[10px] font-semibold" style={{ backgroundColor: 'var(--accent-solid)', color: 'var(--accent-solid-ink)' }}>
                  flagship
                </span>
              ) : null}
            </span>
            <span
              className="inline-flex items-center gap-1.5 rounded px-1.5 py-0.5 text-[11px] font-medium text-[var(--ink-2)]"
              style={{ backgroundColor: 'var(--surface-2)' }}
            >
              <span aria-hidden="true" className="inline-block h-2 w-2 rounded-full" style={{ backgroundColor: CAT_COLOR[displayed.kind] }} />
              {CAT_LABEL[displayed.kind]}
            </span>
          </div>
          <dl className="mt-2 grid grid-cols-2 gap-x-4 gap-y-1 text-xs text-[var(--ink-2)] sm:grid-cols-4">
            <div>
              <dt>Optimality</dt>
              <dd className="tabular font-medium text-[var(--ink)]">{formatPct(displayed.optimality)}</dd>
            </div>
            <div>
              <dt>Size</dt>
              <dd className="tabular font-medium text-[var(--ink)]">{formatBytes(displayed.size_bytes)}</dd>
            </div>
            <div>
              <dt>FLOPs/move</dt>
              <dd className="tabular font-medium text-[var(--ink)]">{formatFlops(displayed.flops_per_move)}</dd>
            </div>
            <div>
              <dt>NeuroFour score</dt>
              <dd className="tabular font-medium text-[var(--ink)]">{displayed.neurogolf_score.toFixed(3)}</dd>
            </div>
          </dl>
          {displayed.pareto ? (
            <p className="mt-2 text-xs font-medium" style={{ color: 'var(--good-text)' }}>
              On the Pareto frontier
            </p>
          ) : null}
          {displayed.over_budget ? (
            <p className="mt-1 text-xs font-medium" style={{ color: 'var(--critical)' }}>
              Over the compute budget — excluded from cheap tiers
            </p>
          ) : null}
        </div>
      ) : null}
    </div>
  );
}
