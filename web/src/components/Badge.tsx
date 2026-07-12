/** One shared badge/chip system used everywhere an agent's status or
 * taxonomy is shown as a small pill (AgentsScreen cards, LeaderboardTable
 * rows, AgentPicker option rows). Previously each of those three places grew
 * its own ad-hoc chip styling independently -- different padding, different
 * font sizes, different baselines -- so a single card could show a
 * green-outline "pareto" chip, an amber-outline "over budget" chip, a solid
 * gray "kind" chip sitting at a totally different corner/baseline, AND a
 * solid blue "champion" chip, none of which agreed on size or alignment
 * (vision review: "badge soup"). One component, four semantic variants,
 * identical padding/radius/font everywhere:
 *  - `accent`   status: THE thing that's true and important (champion)
 *  - `warning`  status: a caveat worth flagging (over budget)
 *  - `good`     status: a positive but secondary fact (on the Pareto frontier)
 *  - `neutral`  taxonomy: kind / tier -- purely descriptive, never a verdict
 */
export type BadgeVariant = 'accent' | 'warning' | 'good' | 'neutral';

const VARIANT_STYLE: Record<BadgeVariant, React.CSSProperties> = {
  accent: { backgroundColor: 'var(--accent-solid)', color: 'var(--accent-solid-ink)' },
  // NOT `color: var(--warning)` on a transparent background: --warning is
  // tuned for a 3:1 non-text UI object (a border, an icon), and axe's
  // color-contrast check measured it failing AA as small (10px) TEXT --
  // 3.44:1 against the page background, well under the 4.5:1 text floor.
  // A warning-tinted BACKGROUND + border with plain high-contrast --ink
  // text keeps the same "this is a caveat, not a status/error" visual
  // identity while guaranteeing the text itself always clears AA.
  warning: {
    color: 'var(--ink)',
    border: '1px solid var(--warning)',
    backgroundColor: 'color-mix(in srgb, var(--warning) 18%, var(--surface))',
  },
  good: { color: 'var(--good-text)', border: '1px solid var(--good)', backgroundColor: 'transparent' },
  neutral: { backgroundColor: 'var(--surface-2)', color: 'var(--ink-2)' },
};

export function Badge({ variant, children }: { variant: BadgeVariant; children: React.ReactNode }) {
  return (
    <span
      className="inline-flex items-center rounded px-1.5 py-0.5 text-[10px] font-semibold leading-none"
      style={VARIANT_STYLE[variant]}
    >
      {children}
    </span>
  );
}
