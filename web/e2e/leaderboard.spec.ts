import AxeBuilder from '@axe-core/playwright';
import { test, expect } from './fixtures';
import { MOCK_LEADERBOARD } from '../scripts/mock-data.mjs';

test.describe('Leaderboard', () => {
  test('renders the scatter plot, frontier line, and highlights the flagship', async ({ mockedPage: page }) => {
    await page.getByRole('button', { name: 'Leaderboard' }).click();

    const svg = page.getByRole('group', { name: /Scatter plot of agent strength/ });
    await expect(svg).toBeVisible();

    // Exactly one focusable point circle per mock agent (each point also has a
    // non-interactive background "halo" circle, and the flagship gets an extra
    // non-interactive accent ring — neither carries role="button", so this
    // selector counts only the real per-agent points). Derived from the fixture
    // roster length, not a hardcoded literal, so it tracks MOCK_LEADERBOARD_AGENTS
    // (web/scripts/mock-data.mjs) automatically as the registry grows and can't
    // silently rot out of sync again.
    const points = svg.locator('circle[role="button"]');
    await expect(points).toHaveCount(MOCK_LEADERBOARD.agents.length);

    // The non-dominated frontier line.
    await expect(svg.locator('path')).toHaveCount(1);

    // Flagship is labeled directly on the chart. The mock leaderboard's
    // headline.agent is 'minimax-4' — the highlight must track that field
    // (never a hardcoded agent name), so minimax-4 is what gets ringed here.
    await expect(svg.getByText('minimax-4')).toBeVisible();
    await expect(page.getByText('minimax-4 (flagship)')).toBeVisible();

    // The table lists every agent and flags the flagship + pareto agents.
    // ('flagship' also appears in the chart's own info-card badge for the
    // same reason — both must track headline.agent — so scope this to the
    // table specifically.)
    await expect(page.getByRole('row', { name: /minimax-4/ })).toBeVisible();
    await expect(page.getByRole('table').getByText('flagship', { exact: true })).toBeVisible();
  });

  test('switches the cost axis between size and FLOPs', async ({ mockedPage: page }) => {
    await page.getByRole('button', { name: 'Leaderboard' }).click();
    const flopsBtn = page.getByRole('radio', { name: 'FLOPs/move' });
    await flopsBtn.click();
    await expect(flopsBtn).toHaveAttribute('aria-checked', 'true');
  });

  test('has no critical or serious axe violations', async ({ mockedPage: page }) => {
    await page.getByRole('button', { name: 'Leaderboard' }).click();
    await page.waitForTimeout(300);
    const results = await new AxeBuilder({ page }).analyze();
    const severe = results.violations.filter((v) => v.impact === 'critical' || v.impact === 'serious');
    expect(severe, JSON.stringify(severe, null, 2)).toEqual([]);
  });
});
