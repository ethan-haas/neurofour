import { test, expect, type Page } from '@playwright/test';

// Regression test for the "Analyze toggle is a dead control" escape.
//
// Unlike every other e2e spec in this suite, this one does NOT use the
// mocked-route fixture -- a mock that resolves instantly can never catch a
// bug that only shows up against the REAL backend's real latency (the
// depth-limited near-empty-position analysis in app/main.py::_bounded_analyze
// is a genuine multi-second pure-Python search, not a mock artifact). This
// test requires a live NeuroFour backend reachable at ANALYZE_LIVE_API (or
// AUDIT_API) -- boot it with
//   python -m uvicorn app.main:app --port 8012
// and build the frontend against it with VITE_API_BASE matching, e.g.
//   VITE_API_BASE=http://localhost:8012 npm run build
// before running `npm run e2e`.
const API = process.env.ANALYZE_LIVE_API || process.env.AUDIT_API || 'http://localhost:8012';

// Generous budget: the "perfect" opponent's own reply (a bounded search) and
// the /analyze call are both genuine multi-second pure-Python searches, and
// this suite may run on a loaded shared machine alongside other tasks.
test.setTimeout(120_000);

async function setupMidgame(page: Page) {
  await page.goto('/');
  await page.locator('select').nth(0).selectOption({ label: 'You (human)' });
  await page.locator('select').nth(1).selectOption({ label: 'perfect' });
  await page.getByRole('button', { name: 'New game' }).click();
  await expect(page.getByText('Red to move (You)')).toBeVisible({ timeout: 15_000 });
  await page.getByRole('button', { name: 'Drop disc in column 4' }).click();

  // Wait for perfect's reply (2 discs total on the board).
  await page.waitForFunction(
    () => {
      const grid = Array.from(document.querySelectorAll('div[aria-hidden="true"]')).find(
        (g) => g.querySelectorAll(':scope>div').length === 42,
      );
      if (!grid) return false;
      return Array.from(grid.querySelectorAll(':scope>div')).filter((c) => c.querySelector('*')).length === 2;
    },
    { timeout: 30_000 },
  );
}

// NeuroFour's Board component puts the `title="solver score {v}"` attribute
// on a <span> nested inside each column button (not on the button element
// itself) -- read it off whichever descendant carries it.
function readColumnTitles() {
  return Array.from(document.querySelectorAll('[role="toolbar"] button')).map(
    (b) => b.querySelector('[title]')?.getAttribute('title') ?? null,
  );
}

test('analyze: overlay renders the solver per-column evaluation & best move, matching a direct /analyze call', async ({
  page,
}) => {
  await setupMidgame(page);

  // Ground truth straight from the real backend for this exact position
  // (moves 3,2) -- the same request shape the app itself sends.
  const apiRes = await page.request.post(`${API}/analyze`, {
    data: { board: [3, 2], mode: 'scored' },
  });
  expect(apiRes.ok(), `real backend /analyze must succeed, got ${apiRes.status()}`).toBe(true);
  const api = await apiRes.json();
  const perCol: Record<string, number> = api.per_col;
  const bestCol: number = api.best_col;

  // Turn Analyze ON via the visible control.
  const cb = page.getByRole('checkbox', { name: 'Analyze' });
  await cb.check();

  // Backend latency for a near-empty (off-book) position is genuinely
  // several seconds -- the UI must show SOME immediate feedback (not look
  // like a dead control) the instant the request is in flight. There are now
  // TWO such indicators on screen at once (the Status card's "Analyzing
  // position…" line, and the board's own skeleton-badge-row legend spinner
  // text "Analyzing…") -- deliberately redundant, so `.first()` here just
  // asserts at least one is visible, not which.
  await expect(page.getByText(/Analyzing/i).first()).toBeVisible({ timeout: 5000 });

  // The overlay must eventually render the REAL per-column values returned
  // by the backend -- read them back out of the DOM (the `title` attribute
  // NeuroFour's Board component sets on every analyzed column) and assert
  // DOM === backend for every column, not just "some element appeared".
  await expect
    .poll(
      async () => {
        const titles = await page.evaluate(readColumnTitles);
        return titles.every((t) => t && /^solver score -?\d+$/.test(t));
      },
      { timeout: 45_000, intervals: [250, 500, 1000] },
    )
    .toBe(true);

  const domScores = await page.evaluate(() => {
    const titles = Array.from(document.querySelectorAll('[role="toolbar"] button')).map(
      (b) => b.querySelector('[title]')?.getAttribute('title') ?? null,
    );
    return titles.map((title) => {
      const m = (title || '').match(/^solver score (-?\d+)$/);
      return m ? Number(m[1]) : null;
    });
  });

  for (let col = 0; col < 7; col += 1) {
    expect(domScores[col], `column ${col} DOM score must match backend per_col[${col}]=${perCol[String(col)]}`).toBe(
      perCol[String(col)],
    );
  }

  // The pending indicator must be gone once the overlay has landed.
  await expect(page.getByText(/Analyzing/i)).toHaveCount(0);

  // Best-move highlight must equal the API's best_col: the star ("★") badge
  // lives on that column's button and nowhere else.
  const starredCols = await page.evaluate(() =>
    Array.from(document.querySelectorAll('[role="toolbar"] button')).map((b) => (b.textContent || '').includes('★')),
  );
  expect(starredCols.filter(Boolean).length, 'exactly one column should carry the best-move star').toBe(1);
  expect(starredCols[bestCol], `the starred column must be the API best_col (${bestCol})`).toBe(true);

  // Turning Analyze OFF must remove the overlay entirely.
  await cb.uncheck();
  await expect
    .poll(async () => {
      const titles = await page.evaluate(readColumnTitles);
      return titles.every((t) => !t);
    })
    .toBe(true);
});

test('analyze: rapid re-toggling does not leak stale requests or show a stale position', async ({ page }) => {
  await setupMidgame(page);

  const cb = page.getByRole('checkbox', { name: 'Analyze' });

  // Flip it on/off/on/off/on quickly while the (slow) first request is still
  // in flight -- the final state (checked) must still end up showing a
  // fresh, correct overlay, and no earlier in-flight response for a stale
  // toggle state may land afterwards and get rendered.
  await cb.check();
  await page.waitForTimeout(150);
  await cb.uncheck();
  await page.waitForTimeout(150);
  await cb.check();
  await page.waitForTimeout(150);
  await cb.uncheck();
  await page.waitForTimeout(150);
  await cb.check();

  await expect(cb).toBeChecked();

  await expect
    .poll(
      async () => {
        const titles = await page.evaluate(readColumnTitles);
        return titles.every((t) => t && /^solver score -?\d+$/.test(t));
      },
      { timeout: 45_000, intervals: [250, 500, 1000] },
    )
    .toBe(true);

  // Give any stale in-flight responses a further window to (wrongly) land,
  // then confirm the overlay is still consistent (checked + fully rendered,
  // not cleared or half-updated).
  await page.waitForTimeout(2000);
  await expect(cb).toBeChecked();
  const stillRendered = await page.evaluate(() => {
    const titles = Array.from(document.querySelectorAll('[role="toolbar"] button')).map(
      (b) => b.querySelector('[title]')?.getAttribute('title') ?? null,
    );
    return titles.every((t) => t && /^solver score -?\d+$/.test(t));
  });
  expect(stillRendered).toBe(true);
});
