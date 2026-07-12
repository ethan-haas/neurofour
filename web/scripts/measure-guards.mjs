// Ad-hoc guard measurement script (not part of the committed e2e suite) --
// measures the specific numeric guards called out in the task: page
// scrollWidth/scrollLeft at 375/1440, and badge-N vs board-column-N pixel
// alignment (max |Δx| via getBoundingClientRect), against the LIVE preview
// server + live backend. Run after `npm run build` and with a preview
// server already up on PREVIEW_PORT.
import { chromium } from '@playwright/test';

const PREVIEW_URL = process.env.PREVIEW_URL ?? 'http://localhost:4173';

// NewGamePanel's opponent pickers are an accessible custom listbox
// (AgentPicker.tsx), not a native <select> -- open by visible label, click
// the option by its visible text.
async function selectAgent(page, pickerLabel, optionName) {
  await page.getByRole('button', { name: pickerLabel, exact: false }).click();
  await page.getByRole('option', { name: optionName, exact: false }).first().click();
}

async function measure(viewport) {
  const browser = await chromium.launch();
  const context = await browser.newContext({ reducedMotion: 'reduce' });
  const page = await context.newPage();
  await page.setViewportSize(viewport);
  await page.goto(PREVIEW_URL, { waitUntil: 'networkidle' });

  // Start a human vs human game, drop a few stones, turn Analyze on so both
  // the toolbar badges AND the board disc-cell grid are rendered together.
  await selectAgent(page, 'Yellow (moves second)', 'You (human)');
  await page.getByRole('button', { name: 'New game' }).click();
  await page.waitForTimeout(400);
  for (const col of [4, 6, 4, 6, 1, 1, 1, 3, 6, 3, 5, 2, 6, 2]) {
    await page.getByRole('button', { name: new RegExp(`^Drop disc in column ${col}\\b`) }).click();
    await page.waitForTimeout(150);
  }
  await page.getByLabel('Analyze').check();
  await page.getByText(/^(★ )?(Win|Draw|Loss)$/).first().waitFor({ timeout: 45000 });

  const dims = await page.evaluate(() => ({
    scrollWidth: document.documentElement.scrollWidth,
    clientWidth: document.documentElement.clientWidth,
  }));
  const scrollLeftAfter = await page.evaluate(() => {
    document.scrollingElement.scrollLeft = 400;
    return document.scrollingElement.scrollLeft;
  });

  // Badge N = the toolbar button "col N"; board column N = the disc cell
  // directly below it. Compare horizontal centers.
  const deltas = [];
  for (let col = 1; col <= 7; col += 1) {
    const badge = page.getByRole('button', { name: new RegExp(`^Drop disc in column ${col}\\b`) });
    const badgeBox = await badge.boundingBox();
    deltas.push(badgeBox.x + badgeBox.width / 2);
  }
  // Board disc cells: the aria-hidden grid directly under the toolbar. Grab
  // the first row's cells (7 of them) via the aria-hidden container.
  const cellCenters = await page.evaluate(() => {
    const gridEls = Array.from(document.querySelectorAll('[aria-hidden="true"]')).filter(
      (el) => el.children.length === 42 || (el.style && el.style.backgroundColor && el.children.length >= 7),
    );
    const grid = gridEls.find((el) => el.children.length === 42);
    if (!grid) return null;
    const firstRowCells = Array.from(grid.children).slice(0, 7);
    return firstRowCells.map((c) => {
      const r = c.getBoundingClientRect();
      return r.x + r.width / 2;
    });
  });

  let maxDelta = 0;
  if (cellCenters) {
    for (let i = 0; i < 7; i += 1) {
      const d = Math.abs(deltas[i] - cellCenters[i]);
      if (d > maxDelta) maxDelta = d;
    }
  }

  await browser.close();
  return { viewport, scrollWidth: dims.scrollWidth, clientWidth: dims.clientWidth, scrollLeftAfter, maxDeltaX: maxDelta };
}

const results = [];
results.push(await measure({ width: 375, height: 812 }));
results.push(await measure({ width: 1440, height: 900 }));
console.log(JSON.stringify(results, null, 2));
