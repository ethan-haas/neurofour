import AxeBuilder from '@axe-core/playwright';
import { test, expect } from './fixtures';
import type { Page } from '@playwright/test';

// Regression test for: the leaderboard table's horizontal-scroll wrapper
// (`.overflow-x-auto.rounded-xl.border` in LeaderboardTable.tsx) scrolls at
// 375px (scrollWidth 721 > clientWidth 341) but had no way for a
// keyboard-only user to actually scroll it (axe `scrollable-region-focusable`,
// SERIOUS). Runs the full WCAG 2.1 A/AA tag set across BOTH routes, BOTH
// viewports the spec requires (375/1440), and BOTH themes -- the widest net
// that still exercises the real failure mode (this suite reproduced it
// pre-fix at 375px/Leaderboard, light theme; every other combination is
// included so a future regression anywhere in that matrix is caught too).
//
// Each case also asserts the rest of SPEC.md's quality bar that this same
// route x viewport x theme visit can observe for free: zero console errors,
// zero uncaught page errors, zero 4xx/5xx XHRs.
const VIEWPORTS = [
  { width: 375, height: 900 },
  { width: 1440, height: 900 },
];
const THEMES = ['light', 'dark'] as const;
const TAGS = ['wcag2a', 'wcag2aa', 'wcag21a', 'wcag21aa'];

async function setTheme(page: Page, theme: 'light' | 'dark') {
  const isDark = await page.evaluate(() => document.documentElement.classList.contains('dark'));
  const current = isDark ? 'dark' : 'light';
  if (current !== theme) {
    await page.getByRole('button', { name: /Switch to (dark|light) mode/ }).click();
  }
}

/** Attach console/pageerror/response listeners BEFORE the reload below so a
 * fresh navigation's errors (not just post-load steady state) are captured,
 * then return the accumulated buffers for the caller to assert on at the
 * end of the test. */
function watchForErrors(page: Page) {
  const consoleErrors: string[] = [];
  const pageErrors: string[] = [];
  const badResponses: string[] = [];
  page.on('console', (msg) => {
    if (msg.type() === 'error') consoleErrors.push(msg.text());
  });
  page.on('pageerror', (err) => pageErrors.push(String(err)));
  page.on('response', (res) => {
    if (res.status() >= 400) badResponses.push(`${res.status()} ${res.url()}`);
  });
  return { consoleErrors, pageErrors, badResponses };
}

function assertClean(buffers: ReturnType<typeof watchForErrors>) {
  expect(buffers.consoleErrors, `console errors: ${JSON.stringify(buffers.consoleErrors, null, 2)}`).toEqual([]);
  expect(buffers.pageErrors, `uncaught page errors: ${JSON.stringify(buffers.pageErrors, null, 2)}`).toEqual([]);
  expect(buffers.badResponses, `4xx/5xx responses: ${JSON.stringify(buffers.badResponses, null, 2)}`).toEqual([]);
}

async function scanNoSeriousOrCritical(page: Page) {
  const results = await new AxeBuilder({ page }).withTags(TAGS).analyze();
  const severe = results.violations.filter((v) => v.impact === 'critical' || v.impact === 'serious');
  expect(severe, JSON.stringify(severe, null, 2)).toEqual([]);
}

test.describe('Full WCAG 2.1 A/AA axe scan — Play + Leaderboard x viewport x theme', () => {
  for (const viewport of VIEWPORTS) {
    for (const theme of THEMES) {
      test(`Play @ ${viewport.width}px, ${theme}`, async ({ mockedPage: page }) => {
        await page.setViewportSize(viewport);
        const buffers = watchForErrors(page);
        await page.reload();
        await setTheme(page, theme);
        await page.waitForTimeout(150);
        await scanNoSeriousOrCritical(page);
        assertClean(buffers);
      });

      test(`Leaderboard @ ${viewport.width}px, ${theme}`, async ({ mockedPage: page }) => {
        await page.setViewportSize(viewport);
        const buffers = watchForErrors(page);
        await page.reload();
        await setTheme(page, theme);
        await page.getByRole('button', { name: 'Leaderboard' }).click();
        await page.waitForTimeout(150);
        await scanNoSeriousOrCritical(page);
        assertClean(buffers);
      });
    }
  }
});
