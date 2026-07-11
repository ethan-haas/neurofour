import { test as base, expect } from '@playwright/test';
import { installMockRoutes } from '../scripts/install-mock-routes.mjs';
import type { Page } from '@playwright/test';

// Every test gets the documented NeuroFour API (SPEC.md §5) mocked in-memory,
// since the real backend is built in parallel and may not be running.
export const test = base.extend<{ mockedPage: Page }>({
  mockedPage: async ({ page }, use) => {
    await installMockRoutes(page);
    await page.goto('/');
    await use(page);
  },
});

export { expect };

export async function startGame(page: Page, opponent = 'neurofour-net') {
  await page.getByLabel('Yellow (moves second)').selectOption(opponent);
  await page.getByRole('button', { name: 'New game' }).click();
  await expect(page.getByText('Red to move (You)')).toBeVisible();
}

export async function clickColumn(page: Page, col: number) {
  await page.getByRole('button', { name: new RegExp(`^Drop disc in column ${col}\\b`) }).click();
  await page.waitForTimeout(900);
}
