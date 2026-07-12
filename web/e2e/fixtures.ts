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

/** NewGamePanel's two opponent pickers are now an accessible custom listbox
 * (AgentPicker.tsx), not a native `<select>` -- `selectOption()` no longer
 * applies. Opens the picker identified by its visible label (e.g. "Red
 * (moves first)" / "Yellow (moves second)") and clicks the option whose
 * visible text matches `optionName` (a display name like "Zero"/"Oracle", or
 * "You (human)"). */
export async function selectAgent(page: Page, pickerLabel: string, optionName: string) {
  await page.getByRole('button', { name: pickerLabel, exact: false }).click();
  await page.getByRole('option', { name: optionName, exact: false }).first().click();
}

// Default opponent is now the 0-byte champion ("Zero" / neurofour-net14, see
// NewGamePanel.tsx) rather than the original "neurofour-net" policy network.
export async function startGame(page: Page, opponent = 'Zero') {
  await selectAgent(page, 'Yellow (moves second)', opponent);
  await page.getByRole('button', { name: 'New game' }).click();
  await expect(page.getByText('Red to move (You)')).toBeVisible();
}

export async function clickColumn(page: Page, col: number) {
  await page.getByRole('button', { name: new RegExp(`^Drop disc in column ${col}\\b`) }).click();
  await page.waitForTimeout(900);
}
