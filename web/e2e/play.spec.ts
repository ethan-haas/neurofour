import AxeBuilder from '@axe-core/playwright';
import { test, expect, startGame, clickColumn } from './fixtures';

test.describe('Play', () => {
  test('plays a full game vs an agent to a win', async ({ mockedPage: page }) => {
    await startGame(page);

    // Red (human) drops in column 1 four times; the mock agent always prefers
    // the center column and never touches column 1, so this produces a clean
    // vertical four-in-a-row for Red.
    await clickColumn(page, 1);
    await clickColumn(page, 1);
    await clickColumn(page, 1);
    await clickColumn(page, 1);

    await expect(page.getByText('Red wins!', { exact: true })).toBeVisible();
    await expect(page.getByText('Red won', { exact: true })).toBeVisible();

    // The column controls should now be disabled (game over).
    await expect(page.getByRole('button', { name: /Drop disc in column 1/ })).toBeDisabled();
  });

  test('toggles Analyze and shows a per-column solver overlay', async ({ mockedPage: page }) => {
    await startGame(page);
    await clickColumn(page, 1);

    await expect(page.getByLabel('Analyze')).not.toBeChecked();
    await page.getByLabel('Analyze').check();

    // At least one column should carry a Win/Draw/Loss analysis badge.
    await expect(page.getByText(/^(★ )?Win$/).first()).toBeVisible({ timeout: 5000 });
    await expect(page.getByText('Winning move')).toBeVisible();
    await expect(page.getByText("Solver's best move")).toBeVisible();

    await page.getByLabel('Analyze').uncheck();
    await expect(page.getByText('Winning move')).toHaveCount(0);
  });

  test('announces moves and results via aria-live for screen readers', async ({ mockedPage: page }) => {
    await startGame(page);
    const live = page.locator('[aria-live="polite"]');
    await expect(live).toContainText(/New game started/);

    // Assert immediately (before the mock agent's delayed auto-reply overwrites
    // this transient message) that the human move itself was announced.
    await page.getByRole('button', { name: /^Drop disc in column 1\b/ }).click();
    await expect(live).toContainText(/Red dropped in column 1/, { timeout: 400 });

    // ...and eventually the agent's reply is announced too.
    await expect(live).toContainText(/Yellow dropped in column/);
  });

  test('is keyboard-playable: arrow keys move the column cursor, Enter drops', async ({ mockedPage: page }) => {
    await startGame(page);

    await page.getByRole('button', { name: /Drop disc in column 1/ }).focus();
    await page.keyboard.press('ArrowRight');
    await page.keyboard.press('ArrowRight');
    await expect(page.getByRole('button', { name: /Drop disc in column 3/ })).toBeFocused();

    await page.keyboard.press('Enter');

    const live = page.locator('[aria-live="polite"]');
    await expect(live).toContainText(/Red dropped in column 3/, { timeout: 400 });
  });

  test('has no critical or serious axe violations', async ({ mockedPage: page }) => {
    await startGame(page);
    await clickColumn(page, 1);
    await page.getByLabel('Analyze').check();
    await page.waitForTimeout(500);

    const results = await new AxeBuilder({ page }).analyze();
    const severe = results.violations.filter((v) => v.impact === 'critical' || v.impact === 'serious');
    expect(severe, JSON.stringify(severe, null, 2)).toEqual([]);
  });
});
