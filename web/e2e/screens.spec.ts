import { test, expect } from './fixtures';

test.describe('Nav + new screens', () => {
  test('Play opponent picker defaults to the 0-byte champion and shows stats in the option rows', async ({
    mockedPage: page,
  }) => {
    // NewGamePanel's default opponent is neurofour-net14 ("Zero"); its
    // subtitle renders next to the trigger's summary without opening the
    // popup.
    await expect(page.getByText('Zero', { exact: true })).toBeVisible();
    await expect(page.getByText('0-byte champion — pure bitboard search')).toBeVisible();

    // Open the picker: stats must be visible in the option rows themselves
    // (not just the collapsed trigger / selected stat card).
    await page.getByRole('button', { name: 'Yellow (moves second)', exact: false }).click();
    const listbox = page.getByRole('listbox').first();
    await expect(listbox).toBeVisible();
    const oracleOption = page.getByRole('option', { name: 'Oracle', exact: false });
    await expect(oracleOption).toBeVisible();
    await expect(oracleOption.getByText(/opt/)).toBeVisible();
  });

  test('opponent picker is keyboard operable (Arrow/Home/End/Enter/Escape)', async ({ mockedPage: page }) => {
    const trigger = page.getByRole('button', { name: 'Yellow (moves second)', exact: false });
    await trigger.focus();
    await page.keyboard.press('ArrowDown');
    await expect(page.getByRole('listbox').first()).toBeVisible();
    await page.keyboard.press('Escape');
    await expect(page.getByRole('listbox')).toHaveCount(0);
    // Focus must return to the trigger after Escape.
    await expect(trigger).toBeFocused();
  });

  test('Nav shows all four screens and About explains the score without regressing across navigation', async ({
    mockedPage: page,
  }) => {
    await expect(page.getByRole('button', { name: 'Play', exact: true })).toBeVisible();
    await expect(page.getByRole('button', { name: 'Agents', exact: true })).toBeVisible();
    await expect(page.getByRole('button', { name: 'Leaderboard', exact: true })).toBeVisible();
    await expect(page.getByRole('button', { name: 'About', exact: true })).toBeVisible();

    await page.getByRole('button', { name: 'About', exact: true }).click();
    await expect(page.getByText('NeuroFour Score', { exact: false }).first()).toBeVisible();
    await expect(page.getByText(/5,000,000/)).toBeVisible();
    // The score's other name must never be displayed anywhere in the app.
    await expect(page.getByText('NeuroGolf', { exact: false })).toHaveCount(0);

    await page.getByRole('button', { name: 'Agents', exact: true }).click();
    await expect(page.getByRole('heading', { name: 'Agents' })).toBeVisible();
  });

  test('Agents card "Play against" preselects that agent on the Play screen', async ({ mockedPage: page }) => {
    await page.getByRole('button', { name: 'Agents', exact: true }).click();
    await page.getByRole('button', { name: 'Play against Oracle', exact: false }).click();

    await expect(page.getByRole('button', { name: 'Play', exact: true })).toHaveAttribute('aria-current', 'page');
    await expect(page.getByText('Oracle', { exact: true }).first()).toBeVisible();
  });

  test('no console errors switching through all four screens', async ({ mockedPage: page }) => {
    const errors: string[] = [];
    page.on('pageerror', (err) => errors.push(String(err)));
    page.on('console', (msg) => {
      if (msg.type() === 'error') errors.push(msg.text());
    });

    for (const label of ['Agents', 'Leaderboard', 'About', 'Play']) {
      await page.getByRole('button', { name: label, exact: true }).click();
      await page.waitForTimeout(100);
    }
    expect(errors, JSON.stringify(errors, null, 2)).toEqual([]);
  });
});
