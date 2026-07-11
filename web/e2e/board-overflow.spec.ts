import { test, expect, startGame, clickColumn } from './fixtures';

// Regression test for: the accessible board description was a `<table
// class="sr-only">`. A <table> lays out to its content width regardless of
// the sr-only rule's `width:1px` (auto table layout sizes from cell
// content), so with a board present the document genuinely scrolled
// horizontally into blank space — a real, user-visible bug (not just an
// audit nit), reproduced with `document.scrollingElement.scrollLeft = 400`
// actually landing on 400 instead of clamping back to 0.
test.describe('Board accessible description does not cause horizontal page scroll', () => {
  for (const viewport of [
    { width: 375, height: 800 },
    { width: 1440, height: 900 },
  ]) {
    test(`documentElement.scrollWidth <= clientWidth at ${viewport.width}px with a board present`, async ({
      mockedPage: page,
    }) => {
      await page.setViewportSize(viewport);
      await startGame(page);
      await clickColumn(page, 1);

      const dims = await page.evaluate(() => ({
        scrollWidth: document.documentElement.scrollWidth,
        clientWidth: document.documentElement.clientWidth,
      }));
      expect(
        dims.scrollWidth,
        `documentElement.scrollWidth (${dims.scrollWidth}) must not exceed clientWidth (${dims.clientWidth}) at ${viewport.width}px`,
      ).toBeLessThanOrEqual(dims.clientWidth);

      // The page must not actually be scrollable horizontally either.
      const scrollLeftAfter = await page.evaluate(() => {
        document.scrollingElement!.scrollLeft = 400;
        return document.scrollingElement!.scrollLeft;
      });
      expect(scrollLeftAfter, 'the page should not scroll horizontally into blank space').toBe(0);
    });
  }

  test('the accessible board description is still present and announces row/column detail', async ({
    mockedPage: page,
  }) => {
    await startGame(page);
    await clickColumn(page, 1);

    // Screen-reader semantics must be preserved (not simply deleted) -- the
    // fix must keep an accessible table-like structure (native <table> or an
    // ARIA role="table"/"row"/"cell" equivalent) describing every cell.
    const table = page.locator('[role="table"], table').filter({ hasText: /Row 1, column 1/ });
    await expect(table).toHaveCount(1);
    await expect(page.getByText(/Row \d, column \d: (empty|Red disc|Yellow disc)/).first()).toBeAttached();
  });
});
