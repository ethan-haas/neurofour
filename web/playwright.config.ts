import { defineConfig, devices } from '@playwright/test';

// Overridable so local runs don't collide with another preview server
// already bound to the default port (e.g. a concurrent task in the same
// sandbox); defaults are unchanged from before.
const PREVIEW_PORT = process.env.PLAYWRIGHT_PREVIEW_PORT ?? '4173';
const BASE_URL = `http://localhost:${PREVIEW_PORT}`;

export default defineConfig({
  testDir: './e2e',
  fullyParallel: true,
  forbidOnly: !!process.env.CI,
  retries: process.env.CI ? 1 : 0,
  reporter: [['list']],
  use: {
    baseURL: BASE_URL,
    trace: 'retain-on-failure',
  },
  webServer: {
    command: `npm run preview -- --port ${PREVIEW_PORT} --strictPort`,
    url: BASE_URL,
    reuseExistingServer: !process.env.CI,
    timeout: 30_000,
  },
  projects: [
    {
      name: 'chromium',
      use: { ...devices['Desktop Chrome'] },
    },
  ],
});
