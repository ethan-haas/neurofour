import { MOCK_AGENTS, MOCK_LEADERBOARD, MockGameStore, mockAnalyze } from './mock-data.mjs';

// Matched by PATHNAME only (see the route predicate below), never by origin.
// The frontend's real API_BASE (src/lib/api.ts) is a build-time-baked env var
// (VITE_API_BASE); if dist/ was last built while pointing at some other
// live backend (e.g. a concurrent dev server on a different port), a
// route pattern anchored to a specific `http://localhost:8000/**` origin
// silently fails to match and every "mocked" test quietly falls through to
// that real, non-deterministic backend instead -- a real incident: this is
// exactly what caused leaderboard.spec.ts's circle-count assertion and two
// play.spec.ts assertions to intermittently fail against a stale dist/
// (built with a different VITE_API_BASE) while the mock fixtures/components
// were both already correct. Matching on pathname alone, independent of
// whatever origin+port the currently-served build happens to embed, makes
// the mock robust to that whole class of drift so it cannot rot again.
const API_PATH_RE = /^\/(health|agents|leaderboard|analyze|evaluate|game(\/.*)?)$/;

/**
 * Installs page.route() handlers that emulate the REAL running NeuroFour
 * backend (app/main.py — verified directly against it, not the originally
 * documented spec which the backend diverges from) against a fresh in-memory
 * game store. Used by both scripts/shoot.mjs and the e2e suite so
 * screenshots and tests exercise the same fixture behavior when the real
 * backend isn't reachable.
 */
export async function installMockRoutes(page, { apiBase = 'http://localhost:8000' } = {}) {
  const store = new MockGameStore();
  void apiBase; // kept for API compatibility; matching is now path-based (see API_PATH_RE)

  await page.route((url) => API_PATH_RE.test(url.pathname), async (route) => {
    const req = route.request();
    const url = new URL(req.url());
    const path = url.pathname;
    const method = req.method();

    const json = (body, status = 200) =>
      route.fulfill({ status, contentType: 'application/json', body: JSON.stringify(body) });

    try {
      if (path === '/health' && method === 'GET') {
        return json({ status: 'ok' });
      }
      if (path === '/agents' && method === 'GET') {
        return json({ agents: MOCK_AGENTS });
      }
      if (path === '/leaderboard' && method === 'GET') {
        return json(MOCK_LEADERBOARD);
      }
      if (path === '/game/new' && method === 'POST') {
        const body = req.postDataJSON() ?? {};
        return json(store.create(body.first_agent ?? null, body.second_agent ?? null));
      }
      if (path === '/analyze' && method === 'POST') {
        const body = req.postDataJSON() ?? {};
        return json(mockAnalyze(body.board, body.mode));
      }
      const gameMatch = path.match(/^\/game\/([^/]+)(?:\/(move|agent-move))?$/);
      if (gameMatch) {
        const [, id, action] = gameMatch;
        if (!action && method === 'GET') {
          return json(store.get(id));
        }
        if (action === 'move' && method === 'POST') {
          const body = req.postDataJSON() ?? {};
          return json(store.applyMove(id, body.col));
        }
        if (action === 'agent-move' && method === 'POST') {
          return json(store.agentMove(id));
        }
      }
      return json({ detail: `mock: no handler for ${method} ${path}` }, 404);
    } catch (err) {
      return json({ detail: String(err?.message ?? err) }, 400);
    }
  });

  return store;
}
