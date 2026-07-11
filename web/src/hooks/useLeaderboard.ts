import { useCallback, useEffect, useState } from 'react';
import { api } from '../lib/api';
import { errorMessage, type AsyncState } from '../lib/asyncState';
import type { LeaderboardResponse } from '../types';

export function useLeaderboard(): AsyncState<LeaderboardResponse> & { reload: () => void } {
  const [state, setState] = useState<AsyncState<LeaderboardResponse>>({ status: 'loading' });

  const reload = useCallback(() => {
    setState({ status: 'loading' });
    api.leaderboard().then(
      (data) => setState({ status: 'success', data }),
      (err: unknown) =>
        setState({ status: 'error', message: errorMessage(err, 'Failed to load the leaderboard.') }),
    );
  }, []);

  useEffect(() => {
    reload();
  }, [reload]);

  return { ...state, reload };
}
