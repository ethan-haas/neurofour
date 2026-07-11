import { useCallback, useEffect, useState } from 'react';
import { api } from '../lib/api';
import { errorMessage, type AsyncState } from '../lib/asyncState';
import type { AgentManifest } from '../types';

export function useAgents(): AsyncState<AgentManifest[]> & { reload: () => void } {
  const [state, setState] = useState<AsyncState<AgentManifest[]>>({ status: 'loading' });

  const reload = useCallback(() => {
    setState({ status: 'loading' });
    api.agents().then(
      (data) => setState({ status: 'success', data }),
      (err: unknown) => setState({ status: 'error', message: errorMessage(err, 'Failed to load agents.') }),
    );
  }, []);

  useEffect(() => {
    reload();
  }, [reload]);

  return { ...state, reload };
}
