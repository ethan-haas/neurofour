import { useCallback, useRef, useState } from 'react';
import { api } from '../lib/api';
import { errorMessage } from '../lib/asyncState';
import type { GameState } from '../types';

export interface UseGameResult {
  game: GameState | null;
  loading: boolean;
  busy: boolean;
  error: string | null;
  newGame: (firstAgent: string | null, secondAgent: string | null) => Promise<void>;
  playMove: (col: number) => Promise<void>;
  requestAgentMove: () => Promise<GameState | null>;
  clearError: () => void;
}

export function useGame(): UseGameResult {
  const [game, setGame] = useState<GameState | null>(null);
  const [loading, setLoading] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const gameIdRef = useRef<string | null>(null);

  const newGame = useCallback(async (firstAgent: string | null, secondAgent: string | null) => {
    setLoading(true);
    setError(null);
    try {
      const state = await api.newGame({ first_agent: firstAgent, second_agent: secondAgent });
      gameIdRef.current = state.id;
      setGame(state);
    } catch (err) {
      setError(errorMessage(err, 'Could not start a new game.'));
      setGame(null);
      gameIdRef.current = null;
    } finally {
      setLoading(false);
    }
  }, []);

  const playMove = useCallback(async (col: number) => {
    const id = gameIdRef.current;
    if (!id) return;
    setBusy(true);
    setError(null);
    try {
      const state = await api.move(id, col);
      setGame(state);
    } catch (err) {
      setError(errorMessage(err, 'That move was rejected.'));
    } finally {
      setBusy(false);
    }
  }, []);

  const requestAgentMove = useCallback(async (): Promise<GameState | null> => {
    const id = gameIdRef.current;
    if (!id) return null;
    setBusy(true);
    setError(null);
    try {
      const state = await api.agentMove(id);
      setGame(state);
      return state;
    } catch (err) {
      setError(errorMessage(err, 'The agent could not move.'));
      return null;
    } finally {
      setBusy(false);
    }
  }, []);

  const clearError = useCallback(() => setError(null), []);

  return { game, loading, busy, error, newGame, playMove, requestAgentMove, clearError };
}
