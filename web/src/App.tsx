import { useCallback, useState } from 'react';
import { NavBar, type Screen } from './components/NavBar';
import { PlayScreen } from './components/PlayScreen';
import { AgentsScreen } from './components/AgentsScreen';
import { LeaderboardScreen } from './components/LeaderboardScreen';
import { AboutScreen } from './components/AboutScreen';
import { Footer } from './components/Footer';
import { useTheme } from './lib/theme';

function App() {
  const [screen, setScreen] = useState<Screen>('play');
  const [theme, toggleTheme] = useTheme();
  // Set by the Agents screen's "Play" button so the Play screen's opponent
  // picker preselects that agent; consumed (and cleared) once PlayScreen
  // reads it, so switching to Play by any other route falls back to the
  // usual default.
  const [presetOpponent, setPresetOpponent] = useState<string | null>(null);

  const handlePlay = useCallback((agentName: string) => {
    setPresetOpponent(agentName);
    setScreen('play');
  }, []);

  const handleScreen = useCallback((s: Screen) => {
    if (s !== 'play') setPresetOpponent(null);
    setScreen(s);
  }, []);

  return (
    <div className="flex min-h-full flex-col bg-[var(--page)]">
      <NavBar screen={screen} onScreen={handleScreen} theme={theme} onToggleTheme={toggleTheme} />
      <main className="flex-1">
        {screen === 'play' ? (
          <PlayScreen presetOpponent={presetOpponent} />
        ) : screen === 'agents' ? (
          <AgentsScreen onPlay={handlePlay} />
        ) : screen === 'leaderboard' ? (
          <LeaderboardScreen />
        ) : (
          <AboutScreen />
        )}
      </main>
      <Footer />
    </div>
  );
}

export default App;
