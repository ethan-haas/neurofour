import { useState } from 'react';
import { NavBar, type Screen } from './components/NavBar';
import { PlayScreen } from './components/PlayScreen';
import { LeaderboardScreen } from './components/LeaderboardScreen';
import { useTheme } from './lib/theme';

function App() {
  const [screen, setScreen] = useState<Screen>('play');
  const [theme, toggleTheme] = useTheme();

  return (
    <div className="min-h-full bg-[var(--page)]">
      <NavBar screen={screen} onScreen={setScreen} theme={theme} onToggleTheme={toggleTheme} />
      <main>{screen === 'play' ? <PlayScreen /> : <LeaderboardScreen />}</main>
    </div>
  );
}

export default App;
