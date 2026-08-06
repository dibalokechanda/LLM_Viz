import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
// Variable weights, self-hosted. Without these the app silently falls back to
// the OS UI font, which is what made it read as undesigned.
import '@fontsource-variable/inter'
import '@fontsource-variable/jetbrains-mono'
import App from './App'
// styles.css carries the inherited layout and content components; theme.css
// replaces its monochrome tokens with the validated colour system; llmviz.css
// adds what is specific to this app. Order matters — theme must win over
// styles, and llmviz reads theme's variables.
import './styles.css'
import './theme.css'
import './llmviz.css'
import './stack.css'
import './editor.css'

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <App />
  </StrictMode>,
)
