```javascript
import { Routes, Route } from 'react-router-dom'
import Home from './pages/Home'
import Macro from './pages/Macro'
import Polymarket from './pages/Polymarket'
import Layout from './components/Layout'

function App() {
  return (
    <Routes>
      <Route path="/" element={<Layout />}>
        <Route index element={<Home />} />
        <Route path="macro" element={<Macro />} />
        <Route path="polymarket" element={<Polymarket />} />
      </Route>
    </Routes>
  )
}

export default App
```
