import { BrowserRouter, Routes, Route, NavLink } from 'react-router-dom';
import PredictPage from './pages/PredictPage';
import ProteinPage from './pages/ProteinPage';
import HistoryPage from './pages/HistoryPage';

export default function App() {
  return (
    <BrowserRouter>
      <div className="min-h-screen bg-gray-50">
        <nav className="bg-white shadow">
          <div className="max-w-7xl mx-auto px-4 py-3 flex items-center gap-8">
            <span className="text-xl font-bold text-blue-900">BPA-22</span>
            <NavLink to="/" end className={({ isActive }) => `px-3 py-1 rounded ${isActive ? 'bg-blue-100 text-blue-800 font-semibold' : 'text-gray-600 hover:text-gray-900'}`}>
              Image Analysis
            </NavLink>
            <NavLink to="/proteins" className={({ isActive }) => `px-3 py-1 rounded ${isActive ? 'bg-blue-100 text-blue-800 font-semibold' : 'text-gray-600 hover:text-gray-900'}`}>
              Protein Ranking
            </NavLink>
            <NavLink to="/history" className={({ isActive }) => `px-3 py-1 rounded ${isActive ? 'bg-blue-100 text-blue-800 font-semibold' : 'text-gray-600 hover:text-gray-900'}`}>
              History
            </NavLink>
          </div>
        </nav>
        <main className="max-w-7xl mx-auto px-4 py-6">
          <Routes>
            <Route path="/" element={<PredictPage />} />
            <Route path="/proteins" element={<ProteinPage />} />
            <Route path="/history" element={<HistoryPage />} />
          </Routes>
        </main>
      </div>
    </BrowserRouter>
  );
}
