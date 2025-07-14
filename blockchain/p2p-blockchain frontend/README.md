# P2P Blockchain Network Frontend

A modern React + TypeScript frontend for visualizing and interacting with the peer-to-peer blockchain network backend.

## Features
- Live network visualization and blockchain explorer
- Transaction creation and mining interface
- Real-time stats and node management
- Tailwind CSS for styling, Vite for fast development

## Requirements
- Node.js (v18+ recommended)
- npm
- Backend nodes running (see backend README)

## Setup & Development

1. **Install dependencies:**
   ```bash
   npm install
   ```
2. **Start the development server:**
   ```bash
   npm run dev
   ```
   - The app will be available at the local address provided by Vite (usually http://localhost:5173).
   - Make sure the backend nodes (Flask servers) are running on ports 5000-5004 for full functionality.

## Build for Production
```bash
npm run build
```
- Outputs static files to the `dist` directory.

## Preview Production Build
```bash
npm run preview
```
- Serves the production build locally for testing.

## Lint the Code
```bash
npm run lint
```

## Project Structure
- `src/` — React components, hooks, and logic
- `index.html` — Entry point
- `tailwind.config.js` — Tailwind CSS configuration
- `vite.config.ts` — Vite configuration

## Integration with Backend
- The frontend communicates with the Flask backend nodes (default: ports 5000-5004).
- Ensure the backend is running before using the frontend for live blockchain operations.

## Troubleshooting
- **Backend not reachable:**
  - Make sure the backend nodes are running and accessible on the expected ports.
- **Port conflicts:**
  - Change the Vite dev server port in `vite.config.ts` if needed.
- **Styling issues:**
  - Ensure Tailwind CSS is installed and configured properly.

---

For backend setup and API details, see the backend's `README.md`.