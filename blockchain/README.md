# Blockchain Flask-React Integration

This project integrates a Flask backend with a React frontend for a complete blockchain application. The backend provides RESTful API endpoints for blockchain operations, while the frontend offers a modern, interactive user interface.

## 🏗️ Architecture

### Backend (Flask)
- **Framework**: Flask with CORS support
- **Blockchain**: Custom blockchain implementation with proof-of-work
- **Network**: P2P network simulation
- **API**: RESTful endpoints for blockchain operations

### Frontend (React)
- **Framework**: React with TypeScript
- **Styling**: Tailwind CSS
- **State Management**: Custom hooks with API integration
- **UI Components**: Modern, responsive design

## 🚀 Quick Start

### Prerequisites
- Python 3.7+
- Node.js 14+
- npm or yarn

### Setup

1. **Clone and navigate to the project:**
   ```bash
   cd blockchain
   ```

2. **Run the setup script:**
   ```bash
   python setup.py
   ```

3. **Start the application:**
   ```bash
   cd "p2p-blockchain backend"
   python app.py
   ```

4. **Access the application:**
   - Open your browser to `http://localhost:5000`
   - The React frontend will be served automatically

## 📡 API Endpoints

### Blockchain Operations
- `GET /api/chain` - Get the full blockchain
- `POST /api/transaction/new` - Create a new transaction
- `GET /api/mine` - Mine a new block
- `GET /api/pending` - Get pending transactions
- `GET /api/stats` - Get blockchain statistics
- `GET /api/validate` - Validate the blockchain

### Network Operations
- `POST /api/nodes/register` - Register new nodes
- `GET /api/nodes/sync` - Sync with network nodes

### Legacy Endpoints
All endpoints are also available without the `/api` prefix for backward compatibility.

## 🔧 Integration Features

### Frontend-Backend Communication
- **Real-time Updates**: Frontend polls backend for latest blockchain state
- **Error Handling**: Comprehensive error handling and user feedback
- **Loading States**: Visual indicators for ongoing operations
- **Type Safety**: TypeScript interfaces for API responses

### CORS Support
- Configured for cross-origin requests
- Supports both development and production environments

### Data Flow
1. User interacts with React frontend
2. Frontend makes API calls to Flask backend
3. Backend processes blockchain operations
4. Frontend updates UI with new data

## 🎯 Key Features

### Blockchain Features
- **Proof of Work**: Configurable difficulty mining
- **Transaction Pool**: Pending transaction management
- **Chain Validation**: Integrity verification
- **P2P Network**: Multi-node support

### Frontend Features
- **Interactive UI**: Modern, responsive design
- **Real-time Updates**: Live blockchain data
- **Dark/Light Mode**: Theme switching
- **Error Handling**: User-friendly error messages
- **Loading States**: Visual feedback for operations

### API Features
- **RESTful Design**: Clean, consistent API structure
- **Error Handling**: Proper HTTP status codes
- **Data Validation**: Input validation and sanitization
- **Backward Compatibility**: Legacy endpoint support

## 🛠️ Development

### Backend Development
```bash
cd "p2p-blockchain backend"
pip install -r requirements.txt
python app.py --port 5000
```

### Frontend Development
```bash
cd "p2p-blockchain frontend"
npm install
npm run dev  # Development server
npm run build  # Production build
```

### API Testing
You can test the API endpoints using curl:

```bash
# Get blockchain
curl http://localhost:5000/api/chain

# Create transaction
curl -X POST http://localhost:5000/api/transaction/new \
  -H "Content-Type: application/json" \
  -d '{"sender": "Alice", "recipient": "Bob", "amount": 50}'

# Mine block
curl http://localhost:5000/api/mine

# Get stats
curl http://localhost:5000/api/stats
```

## 📁 Project Structure

```
blockchain/
├── p2p-blockchain backend/
│   ├── app.py                 # Flask application
│   ├── blockchain.py          # Blockchain implementation
│   ├── network.py            # P2P network
│   ├── requirements.txt      # Python dependencies
│   └── templates/
│       └── index.html        # Legacy HTML interface
├── p2p-blockchain frontend/
│   ├── src/
│   │   ├── components/       # React components
│   │   ├── hooks/           # Custom hooks
│   │   ├── services/        # API services
│   │   ├── types/           # TypeScript types
│   │   └── App.tsx          # Main app component
│   ├── package.json         # Node.js dependencies
│   └── dist/               # Built frontend files
├── setup.py                # Setup script
└── README.md               # This file
```

## 🔄 Data Flow

1. **Transaction Creation**:
   - User submits transaction via frontend
   - Frontend sends POST request to `/api/transaction/new`
   - Backend adds transaction to pending pool
   - Frontend updates UI with new pending transaction

2. **Block Mining**:
   - User clicks mine button
   - Frontend sends GET request to `/api/mine`
   - Backend mines new block with pending transactions
   - Frontend updates UI with new block

3. **Chain Synchronization**:
   - Frontend periodically fetches chain data
   - Backend provides latest blockchain state
   - Frontend updates visualization

## 🐛 Troubleshooting

### Common Issues

1. **CORS Errors**:
   - Ensure Flask-CORS is installed
   - Check that CORS is enabled in app.py

2. **API Connection Failed**:
   - Verify backend is running on correct port
   - Check firewall settings
   - Confirm API_BASE_URL in frontend

3. **Build Errors**:
   - Run `npm install` in frontend directory
   - Check Node.js and npm versions
   - Clear npm cache if needed

### Port Configuration
- Backend runs on port 5000 by default
- Use `--port` argument to change: `python app.py --port 5001`
- Update `API_BASE_URL` in frontend if port changes

## 📊 Performance Considerations

- **Mining**: Adjust difficulty for faster/slower mining
- **Updates**: Consider WebSocket for real-time updates
- **Caching**: Implement caching for frequently accessed data
- **Pagination**: Add pagination for large blockchain data

## 🔐 Security Notes

- This is a demonstration project
- Implement proper authentication for production
- Add input validation and sanitization
- Use HTTPS in production environments
- Consider rate limiting for API endpoints

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is for educational purposes. Feel free to use and modify as needed.

---

**Happy Blockchain Building! 🔗**