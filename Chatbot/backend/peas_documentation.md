# PEAS Auto-Tuning System Documentation

## Overview

The PEAS (Performance Environment Agent Sensor) auto-tuning system is an intelligent model optimization feature that automatically adjusts user performance scores based on your specific requirements.

### Key Feature
**If PEAS sum = 4 OR ceiling of any component = 4, then reduce total to 2 for auto-tuning**

## PEAS Components

### 1. **Performance (P)**
- **Purpose**: Measures response time and system performance
- **Metrics**: PDF fetch speed, content extraction speed, LLM response time
- **Range**: 0.01 - 1.0

### 2. **Environment (E)**
- **Purpose**: Tracks user environment and context
- **Metrics**: PIN code verification, address validation, location context
- **Range**: 0.5 - 1.0

### 3. **Actuators (A)**
- **Purpose**: Measures system actions and PDF processing
- **Metrics**: PDF availability, model loading, document processing
- **Range**: 0.1 - 1.0

### 4. **Sensors (S)**
- **Purpose**: Tracks input detection and data collection
- **Metrics**: Question analysis, media processing, user input parsing
- **Range**: 0.5 - 1.0

## Auto-Tuning Logic

### Trigger Conditions
The auto-tuning system activates when:
1. **Sum Condition**: Total PEAS sum equals 4.0 (±0.01)
2. **Ceiling Condition**: Any component's ceiling equals 4
3. **Force Tune**: Manual trigger

### Tuning Process
1. **Detection**: System detects trigger condition
2. **Scaling**: Reduces total sum to 2.0 for optimization
3. **Balancing**: Redistributes scores proportionally
4. **Capping**: Ensures no component exceeds 1.0
5. **Logging**: Records tuning metadata

### Example
```
Before Tuning:
- Performance: 1.5
- Environment: 1.0
- Actuators: 0.8
- Sensors: 0.7
- Total: 4.0 ✓ (Triggers auto-tuning)

After Tuning:
- Performance: 0.75
- Environment: 0.50
- Actuators: 0.40
- Sensors: 0.35
- Total: 2.0 ✓ (Optimized)
```

## API Endpoints

### 1. Get PEAS Status
```http
POST /api/peas_status
Content-Type: application/json

{
  "username": "john_doe"
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "user": "john_doe",
    "components": {
      "Performance": 0.75,
      "Environment": 0.50,
      "Actuators": 0.40,
      "Sensors": 0.35
    },
    "sum": 2.0,
    "ceiling": 1,
    "needs_tuning": false,
    "last_tuning": "2024-01-15T10:30:00",
    "tuning_reason": "sum_4_or_ceiling_4",
    "original_sum": 4.0,
    "tuned_sum": 2.0
  }
}
```

### 2. Manual PEAS Tuning
```http
POST /api/peas_tune
Content-Type: application/json

{
  "username": "john_doe",
  "force_tune": false
}
```

**Response:**
```json
{
  "success": true,
  "message": "PEAS auto-tuning completed",
  "data": {
    // ... PEAS status data
  }
}
```

### 3. Batch PEAS Tuning
```http
POST /api/peas_batch_tune
```

**Response:**
```json
{
  "success": true,
  "message": "PEAS batch tuning completed for 3 users",
  "tuned_users": ["john_doe", "jane_smith", "bob_wilson"]
}
```

### 4. PEAS Overview
```http
GET /api/peas_overview
```

**Response:**
```json
{
  "success": true,
  "data": {
    "total_users": 10,
    "users_needing_tuning": 3,
    "users": [
      // ... array of user PEAS status
    ]
  }
}
```

## Integration Points

### 1. **Onboarding Integration**
- New users get initial PEAS tuning with `force_tune=True`
- Ensures optimal starting configuration

### 2. **Query Processing Integration**
- PEAS auto-tuning check after each user interaction
- Background processing prevents response delays

### 3. **Scheduled Tuning**
- Automatic batch tuning every 6 hours
- Maintains optimal performance across all users

## Configuration

### Environment Variables
```bash
# PEAS tuning interval (in seconds)
PEAS_TUNING_INTERVAL=21600  # 6 hours

# Target sum for auto-tuning
PEAS_TARGET_SUM=2.0

# Enable/disable scheduled tuning
PEAS_SCHEDULED_TUNING=true
```

### Configuration in Code
```python
# Modify these values in chatbot_optimized.py
TARGET_SUM = 2.0  # Auto-tuning target
TUNING_INTERVAL = 6 * 3600  # 6 hours
```

## Monitoring and Logging

### Log Messages
```
[INFO] PEAS Analysis for john_doe: Sum=4.000, Ceiling=1
[INFO] Components: P=1.500, E=1.000, A=0.800, S=0.700
[INFO] PEAS Auto-tuning triggered for john_doe
[INFO] PEAS Auto-tuning completed for john_doe
[INFO] New components: P=0.750, E=0.500, A=0.400, S=0.350
[INFO] New sum: 2.000
```

### Monitoring Dashboard
Access PEAS overview at: `GET /api/peas_overview`

## Best Practices

### 1. **Regular Monitoring**
- Check PEAS overview weekly
- Monitor users needing tuning
- Review tuning effectiveness

### 2. **Manual Tuning**
- Use `force_tune=true` for problem users
- Monitor performance after manual tuning
- Document tuning decisions

### 3. **Performance Optimization**
- Users with sum=2.0 have optimal performance
- Higher sums may indicate over-optimization
- Lower sums may indicate under-utilization

### 4. **Troubleshooting**
- Check logs for tuning failures
- Verify API responses
- Monitor system resources

## Advanced Features

### 1. **Custom Tuning Logic**
```python
def custom_peas_tuning(user_key, learning_data):
    """Custom PEAS tuning logic."""
    # Implement custom tuning rules
    pass
```

### 2. **Performance Metrics**
```python
# Track tuning effectiveness
def measure_tuning_impact(user_key, before_scores, after_scores):
    """Measure impact of PEAS tuning."""
    performance_gain = calculate_performance_improvement(before_scores, after_scores)
    return performance_gain
```

### 3. **A/B Testing**
```python
# Test different tuning strategies
def ab_test_tuning(user_key, strategy='default'):
    """A/B test different tuning strategies."""
    if strategy == 'aggressive':
        return tune_aggressive(user_key)
    else:
        return auto_tune_peas_scores(user_key)
```

## Troubleshooting

### Common Issues

1. **PEAS scores not updating**
   - Check if learning data is being saved
   - Verify API permissions
   - Review error logs

2. **Auto-tuning not triggering**
   - Verify trigger conditions (sum=4 or ceiling=4)
   - Check scheduled tuning is enabled
   - Review background processing

3. **Performance degradation after tuning**
   - Monitor response times
   - Check component balance
   - Consider manual re-tuning

### Debug Commands

```bash
# Check PEAS status
curl -X POST http://localhost:5000/api/peas_status \
  -H "Content-Type: application/json" \
  -d '{"username": "test_user"}'

# Force tuning
curl -X POST http://localhost:5000/api/peas_tune \
  -H "Content-Type: application/json" \
  -d '{"username": "test_user", "force_tune": true}'

# Get overview
curl -X GET http://localhost:5000/api/peas_overview
```

## Integration with Frontend

### JavaScript API Client
```javascript
class PEASClient {
  async getPEASStatus(username) {
    const response = await fetch('/api/peas_status', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({username})
    });
    return response.json();
  }
  
  async tunePEAS(username, forceTune = false) {
    const response = await fetch('/api/peas_tune', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({username, force_tune: forceTune})
    });
    return response.json();
  }
}
```

### React Component
```jsx
function PEASStatus({ username }) {
  const [peasData, setPeasData] = useState(null);
  
  useEffect(() => {
    fetchPEASStatus();
  }, [username]);
  
  const fetchPEASStatus = async () => {
    const client = new PEASClient();
    const data = await client.getPEASStatus(username);
    setPeasData(data);
  };
  
  return (
    <div>
      <h3>PEAS Status</h3>
      {peasData && (
        <div>
          <p>Sum: {peasData.data.sum}</p>
          <p>Needs Tuning: {peasData.data.needs_tuning ? 'Yes' : 'No'}</p>
          {/* ... render components */}
        </div>
      )}
    </div>
  );
}
```

## Conclusion

The PEAS auto-tuning system provides intelligent performance optimization by automatically adjusting user scores when they reach optimal thresholds. This ensures consistent performance while preventing over-optimization that could degrade user experience.

The system operates transparently in the background while providing comprehensive monitoring and manual control options for administrators.