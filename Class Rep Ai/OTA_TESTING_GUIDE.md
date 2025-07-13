# OTA Updates Testing Guide

## Overview
This guide explains how to test OTA (Over-The-Air) updates using `expo publish` for your standalone APK.

## Prerequisites
- Standalone APK built with `eas build` and installed on device
- Expo CLI installed: `npm install -g @expo/cli`
- Logged into Expo: `npx expo login`

## Testing Workflow

### 1. Initial Setup Check
Before making any changes, verify your setup:

```bash
# Check your current Expo project
npx expo whoami

# Verify your app.json configuration
cat app.json | grep -A 5 "updates"
```

### 2. Make a Code Change
Make a visible change to test the update:

**Example: Update the welcome message in App.js**
```javascript
// Change this line:
<Text style={styles.cardTitle}>Welcome!</Text>
// To:
<Text style={styles.cardTitle}>Welcome! (Updated v1.1)</Text>
```

**Or in your existing app, make any visible change like:**
- Update text content
- Change colors
- Add/remove UI elements
- Update functionality

### 3. Publish the Update
Run the publish command:

```bash
npx expo publish
```

**Expected output:**
```
📝 Expo SDK: 53.0.0
🔗 Release channel: default
📱 Android index: https://classic-assets.eascdn.net/...
📱 Android bundle: https://classic-assets.eascdn.net/...
💿 Assets: https://classic-assets.eascdn.net/...
✅ Published
```

### 4. Test on Device

#### Method 1: Restart App
1. **Close your app completely** (don't just minimize)
2. **Reopen the app**
3. **Watch the console logs** if you have Metro/debugging enabled
4. **Look for update modal** if an update is available

#### Method 2: Force Close and Wait
1. **Force close the app** (swipe away from recent apps)
2. **Wait 10-15 seconds** for the update to propagate
3. **Reopen the app**
4. **Check for your changes**

### 5. Verify Update Applied
After reopening the app:
- Look for your code changes
- Check console logs for update messages
- If using the simple App.js, you should see your updated text

## Troubleshooting

### Update Not Showing
1. **Check publish status:**
   ```bash
   npx expo publish --clear
   ```

2. **Verify release channel:**
   ```bash
   npx expo publish --release-channel production
   ```

3. **Clear app cache:**
   - Uninstall and reinstall the APK
   - Or clear app data in Android settings

### Console Debugging
Enable remote debugging to see console logs:
1. Open app
2. Shake device (or press hardware menu button)
3. Select "Debug"
4. Open Chrome DevTools
5. Look for update-related console messages

### Common Issues

#### Issue: "Updates are not enabled"
**Solution:** This message appears in development mode. Updates only work in production builds.

#### Issue: No update detected
**Possible causes:**
- App is in development mode
- Same code published twice
- Network connectivity issues
- App not properly closed/reopened

#### Issue: App crashes after update
**Solution:** 
```bash
# Clear the update cache
npx expo publish --clear

# Rebuild if necessary
eas build --platform android --profile production
```

## Testing Scenarios

### Test 1: Text Update
1. Change any text in your app
2. Publish: `npx expo publish`
3. Restart app
4. Verify text changed

### Test 2: Style Update
1. Change colors or styling
2. Publish: `npx expo publish`
3. Restart app
4. Verify visual changes

### Test 3: Functionality Update
1. Add/modify a button or feature
2. Publish: `npx expo publish`
3. Restart app
4. Test the new functionality

## Best Practices

### 1. Always Test Changes
- Test every change before publishing
- Use a test device or emulator
- Verify updates work as expected

### 2. Gradual Updates
- Make small, incremental changes
- Test each update thoroughly
- Keep track of what was changed

### 3. Rollback Strategy
If an update causes issues:
```bash
# Revert your code changes
git revert HEAD

# Publish the reverted version
npx expo publish
```

### 4. Version Tracking
Update your version number in package.json for major changes:
```json
{
  "version": "1.0.1"
}
```

## Advanced Testing

### Testing with Multiple Devices
1. Install APK on multiple devices
2. Publish update
3. Verify all devices receive the update
4. Test different network conditions

### Testing Update Timing
1. Publish update
2. Test how quickly devices receive it
3. Verify behavior with poor connectivity

### Testing Update Conflicts
1. Make conflicting changes
2. Test how the app handles update failures
3. Verify graceful degradation

## Monitoring Updates

### Check Update Status
```bash
# View published releases
npx expo publish --release-channel default

# Check what's currently published
npx expo publish --dry-run
```

### Update Analytics
- Monitor how many users receive updates
- Track update success rates
- Monitor app crashes post-update

## Emergency Procedures

### If Update Breaks App
1. **Immediate fix:**
   ```bash
   git revert HEAD
   npx expo publish
   ```

2. **If revert doesn't work:**
   ```bash
   # Publish last known good version
   git checkout <last-good-commit>
   npx expo publish
   ```

3. **Nuclear option:**
   - Rebuild and redistribute APK
   - Clear all published updates

## Summary

The OTA update workflow is:
1. **Make code changes**
2. **Run `npx expo publish`**
3. **Close and reopen app on device**
4. **Verify changes applied**

Remember: OTA updates only work with production builds (APK), not in development mode with Expo Go.