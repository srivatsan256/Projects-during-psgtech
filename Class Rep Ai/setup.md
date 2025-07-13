# Quick Setup Guide

## Prerequisites

1. **Node.js** (v16 or later)
2. **npm** or **yarn**
3. **Expo CLI** and **EAS CLI**
4. **GitHub account** (for automated publishing)
5. **Expo account** (free tier is sufficient)

## Step-by-Step Setup

### 1. Initial Configuration

1. **Install required CLIs**:
   ```bash
   npm install -g @expo/cli eas-cli
   ```

2. **Login to Expo**:
   ```bash
   npx expo login
   ```

3. **Create a new Expo project** (if starting from scratch):
   ```bash
   npx create-expo-app --template
   ```

### 2. Configure Your Project

1. **Update app.json**:
   - Replace `your-project-id` with your actual Expo project ID
   - Replace `your-expo-username` with your Expo username
   - Update app name, slug, and package identifiers

2. **Set up authorized users**:
   - Edit `components/UserAuth.tsx`
   - Replace the `AUTHORIZED_USERS` array with your actual users

3. **Configure EAS**:
   ```bash
   eas build:configure
   ```

### 3. Build Your First APK

1. **Build for Android**:
   ```bash
   eas build --platform android --profile production
   ```

2. **Download the APK**:
   - Check the EAS dashboard for your build
   - Download the APK when ready

### 4. Set Up OTA Updates

1. **Get your project ID**:
   - Visit the Expo dashboard
   - Copy your project ID
   - Update `app.json` with the correct project ID

2. **Test OTA publishing**:
   ```bash
   npx expo publish
   ```

### 5. Configure GitHub Actions (Optional)

1. **Add secrets to your GitHub repository**:
   - Go to Settings → Secrets and variables → Actions
   - Add these secrets:
     - `EXPO_TOKEN`: Your Expo access token
     - `EXPO_USERNAME`: Your Expo username
     - `EXPO_PASSWORD`: Your Expo password

2. **Push to main branch**:
   - The workflow will automatically publish updates

### 6. Test the Complete Flow

1. **Install APK on your device**
2. **Login with authorized credentials**
3. **Make a code change**
4. **Publish update**: `npx expo publish`
5. **Restart the app** - it should check for and apply the update

## Important Notes

- **Private Distribution**: This app is configured for private use only
- **User Limit**: Only 3 users can access the app (configured in UserAuth.tsx)
- **Notification Time**: Daily notifications are set for 17:45 (5:45 PM)
- **WhatsApp Integration**: Requires WhatsApp to be installed on the device

## Troubleshooting

### Common Issues

1. **Build fails**: Check EAS build logs for specific errors
2. **OTA not working**: Verify project ID matches in app.json
3. **Auth issues**: Check user credentials in UserAuth.tsx
4. **Notifications not working**: Ensure permissions are granted

### Getting Help

- Check the main README.md for detailed documentation
- Review Expo documentation for specific issues
- Check GitHub Actions logs for publishing issues

## Next Steps

1. Distribute your APK to authorized users
2. Test the notification system
3. Customize the app for your specific needs
4. Set up automated publishing if desired

## Security Considerations

- Keep your Expo tokens secure
- Don't commit sensitive information to version control
- Regularly update dependencies
- Use environment variables for sensitive configurations