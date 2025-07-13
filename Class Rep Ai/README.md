# Class Rep AI - Private React Native App with OTA Updates

A private React Native app built with Expo for managing class assignments with automatic over-the-air (OTA) updates.

## Features

- 📱 **Standalone APK**: Build APK files for private distribution
- 🔄 **OTA Updates**: Automatic over-the-air updates when code changes are pushed to GitHub
- 👥 **User Restriction**: Access limited to 3 authorized users
- 📝 **Assignment Management**: Create, edit, and manage class assignments
- 📢 **Daily Notifications**: Automated daily reminders at 17:45 to share assignments
- 💬 **WhatsApp Integration**: Share assignments directly to WhatsApp
- 🎨 **Theme Support**: Light and dark theme support

## Setup Instructions

### 1. Configure Expo Project

1. **Install Expo CLI**:
   ```bash
   npm install -g @expo/cli
   ```

2. **Login to Expo**:
   ```bash
   npx expo login
   ```

3. **Update app.json**:
   - Replace `your-project-id` with your actual Expo project ID
   - Replace `your-expo-username` with your Expo username
   - Update `bundleIdentifier` and `package` to your desired app identifiers

### 2. Set up EAS Build

1. **Install EAS CLI**:
   ```bash
   npm install -g eas-cli
   ```

2. **Configure EAS**:
   ```bash
   eas build:configure
   ```

3. **Build APK**:
   ```bash
   # For preview build
   eas build --platform android --profile preview
   
   # For production build
   eas build --platform android --profile production
   ```

### 3. Configure User Authentication

Update the `AUTHORIZED_USERS` array in `components/UserAuth.tsx` with your actual users:

```typescript
const AUTHORIZED_USERS: AuthorizedUser[] = [
  {
    id: '1',
    username: 'your-username',
    email: 'your-email@example.com',
    role: 'Admin'
  },
  // Add up to 3 users
];
```

### 4. Set up GitHub Actions (Optional)

1. **Add Repository Secrets**:
   Go to your GitHub repository settings → Secrets and variables → Actions, and add:
   - `EXPO_TOKEN`: Your Expo access token
   - `EXPO_USERNAME`: Your Expo username
   - `EXPO_PASSWORD`: Your Expo password

2. **Enable Actions**:
   The workflow file is already included in `.github/workflows/publish.yml`

### 5. OTA Updates Configuration

1. **Update Project ID**:
   In `app.json`, replace `your-project-id` with your actual Expo project ID from the Expo dashboard.

2. **Test OTA Updates**:
   ```bash
   # Publish an update
   npx expo publish
   
   # Check for updates in the app
   # The app will automatically check for updates on launch
   ```

## Usage

### Building and Distributing

1. **Build APK**:
   ```bash
   eas build --platform android --profile production
   ```

2. **Download APK**:
   Once the build is complete, download the APK from the EAS dashboard or the provided link.

3. **Distribute**:
   Share the APK file with your authorized users via email, cloud storage, or direct transfer.

### Publishing Updates

1. **Automatic** (via GitHub Actions):
   - Push changes to the `main` branch
   - GitHub Actions will automatically publish the update

2. **Manual**:
   ```bash
   npx expo publish
   ```

### Daily Notifications

- The app schedules daily notifications at 17:45
- Users receive reminders to share assignment details
- Notifications include a direct link to share via WhatsApp

## App Structure

```
Class Rep Ai/
├── app/
│   ├── (tabs)/
│   │   ├── assignments.tsx    # Main assignments screen
│   │   ├── attendance.tsx     # Attendance tracking
│   │   ├── reports.tsx        # Reports and analytics
│   │   └── settings.tsx       # App settings
│   ├── _layout.tsx            # Root layout with auth and OTA
│   └── login.tsx              # Login screen
├── components/
│   ├── OTAUpdateChecker.tsx   # OTA update handling
│   └── UserAuth.tsx           # User authentication
├── context/
│   ├── DataContext.tsx        # Data management
│   └── ThemeContext.tsx       # Theme management
├── eas.json                   # EAS build configuration
├── app.json                   # Expo configuration
└── .github/workflows/
    └── publish.yml            # GitHub Actions workflow
```

## Customization

### Adding New Users

Edit `components/UserAuth.tsx` and update the `AUTHORIZED_USERS` array:

```typescript
const AUTHORIZED_USERS: AuthorizedUser[] = [
  // Add your users here (max 3)
];
```

### Changing Notification Time

Edit `app/(tabs)/assignments.tsx` and update the `scheduleDailyNotification` function:

```typescript
trigger: {
  type: Notifications.SchedulableTriggerInputTypes.CALENDAR,
  hour: 17,    // Change hour (24-hour format)
  minute: 45,  // Change minute
  repeats: true,
},
```

### Customizing WhatsApp Messages

Edit the message templates in `shareAssignmentToWhatsApp` and `shareAllAssignmentsToWhatsApp` functions.

## Troubleshooting

### Common Issues

1. **OTA Updates Not Working**:
   - Ensure the project ID in `app.json` matches your Expo project
   - Check that the app version matches the published update
   - Verify that Updates are enabled in the build

2. **Authentication Issues**:
   - Verify user credentials match the `AUTHORIZED_USERS` array
   - Check AsyncStorage permissions

3. **Notification Issues**:
   - Ensure notification permissions are granted
   - Check that the notification trigger is properly configured

### Build Issues

1. **APK Build Fails**:
   - Check EAS build logs for specific errors
   - Ensure all dependencies are compatible with the Expo SDK version

2. **GitHub Actions Fails**:
   - Verify all required secrets are set in the repository
   - Check the workflow logs for specific errors

## License

This project is for private use only. Not for commercial distribution.

## Support

For issues or questions, please contact the project maintainer.