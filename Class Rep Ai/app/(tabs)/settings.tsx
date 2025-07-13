import React, { useState, useEffect } from 'react';
import { View, Text, TouchableOpacity, StyleSheet, ScrollView, Switch, Alert } from 'react-native';
import { useData } from '@/context/DataContext';
import { useTheme } from '@/context/ThemeContext';
import { Moon, Sun, Download, Upload, Bell, Trash2, LogOut } from 'lucide-react-native';
import { router } from 'expo-router';

export default function SettingsScreen() {
  const { exportAllData, importData, clearAllData, getUserData } = useData();
  const { theme, isDark, toggleTheme } = useTheme();
  const [userData, setUserData] = useState<any>(null);
  const [notificationsEnabled, setNotificationsEnabled] = useState(true);

  useEffect(() => {
    loadUserData();
  }, []);

  const loadUserData = async () => {
    const user = await getUserData();
    setUserData(user);
  };

  const handleBackup = async () => {
    await exportAllData();
    Alert.alert('Success', 'Data exported successfully');
  };

  const handleRestore = async () => {
    Alert.alert(
      'Restore Data',
      'This will replace all current data. Are you sure?',
      [
        { text: 'Cancel', style: 'cancel' },
        { text: 'Restore', style: 'destructive', onPress: () => importData() }
      ]
    );
  };

  const handleClearData = async () => {
    Alert.alert(
      'Clear All Data',
      'This will permanently delete all data. Are you sure?',
      [
        { text: 'Cancel', style: 'cancel' },
        { 
          text: 'Delete All', 
          style: 'destructive', 
          onPress: async () => {
            await clearAllData();
            router.replace('/login');
          }
        }
      ]
    );
  };

  const handleLogout = () => {
    Alert.alert(
      'Logout',
      'Are you sure you want to logout?',
      [
        { text: 'Cancel', style: 'cancel' },
        { 
          text: 'Logout', 
          style: 'destructive', 
          onPress: async () => {
            await clearAllData();
            router.replace('/login');
          }
        }
      ]
    );
  };

  const styles = createStyles(theme);

  return (
    <ScrollView style={styles.container}>
      <View style={styles.header}>
        <Text style={styles.title}>Settings</Text>
      </View>

      <View style={styles.section}>
        <Text style={styles.sectionTitle}>👤 Account</Text>
        <View style={styles.card}>
          <View style={styles.userInfo}>
            <Text style={styles.userName}>{userData?.studentName}</Text>
            <Text style={styles.userRoll}>Roll No: {userData?.rollNumber}</Text>
          </View>
          <TouchableOpacity style={styles.logoutButton} onPress={handleLogout}>
            <LogOut size={20} color="#DC2626" />
            <Text style={styles.logoutButtonText}>Logout</Text>
          </TouchableOpacity>
        </View>
      </View>

      <View style={styles.section}>
        <Text style={styles.sectionTitle}>🎨 Appearance</Text>
        <View style={styles.card}>
          <View style={styles.settingItem}>
            <View style={styles.settingLeft}>
              {isDark ? <Moon size={24} color={theme.text} /> : <Sun size={24} color={theme.text} />}
              <View style={styles.settingText}>
                <Text style={styles.settingTitle}>Dark Mode</Text>
                <Text style={styles.settingSubtitle}>Switch between light and dark themes</Text>
              </View>
            </View>
            <Switch
              value={isDark}
              onValueChange={toggleTheme}
              trackColor={{ false: '#D1D5DB', true: '#2563EB' }}
              thumbColor={isDark ? '#FFFFFF' : '#F3F4F6'}
            />
          </View>
        </View>
      </View>

      <View style={styles.section}>
        <Text style={styles.sectionTitle}>🔔 Notifications</Text>
        <View style={styles.card}>
          <View style={styles.settingItem}>
            <View style={styles.settingLeft}>
              <Bell size={24} color={theme.text} />
              <View style={styles.settingText}>
                <Text style={styles.settingTitle}>Assignment Reminders</Text>
                <Text style={styles.settingSubtitle}>Get notified about upcoming deadlines</Text>
              </View>
            </View>
            <Switch
              value={notificationsEnabled}
              onValueChange={setNotificationsEnabled}
              trackColor={{ false: '#D1D5DB', true: '#2563EB' }}
              thumbColor={notificationsEnabled ? '#FFFFFF' : '#F3F4F6'}
            />
          </View>
        </View>
      </View>

      <View style={styles.section}>
        <Text style={styles.sectionTitle}>💾 Data Management</Text>
        <View style={styles.card}>
          <TouchableOpacity style={styles.actionItem} onPress={handleBackup}>
            <Download size={24} color="#16A34A" />
            <View style={styles.actionText}>
              <Text style={styles.actionTitle}>Backup Data</Text>
              <Text style={styles.actionSubtitle}>Export all data to file</Text>
            </View>
          </TouchableOpacity>

          <View style={styles.divider} />

          <TouchableOpacity style={styles.actionItem} onPress={handleRestore}>
            <Upload size={24} color="#2563EB" />
            <View style={styles.actionText}>
              <Text style={styles.actionTitle}>Restore Data</Text>
              <Text style={styles.actionSubtitle}>Import data from backup file</Text>
            </View>
          </TouchableOpacity>

          <View style={styles.divider} />

          <TouchableOpacity style={styles.actionItem} onPress={handleClearData}>
            <Trash2 size={24} color="#DC2626" />
            <View style={styles.actionText}>
              <Text style={styles.actionTitle}>Clear All Data</Text>
              <Text style={styles.actionSubtitle}>Permanently delete all data</Text>
            </View>
          </TouchableOpacity>
        </View>
      </View>

      <View style={styles.section}>
        <Text style={styles.sectionTitle}>🔒 Privacy & Security</Text>
        <View style={styles.card}>
          <Text style={styles.privacyText}>
            🛡️ Your data is stored locally on your device and never uploaded to any cloud service.
          </Text>
          <Text style={styles.privacyText}>
            📱 All exports happen offline and are shared only when you choose to do so.
          </Text>
          <Text style={styles.privacyText}>
            🔐 No internet connection required for the app to function.
          </Text>
        </View>
      </View>

      <View style={styles.footer}>
        <Text style={styles.footerText}>Class Rep AI v1.0.0</Text>
        <Text style={styles.footerText}>Built for students, by students</Text>
      </View>
    </ScrollView>
  );
}

const createStyles = (theme: any) => StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: theme.background,
  },
  header: {
    padding: 24,
    paddingTop: 60,
  },
  title: {
    fontSize: 24,
    fontWeight: 'bold',
    color: theme.text,
  },
  section: {
    padding: 24,
    paddingTop: 0,
  },
  sectionTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    color: theme.text,
    marginBottom: 16,
  },
  card: {
    backgroundColor: theme.surface,
    borderRadius: 16,
    padding: 20,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 8,
    elevation: 3,
  },
  userInfo: {
    marginBottom: 16,
  },
  userName: {
    fontSize: 18,
    fontWeight: 'bold',
    color: theme.text,
  },
  userRoll: {
    fontSize: 14,
    color: theme.textSecondary,
    marginTop: 4,
  },
  logoutButton: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
    alignSelf: 'flex-start',
  },
  logoutButtonText: {
    color: '#DC2626',
    fontSize: 16,
    fontWeight: '600',
  },
  settingItem: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  settingLeft: {
    flexDirection: 'row',
    alignItems: 'center',
    flex: 1,
  },
  settingText: {
    marginLeft: 16,
    flex: 1,
  },
  settingTitle: {
    fontSize: 16,
    fontWeight: '600',
    color: theme.text,
  },
  settingSubtitle: {
    fontSize: 14,
    color: theme.textSecondary,
    marginTop: 2,
  },
  actionItem: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingVertical: 12,
  },
  actionText: {
    marginLeft: 16,
    flex: 1,
  },
  actionTitle: {
    fontSize: 16,
    fontWeight: '600',
    color: theme.text,
  },
  actionSubtitle: {
    fontSize: 14,
    color: theme.textSecondary,
    marginTop: 2,
  },
  divider: {
    height: 1,
    backgroundColor: theme.border,
    marginVertical: 8,
  },
  privacyText: {
    fontSize: 14,
    color: theme.textSecondary,
    marginBottom: 12,
    lineHeight: 20,
  },
  footer: {
    padding: 24,
    alignItems: 'center',
  },
  footerText: {
    fontSize: 12,
    color: theme.textSecondary,
    textAlign: 'center',
  },
});