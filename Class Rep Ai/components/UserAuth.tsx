import React, { useState, useEffect } from 'react';
import { View, Text, TextInput, TouchableOpacity, StyleSheet, Alert, ScrollView } from 'react-native';
import AsyncStorage from '@react-native-async-storage/async-storage';

interface UserAuthProps {
  children: React.ReactNode;
  onAuthSuccess: (user: AuthorizedUser) => void;
}

interface AuthorizedUser {
  id: string;
  username: string;
  email: string;
  role: string;
}

// Hardcoded authorized users (replace with your actual users)
const AUTHORIZED_USERS: AuthorizedUser[] = [
  {
    id: '1',
    username: 'admin',
    email: 'admin@classrepai.com',
    role: 'Admin'
  },
  {
    id: '2',
    username: 'teacher',
    email: 'teacher@classrepai.com',
    role: 'Teacher'
  },
  {
    id: '3',
    username: 'student',
    email: 'student@classrepai.com',
    role: 'Student'
  }
];

const AUTH_STORAGE_KEY = 'user_auth_data';

export default function UserAuth({ children, onAuthSuccess }: UserAuthProps) {
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [isLoading, setIsLoading] = useState(true);
  const [username, setUsername] = useState('');
  const [email, setEmail] = useState('');
  const [currentUser, setCurrentUser] = useState<AuthorizedUser | null>(null);

  useEffect(() => {
    checkStoredAuth();
  }, []);

  const checkStoredAuth = async () => {
    try {
      const storedAuth = await AsyncStorage.getItem(AUTH_STORAGE_KEY);
      if (storedAuth) {
        const authData = JSON.parse(storedAuth);
        const user = AUTHORIZED_USERS.find(u => u.id === authData.userId);
        if (user) {
          setCurrentUser(user);
          setIsAuthenticated(true);
          onAuthSuccess(user);
        }
      }
    } catch (error) {
      console.error('Error checking stored auth:', error);
    } finally {
      setIsLoading(false);
    }
  };

  const handleLogin = async () => {
    const trimmedUsername = username.trim().toLowerCase();
    const trimmedEmail = email.trim().toLowerCase();

    if (!trimmedUsername || !trimmedEmail) {
      Alert.alert('Error', 'Please enter both username and email');
      return;
    }

    const user = AUTHORIZED_USERS.find(
      u => u.username.toLowerCase() === trimmedUsername && 
           u.email.toLowerCase() === trimmedEmail
    );

    if (user) {
      try {
        await AsyncStorage.setItem(AUTH_STORAGE_KEY, JSON.stringify({
          userId: user.id,
          loginTime: new Date().toISOString()
        }));
        
        setCurrentUser(user);
        setIsAuthenticated(true);
        onAuthSuccess(user);
        
        Alert.alert('Success', `Welcome, ${user.role}!`);
      } catch (error) {
        console.error('Error saving auth data:', error);
        Alert.alert('Error', 'Could not save authentication data');
      }
    } else {
      Alert.alert('Access Denied', 'You are not authorized to use this app. Please contact the administrator.');
    }
  };

  const handleLogout = async () => {
    Alert.alert(
      'Logout',
      'Are you sure you want to logout?',
      [
        { text: 'Cancel', style: 'cancel' },
        {
          text: 'Logout',
          onPress: async () => {
            try {
              await AsyncStorage.removeItem(AUTH_STORAGE_KEY);
              setIsAuthenticated(false);
              setCurrentUser(null);
              setUsername('');
              setEmail('');
            } catch (error) {
              console.error('Error during logout:', error);
            }
          }
        }
      ]
    );
  };

  if (isLoading) {
    return (
      <View style={styles.loadingContainer}>
        <Text style={styles.loadingText}>Loading...</Text>
      </View>
    );
  }

  if (!isAuthenticated) {
    return (
      <View style={styles.container}>
        <ScrollView contentContainerStyle={styles.scrollContainer}>
          <View style={styles.authContainer}>
            <Text style={styles.title}>Class Rep AI</Text>
            <Text style={styles.subtitle}>Private Access Only</Text>
            
            <View style={styles.formContainer}>
              <Text style={styles.label}>Username</Text>
              <TextInput
                style={styles.input}
                value={username}
                onChangeText={setUsername}
                placeholder="Enter your username"
                autoCapitalize="none"
                autoCorrect={false}
              />
              
              <Text style={styles.label}>Email</Text>
              <TextInput
                style={styles.input}
                value={email}
                onChangeText={setEmail}
                placeholder="Enter your email"
                autoCapitalize="none"
                autoCorrect={false}
                keyboardType="email-address"
              />
              
              <TouchableOpacity style={styles.loginButton} onPress={handleLogin}>
                <Text style={styles.loginButtonText}>Login</Text>
              </TouchableOpacity>
            </View>
            
            <View style={styles.infoContainer}>
              <Text style={styles.infoTitle}>Authorized Users Only</Text>
              <Text style={styles.infoText}>
                This app is restricted to 3 authorized users. 
                If you need access, please contact the administrator.
              </Text>
            </View>
          </View>
        </ScrollView>
      </View>
    );
  }

  return (
    <View style={styles.container}>
      <View style={styles.userBar}>
        <Text style={styles.welcomeText}>
          Welcome, {currentUser?.role} ({currentUser?.username})
        </Text>
        <TouchableOpacity onPress={handleLogout} style={styles.logoutButton}>
          <Text style={styles.logoutButtonText}>Logout</Text>
        </TouchableOpacity>
      </View>
      {children}
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#f9fafb',
  },
  loadingContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: '#f9fafb',
  },
  loadingText: {
    fontSize: 18,
    color: '#666',
  },
  scrollContainer: {
    flexGrow: 1,
    justifyContent: 'center',
    padding: 20,
  },
  authContainer: {
    backgroundColor: 'white',
    borderRadius: 16,
    padding: 24,
    elevation: 4,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
  },
  title: {
    fontSize: 28,
    fontWeight: 'bold',
    textAlign: 'center',
    marginBottom: 8,
    color: '#1f2937',
  },
  subtitle: {
    fontSize: 16,
    textAlign: 'center',
    color: '#6b7280',
    marginBottom: 32,
  },
  formContainer: {
    marginBottom: 24,
  },
  label: {
    fontSize: 16,
    fontWeight: '600',
    marginBottom: 8,
    color: '#374151',
  },
  input: {
    borderWidth: 1,
    borderColor: '#d1d5db',
    borderRadius: 8,
    paddingHorizontal: 16,
    paddingVertical: 12,
    fontSize: 16,
    marginBottom: 16,
    backgroundColor: '#f9fafb',
  },
  loginButton: {
    backgroundColor: '#2563eb',
    paddingVertical: 14,
    paddingHorizontal: 24,
    borderRadius: 8,
    alignItems: 'center',
    marginTop: 8,
  },
  loginButtonText: {
    color: 'white',
    fontSize: 16,
    fontWeight: '600',
  },
  infoContainer: {
    padding: 16,
    backgroundColor: '#fef3c7',
    borderRadius: 8,
    borderWidth: 1,
    borderColor: '#f59e0b',
  },
  infoTitle: {
    fontSize: 14,
    fontWeight: '600',
    color: '#92400e',
    marginBottom: 4,
  },
  infoText: {
    fontSize: 12,
    color: '#92400e',
    lineHeight: 16,
  },
  userBar: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    backgroundColor: '#2563eb',
    paddingHorizontal: 16,
    paddingVertical: 12,
    paddingTop: 50,
  },
  welcomeText: {
    color: 'white',
    fontSize: 14,
    fontWeight: '500',
  },
  logoutButton: {
    backgroundColor: 'rgba(255, 255, 255, 0.2)',
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: 6,
  },
  logoutButtonText: {
    color: 'white',
    fontSize: 12,
    fontWeight: '600',
  },
});