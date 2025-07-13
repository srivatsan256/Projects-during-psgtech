import React, { useState, useEffect } from 'react';
import { View, Text, TextInput, TouchableOpacity, StyleSheet, Alert, KeyboardAvoidingView, Platform } from 'react-native';
import { router } from 'expo-router';
import { useData } from '@/context/DataContext';
import { useTheme } from '@/context/ThemeContext';
import { GraduationCap } from 'lucide-react-native';

export default function LoginScreen() {
  const [rollNumber, setRollNumber] = useState('');
  const [studentName, setStudentName] = useState('');
  const { saveUserData, getUserData } = useData();
  const { theme } = useTheme();

  useEffect(() => {
    checkExistingUser();
  }, []);

  const checkExistingUser = async () => {
    const userData = await getUserData();
    if (userData) {
      router.replace('/(tabs)');
    }
  };

  const validateRollNumber = (roll: string): boolean => {
    const rollNum = parseInt(roll);
    if (isNaN(rollNum)) return false;
    
    // Check range 201-269 excluding 248
    if (rollNum >= 201 && rollNum <= 269 && rollNum !== 248) return true;
    
    // Check range 431-436
    if (rollNum >= 431 && rollNum <= 436) return true;
    
    return false;
  };

  const handleLogin = async () => {
    if (!rollNumber.trim() || !studentName.trim()) {
      Alert.alert('Error', 'Please fill in all fields');
      return;
    }

    if (!validateRollNumber(rollNumber)) {
      Alert.alert('Invalid Roll Number', 'Roll number must be between 201-269 (excluding 248) or 431-436');
      return;
    }

    await saveUserData({
      rollNumber: rollNumber.trim(),
      studentName: studentName.trim(),
    });

    router.replace('/(tabs)');
  };

  const styles = createStyles(theme);

  return (
    <KeyboardAvoidingView 
      style={styles.container} 
      behavior={Platform.OS === 'ios' ? 'padding' : 'height'}
    >
      <View style={styles.content}>
        <View style={styles.header}>
          <GraduationCap size={64} color="#2563EB" />
          <Text style={styles.title}>Class Rep AI</Text>
          <Text style={styles.subtitle}>Privacy-focused attendance & assignment manager</Text>
        </View>

        <View style={styles.form}>
          <Text style={styles.label}>Roll Number</Text>
          <TextInput
            style={styles.input}
            value={rollNumber}
            onChangeText={setRollNumber}
            keyboardType="numeric"
            placeholder="Enter your roll number"
            placeholderTextColor={theme.textSecondary}
          />

          <Text style={styles.label}>Student Name</Text>
          <TextInput
            style={styles.input}
            value={studentName}
            onChangeText={setStudentName}
            placeholder="Enter your full name"
            placeholderTextColor={theme.textSecondary}
          />

          <TouchableOpacity style={styles.loginButton} onPress={handleLogin}>
            <Text style={styles.loginButtonText}>Login</Text>
          </TouchableOpacity>
        </View>

        <Text style={styles.footer}>
          Valid roll numbers: 201-269 (excluding 248) or 431-436
        </Text>
      </View>
    </KeyboardAvoidingView>
  );
}

const createStyles = (theme: any) => StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: theme.background,
  },
  content: {
    flex: 1,
    justifyContent: 'center',
    padding: 24,
  },
  header: {
    alignItems: 'center',
    marginBottom: 48,
  },
  title: {
    fontSize: 32,
    fontWeight: 'bold',
    color: theme.text,
    marginTop: 16,
  },
  subtitle: {
    fontSize: 16,
    color: theme.textSecondary,
    textAlign: 'center',
    marginTop: 8,
    lineHeight: 22,
  },
  form: {
    marginBottom: 32,
  },
  label: {
    fontSize: 16,
    fontWeight: '600',
    color: theme.text,
    marginBottom: 8,
    marginTop: 16,
  },
  input: {
    borderWidth: 1,
    borderColor: theme.border,
    borderRadius: 12,
    padding: 16,
    fontSize: 16,
    backgroundColor: theme.surface,
    color: theme.text,
  },
  loginButton: {
    backgroundColor: '#2563EB',
    borderRadius: 12,
    padding: 16,
    alignItems: 'center',
    marginTop: 24,
  },
  loginButtonText: {
    color: '#FFFFFF',
    fontSize: 16,
    fontWeight: 'bold',
  },
  footer: {
    fontSize: 14,
    color: theme.textSecondary,
    textAlign: 'center',
    lineHeight: 20,
  },
});