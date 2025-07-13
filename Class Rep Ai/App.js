import React from 'react';
import { View, Text, StyleSheet, SafeAreaView } from 'react-native';
import SimpleOTAChecker from './components/SimpleOTAChecker';

export default function App() {
  return (
    <SimpleOTAChecker>
      <SafeAreaView style={styles.container}>
        <View style={styles.content}>
          <Text style={styles.title}>Class Rep AI</Text>
          <Text style={styles.subtitle}>OTA Updates Enabled</Text>
          
          {/* Replace this with your actual app content */}
          <View style={styles.card}>
            <Text style={styles.cardTitle}>Welcome!</Text>
            <Text style={styles.cardText}>
              This app will automatically check for updates when you open it.
            </Text>
          </View>
          
          <View style={styles.statusContainer}>
            <Text style={styles.statusText}>
              ✅ OTA Updates: Active
            </Text>
            <Text style={styles.statusText}>
              📱 Version: 1.0.0
            </Text>
          </View>
        </View>
      </SafeAreaView>
    </SimpleOTAChecker>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#f5f5f5',
  },
  content: {
    flex: 1,
    padding: 20,
    justifyContent: 'center',
  },
  title: {
    fontSize: 32,
    fontWeight: 'bold',
    textAlign: 'center',
    color: '#333',
    marginBottom: 8,
  },
  subtitle: {
    fontSize: 16,
    textAlign: 'center',
    color: '#666',
    marginBottom: 40,
  },
  card: {
    backgroundColor: 'white',
    borderRadius: 12,
    padding: 24,
    marginBottom: 32,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
  },
  cardTitle: {
    fontSize: 20,
    fontWeight: '600',
    color: '#333',
    marginBottom: 12,
  },
  cardText: {
    fontSize: 16,
    color: '#666',
    lineHeight: 24,
  },
  statusContainer: {
    backgroundColor: '#e8f5e8',
    borderRadius: 8,
    padding: 16,
    alignItems: 'center',
  },
  statusText: {
    fontSize: 14,
    color: '#2d5a2d',
    marginBottom: 4,
  },
});