import React, { useEffect, useState } from 'react';
import { View, Text, Modal, TouchableOpacity, StyleSheet, Alert } from 'react-native';
import * as Updates from 'expo-updates';

interface SimpleOTACheckerProps {
  children: React.ReactNode;
}

export default function SimpleOTAChecker({ children }: SimpleOTACheckerProps) {
  const [isUpdateAvailable, setIsUpdateAvailable] = useState(false);
  const [isUpdating, setIsUpdating] = useState(false);

  useEffect(() => {
    checkForUpdates();
  }, []);

  const checkForUpdates = async () => {
    try {
      // Skip update check in development
      if (__DEV__) {
        console.log('Skipping update check in development');
        return;
      }

      console.log('Checking for updates...');
      const update = await Updates.checkForUpdateAsync();
      
      if (update.isAvailable) {
        console.log('Update available');
        setIsUpdateAvailable(true);
      } else {
        console.log('App is up to date');
      }
    } catch (error) {
      console.error('Error checking for updates:', error);
      // Don't show error alert to user, just log it
    }
  };

  const downloadAndInstallUpdate = async () => {
    try {
      setIsUpdating(true);
      console.log('Downloading update...');
      
      const downloadResult = await Updates.fetchUpdateAsync();
      
      if (downloadResult.isNew) {
        console.log('Update downloaded, restarting app...');
        // Restart the app to apply update
        await Updates.reloadAsync();
      } else {
        console.log('No new update found');
        setIsUpdating(false);
        setIsUpdateAvailable(false);
      }
    } catch (error) {
      console.error('Error downloading update:', error);
      setIsUpdating(false);
      Alert.alert('Update Failed', 'Could not download update. Please try again later.');
    }
  };

  const skipUpdate = () => {
    setIsUpdateAvailable(false);
  };

  // Show update modal if update is available
  if (isUpdateAvailable) {
    return (
      <View style={styles.container}>
        <Modal
          visible={true}
          transparent={true}
          animationType="fade"
        >
          <View style={styles.modalOverlay}>
            <View style={styles.modalContent}>
              <Text style={styles.title}>App Update Available</Text>
              <Text style={styles.message}>
                A new version of the app is available. Would you like to update now?
              </Text>
              
              {isUpdating && (
                <Text style={styles.updating}>Updating...</Text>
              )}
              
              <View style={styles.buttonContainer}>
                <TouchableOpacity
                  style={[styles.button, styles.skipButton]}
                  onPress={skipUpdate}
                  disabled={isUpdating}
                >
                  <Text style={styles.skipButtonText}>Skip</Text>
                </TouchableOpacity>
                
                <TouchableOpacity
                  style={[styles.button, styles.updateButton]}
                  onPress={downloadAndInstallUpdate}
                  disabled={isUpdating}
                >
                  <Text style={styles.updateButtonText}>
                    {isUpdating ? 'Updating...' : 'Update Now'}
                  </Text>
                </TouchableOpacity>
              </View>
            </View>
          </View>
        </Modal>
        {children}
      </View>
    );
  }

  return <>{children}</>;
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
  modalOverlay: {
    flex: 1,
    backgroundColor: 'rgba(0, 0, 0, 0.5)',
    justifyContent: 'center',
    alignItems: 'center',
  },
  modalContent: {
    backgroundColor: 'white',
    borderRadius: 16,
    padding: 24,
    margin: 20,
    minWidth: 300,
  },
  title: {
    fontSize: 20,
    fontWeight: 'bold',
    marginBottom: 12,
    textAlign: 'center',
    color: '#333',
  },
  message: {
    fontSize: 16,
    marginBottom: 24,
    textAlign: 'center',
    color: '#666',
    lineHeight: 22,
  },
  updating: {
    fontSize: 14,
    color: '#007AFF',
    textAlign: 'center',
    marginBottom: 16,
  },
  buttonContainer: {
    flexDirection: 'row',
    gap: 12,
  },
  button: {
    flex: 1,
    paddingVertical: 12,
    paddingHorizontal: 24,
    borderRadius: 8,
    alignItems: 'center',
  },
  skipButton: {
    backgroundColor: '#f3f4f6',
    borderWidth: 1,
    borderColor: '#d1d5db',
  },
  updateButton: {
    backgroundColor: '#007AFF',
  },
  skipButtonText: {
    color: '#374151',
    fontSize: 16,
    fontWeight: '600',
  },
  updateButtonText: {
    color: 'white',
    fontSize: 16,
    fontWeight: '600',
  },
});