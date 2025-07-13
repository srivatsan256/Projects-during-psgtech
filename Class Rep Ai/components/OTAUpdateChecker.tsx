import React, { useEffect, useState } from 'react';
import { View, Text, Modal, TouchableOpacity, StyleSheet, Alert } from 'react-native';
import * as Updates from 'expo-updates';

interface OTAUpdateCheckerProps {
  children: React.ReactNode;
}

export default function OTAUpdateChecker({ children }: OTAUpdateCheckerProps) {
  const [isUpdateAvailable, setIsUpdateAvailable] = useState(false);
  const [isUpdating, setIsUpdating] = useState(false);
  const [updateProgress, setUpdateProgress] = useState(0);

  useEffect(() => {
    checkForUpdates();
  }, []);

  const checkForUpdates = async () => {
    try {
      if (!Updates.isEnabled) {
        console.log('Updates are not enabled');
        return;
      }

      console.log('Checking for updates...');
      const update = await Updates.checkForUpdateAsync();
      
      if (update.isAvailable) {
        console.log('Update available');
        setIsUpdateAvailable(true);
      } else {
        console.log('No updates available');
      }
    } catch (error) {
      console.error('Error checking for updates:', error);
      Alert.alert('Update Check Failed', 'Could not check for updates. Please try again later.');
    }
  };

  const downloadAndInstallUpdate = async () => {
    try {
      setIsUpdating(true);
      console.log('Downloading update...');
      
      const downloadResult = await Updates.fetchUpdateAsync();
      
      if (downloadResult.isNew) {
        console.log('Update downloaded, restarting app...');
        Alert.alert(
          'Update Downloaded',
          'The app will restart to apply the update.',
          [
            {
              text: 'OK',
              onPress: () => Updates.reloadAsync(),
            },
          ]
        );
      }
    } catch (error) {
      console.error('Error downloading update:', error);
      Alert.alert('Update Failed', 'Could not download the update. Please try again later.');
      setIsUpdating(false);
    }
  };

  const skipUpdate = () => {
    setIsUpdateAvailable(false);
  };

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
              <Text style={styles.title}>Update Available</Text>
              <Text style={styles.message}>
                A new version of the app is available. Would you like to download and install it?
              </Text>
              
              {isUpdating && (
                <View style={styles.progressContainer}>
                  <Text style={styles.progressText}>Downloading update...</Text>
                  <View style={styles.progressBar}>
                    <View 
                      style={[styles.progressFill, { width: `${updateProgress}%` }]} 
                    />
                  </View>
                </View>
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
  progressContainer: {
    marginBottom: 24,
  },
  progressText: {
    fontSize: 14,
    color: '#666',
    textAlign: 'center',
    marginBottom: 8,
  },
  progressBar: {
    height: 4,
    backgroundColor: '#e0e0e0',
    borderRadius: 2,
    overflow: 'hidden',
  },
  progressFill: {
    height: '100%',
    backgroundColor: '#2563EB',
    borderRadius: 2,
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
    backgroundColor: '#2563EB',
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