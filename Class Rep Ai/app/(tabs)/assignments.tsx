import React, { useState, useEffect } from 'react';
import { View, Text, TouchableOpacity, StyleSheet, ScrollView, TextInput, Modal, Alert, Linking } from 'react-native';
import { useData } from '@/context/DataContext';
import { useTheme } from '@/context/ThemeContext';
import { Plus, Calendar, List, CreditCard as Edit, Trash2, ExternalLink, Bell, Share, MessageCircle } from 'lucide-react-native';
import DateTimePicker from '@react-native-community/datetimepicker';
import * as Notifications from 'expo-notifications';

// Configure notification handler
Notifications.setNotificationHandler({
  handleNotification: async () => ({
    shouldShowAlert: true,
    shouldPlaySound: true,
    shouldSetBadge: false,
    shouldShowBanner: true,
    shouldShowList: true,
  }),
});

export default function AssignmentsScreen() {
  const { getAssignments, saveAssignment, deleteAssignment, exportAssignments } = useData();
  const { theme } = useTheme();
  const [assignments, setAssignments] = useState<any[]>([]);
  const [showModal, setShowModal] = useState(false);
  const [viewMode, setViewMode] = useState<'list' | 'calendar'>('list');
  const [editingAssignment, setEditingAssignment] = useState<any>(null);
  const [showDatePicker, setShowDatePicker] = useState(false);
  const [notificationEnabled, setNotificationEnabled] = useState(false);

  const [formData, setFormData] = useState({
    subject: '',
    requirement: '',
    details: '',
    dueDate: new Date(),
    uploadType: '',
    uploadLink: '',
  });

  useEffect(() => {
    loadAssignments();
    setupNotifications();
  }, []);

  const loadAssignments = async () => {
    const assignmentList = await getAssignments();
    setAssignments(assignmentList);
  };

  const setupNotifications = async () => {
    try {
      // Request permissions
      const { status } = await Notifications.requestPermissionsAsync();
      if (status !== 'granted') {
        Alert.alert('Permission denied', 'Please enable notifications to receive daily assignment reminders.');
        return;
      }

      setNotificationEnabled(true);
      await scheduleDailyNotification();
    } catch (error) {
      console.error('Error setting up notifications:', error);
    }
  };

  const scheduleDailyNotification = async () => {
    try {
      // Cancel existing notifications
      await Notifications.cancelAllScheduledNotificationsAsync();

      // Schedule daily notification at 17:45
      await Notifications.scheduleNotificationAsync({
        content: {
          title: 'Daily Assignment Reminder',
          body: 'Time to share today\'s assignment details with your classmates!',
          sound: 'default',
          data: { action: 'share_assignments' },
        },
        trigger: {
          hour: 17,
          minute: 45,
          repeats: true,
        } as any,
      });

      console.log('Daily notification scheduled for 17:45');
    } catch (error) {
      console.error('Error scheduling notification:', error);
    }
  };

  const resetForm = () => {
    setFormData({
      subject: '',
      requirement: '',
      details: '',
      dueDate: new Date(),
      uploadType: '',
      uploadLink: '',
    });
    setEditingAssignment(null);
  };

  const handleSave = async () => {
    if (!formData.subject.trim() || !formData.requirement.trim()) {
      Alert.alert('Error', 'Please fill in subject and requirement fields.');
      return;
    }

    const assignment = {
      id: editingAssignment?.id || Date.now().toString(),
      ...formData,
      dueDate: formData.dueDate.toISOString().split('T')[0],
      createdAt: editingAssignment?.createdAt || new Date().toISOString(),
    };

    await saveAssignment(assignment);
    await loadAssignments();
    setShowModal(false);
    resetForm();
  };

  const handleEdit = (assignment: any) => {
    setEditingAssignment(assignment);
    setFormData({
      ...assignment,
      dueDate: new Date(assignment.dueDate),
    });
    setShowModal(true);
  };

  const handleDelete = async (id: string) => {
    Alert.alert(
      'Delete Assignment',
      'Are you sure you want to delete this assignment?',
      [
        { text: 'Cancel', style: 'cancel' },
        { 
          text: 'Delete', 
          style: 'destructive',
          onPress: async () => {
            await deleteAssignment(id);
            await loadAssignments();
          }
        }
      ]
    );
  };

  const handleExport = async () => {
    await exportAssignments();
  };

  const shareAssignmentToWhatsApp = async (assignment: any) => {
    const message = `📚 *Assignment Update*\n\n` +
      `📘 **Subject:** ${assignment.subject}\n` +
      `📝 **Requirement:** ${assignment.requirement}\n` +
      `📅 **Due Date:** ${assignment.dueDate}\n` +
      `📋 **Details:** ${assignment.details}\n` +
      `${assignment.uploadType ? `📎 **Upload:** ${assignment.uploadType}\n` : ''}` +
      `${assignment.uploadLink ? `🔗 **Link:** ${assignment.uploadLink}\n` : ''}\n` +
      `⏰ *Shared at ${new Date().toLocaleTimeString()}*`;

    const whatsappUrl = `whatsapp://send?text=${encodeURIComponent(message)}`;
    
    try {
      const supported = await Linking.canOpenURL(whatsappUrl);
      if (supported) {
        await Linking.openURL(whatsappUrl);
      } else {
        Alert.alert('WhatsApp not found', 'Please install WhatsApp to share assignments.');
      }
    } catch (error) {
      console.error('Error sharing to WhatsApp:', error);
      Alert.alert('Error', 'Could not share to WhatsApp. Please try again.');
    }
  };

  const shareAllAssignmentsToWhatsApp = async () => {
    if (assignments.length === 0) {
      Alert.alert('No Assignments', 'There are no assignments to share.');
      return;
    }

    const todayAssignments = assignments.filter(a => {
      const today = new Date().toISOString().split('T')[0];
      return a.dueDate === today || new Date(a.dueDate) > new Date();
    });

    if (todayAssignments.length === 0) {
      Alert.alert('No Upcoming Assignments', 'There are no upcoming assignments to share.');
      return;
    }

    let message = `📚 *Daily Assignment Summary*\n`;
    message += `📅 ${new Date().toLocaleDateString()}\n\n`;

    todayAssignments.forEach((assignment, index) => {
      message += `${index + 1}. 📘 **${assignment.subject}**\n`;
      message += `   📝 ${assignment.requirement}\n`;
      message += `   📅 Due: ${assignment.dueDate}\n`;
      if (assignment.uploadType) {
        message += `   📎 ${assignment.uploadType}\n`;
      }
      message += `\n`;
    });

    message += `⏰ *Shared at ${new Date().toLocaleTimeString()}*`;

    const whatsappUrl = `whatsapp://send?text=${encodeURIComponent(message)}`;
    
    try {
      const supported = await Linking.canOpenURL(whatsappUrl);
      if (supported) {
        await Linking.openURL(whatsappUrl);
      } else {
        Alert.alert('WhatsApp not found', 'Please install WhatsApp to share assignments.');
      }
    } catch (error) {
      console.error('Error sharing to WhatsApp:', error);
      Alert.alert('Error', 'Could not share to WhatsApp. Please try again.');
    }
  };

  const sendAssignmentNotification = (assignment: any) => {
    Alert.alert(
      'Send Assignment Notification',
      `Share "${assignment.subject}" assignment details via WhatsApp?`,
      [
        { text: 'Cancel', style: 'cancel' },
        { 
          text: 'Share', 
          onPress: () => shareAssignmentToWhatsApp(assignment)
        }
      ]
    );
  };

  const groupAssignmentsByDate = () => {
    const grouped = assignments.reduce((acc: any, assignment) => {
      const date = assignment.dueDate;
      if (!acc[date]) acc[date] = [];
      acc[date].push(assignment);
      return acc;
    }, {});

    return Object.keys(grouped)
      .sort()
      .map(date => ({
        date,
        assignments: grouped[date],
      }));
  };

  const styles = createStyles(theme);

  return (
    <View style={styles.container}>
      <View style={styles.header}>
        <Text style={styles.title}>Assignments</Text>
        <View style={styles.headerButtons}>
          <TouchableOpacity
            style={[styles.viewButton, viewMode === 'list' && styles.activeViewButton]}
            onPress={() => setViewMode('list')}
          >
            <List size={20} color={viewMode === 'list' ? '#FFFFFF' : theme.text} />
          </TouchableOpacity>
          <TouchableOpacity
            style={[styles.viewButton, viewMode === 'calendar' && styles.activeViewButton]}
            onPress={() => setViewMode('calendar')}
          >
            <Calendar size={20} color={viewMode === 'calendar' ? '#FFFFFF' : theme.text} />
          </TouchableOpacity>
          <TouchableOpacity
            style={styles.addButton}
            onPress={() => setShowModal(true)}
          >
            <Plus size={20} color="#FFFFFF" />
          </TouchableOpacity>
        </View>
      </View>

      {notificationEnabled && (
        <View style={styles.notificationBanner}>
          <Bell size={16} color="#16A34A" />
          <Text style={styles.notificationText}>
            Daily reminders set for 17:45 to share assignments
          </Text>
        </View>
      )}

      <ScrollView style={styles.content}>
        {viewMode === 'list' ? (
          assignments.map((assignment) => (
            <View key={assignment.id} style={styles.assignmentCard}>
              <View style={styles.assignmentHeader}>
                <Text style={styles.subject}>📘 {assignment.subject}</Text>
                <View style={styles.actions}>
                  <TouchableOpacity onPress={() => shareAssignmentToWhatsApp(assignment)}>
                    <MessageCircle size={20} color="#25D366" />
                  </TouchableOpacity>
                  <TouchableOpacity onPress={() => sendAssignmentNotification(assignment)}>
                    <Bell size={20} color="#16A34A" />
                  </TouchableOpacity>
                  <TouchableOpacity onPress={() => handleEdit(assignment)}>
                    <Edit size={20} color="#2563EB" />
                  </TouchableOpacity>
                  <TouchableOpacity onPress={() => handleDelete(assignment.id)}>
                    <Trash2 size={20} color="#DC2626" />
                  </TouchableOpacity>
                </View>
              </View>
              <Text style={styles.requirement}>{assignment.requirement}</Text>
              <Text style={styles.details}>{assignment.details}</Text>
              <Text style={styles.dueDate}>📅 Due: {assignment.dueDate}</Text>
              {assignment.uploadType && (
                <View style={styles.uploadInfo}>
                  <Text style={styles.uploadType}>📎 {assignment.uploadType}</Text>
                  {assignment.uploadLink && (
                    <TouchableOpacity 
                      style={styles.linkButton}
                      onPress={() => Linking.openURL(assignment.uploadLink)}
                    >
                      <ExternalLink size={16} color="#2563EB" />
                      <Text style={styles.linkText}>Open Link</Text>
                    </TouchableOpacity>
                  )}
                </View>
              )}
            </View>
          ))
        ) : (
          groupAssignmentsByDate().map((group) => (
            <View key={group.date} style={styles.dateGroup}>
              <Text style={styles.dateHeader}>{group.date}</Text>
              {group.assignments.map((assignment: any) => (
                <View key={assignment.id} style={styles.calendarAssignment}>
                  <Text style={styles.calendarSubject}>{assignment.subject}</Text>
                  <Text style={styles.calendarRequirement}>{assignment.requirement}</Text>
                </View>
              ))}
            </View>
          ))
        )}
      </ScrollView>

      <View style={styles.bottomButtons}>
        <TouchableOpacity style={styles.exportButton} onPress={handleExport}>
          <Share size={20} color="#FFFFFF" />
          <Text style={styles.exportButtonText}>Export All</Text>
        </TouchableOpacity>
        
        <TouchableOpacity style={styles.whatsappButton} onPress={shareAllAssignmentsToWhatsApp}>
          <MessageCircle size={20} color="#FFFFFF" />
          <Text style={styles.whatsappButtonText}>Share to WhatsApp</Text>
        </TouchableOpacity>
      </View>

      <Modal visible={showModal} transparent animationType="slide">
        <View style={styles.modalOverlay}>
          <View style={styles.modalContent}>
            <Text style={styles.modalTitle}>
              {editingAssignment ? 'Edit Assignment' : 'Add Assignment'}
            </Text>

            <ScrollView>
              <TextInput
                style={styles.input}
                value={formData.subject}
                onChangeText={(text) => setFormData(prev => ({ ...prev, subject: text }))}
                placeholder="Subject"
                placeholderTextColor={theme.textSecondary}
              />

              <TextInput
                style={styles.input}
                value={formData.requirement}
                onChangeText={(text) => setFormData(prev => ({ ...prev, requirement: text }))}
                placeholder="Requirement"
                placeholderTextColor={theme.textSecondary}
              />

              <TextInput
                style={[styles.input, styles.textArea]}
                value={formData.details}
                onChangeText={(text) => setFormData(prev => ({ ...prev, details: text }))}
                placeholder="Assignment Details"
                placeholderTextColor={theme.textSecondary}
                multiline
                numberOfLines={4}
              />

              <TouchableOpacity
                style={styles.dateButton}
                onPress={() => setShowDatePicker(true)}
              >
                <Text style={styles.dateButtonText}>
                  Due Date: {formData.dueDate.toISOString().split('T')[0]}
                </Text>
              </TouchableOpacity>

              <View style={styles.pickerContainer}>
                <Text style={styles.label}>Upload Type:</Text>
                <View style={styles.uploadTypes}>
                  {['Google Classroom', 'Google Drive'].map((type) => (
                    <TouchableOpacity
                      key={type}
                      style={[
                        styles.uploadTypeButton,
                        formData.uploadType === type && styles.selectedUploadType
                      ]}
                      onPress={() => setFormData(prev => ({ ...prev, uploadType: type }))}
                    >
                      <Text style={[
                        styles.uploadTypeText,
                        formData.uploadType === type && styles.selectedUploadTypeText
                      ]}>
                        {type}
                      </Text>
                    </TouchableOpacity>
                  ))}
                </View>
              </View>

              {formData.uploadType && (
                <TextInput
                  style={styles.input}
                  value={formData.uploadLink}
                  onChangeText={(text) => setFormData(prev => ({ ...prev, uploadLink: text }))}
                  placeholder="Upload Link"
                  placeholderTextColor={theme.textSecondary}
                />
              )}
            </ScrollView>

            <View style={styles.modalButtons}>
              <TouchableOpacity
                style={[styles.modalButton, styles.cancelButton]}
                onPress={() => {
                  setShowModal(false);
                  resetForm();
                }}
              >
                <Text style={styles.cancelButtonText}>Cancel</Text>
              </TouchableOpacity>
              <TouchableOpacity
                style={[styles.modalButton, styles.saveButton]}
                onPress={handleSave}
              >
                <Text style={styles.saveButtonText}>Save</Text>
              </TouchableOpacity>
            </View>
          </View>
        </View>

        {showDatePicker && (
          <DateTimePicker
            value={formData.dueDate}
            mode="date"
            display="default"
            onChange={(event, date) => {
              setShowDatePicker(false);
              if (date) {
                setFormData(prev => ({ ...prev, dueDate: date }));
              }
            }}
          />
        )}
      </Modal>
    </View>
  );
}

const createStyles = (theme: any) => StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: theme.background,
  },
  header: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    padding: 24,
    paddingTop: 60,
  },
  title: {
    fontSize: 24,
    fontWeight: 'bold',
    color: theme.text,
  },
  headerButtons: {
    flexDirection: 'row',
    gap: 8,
  },
  viewButton: {
    padding: 8,
    borderRadius: 8,
    borderWidth: 1,
    borderColor: theme.border,
  },
  activeViewButton: {
    backgroundColor: '#2563EB',
  },
  addButton: {
    backgroundColor: '#16A34A',
    padding: 8,
    borderRadius: 8,
  },
  notificationBanner: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: '#E0F2F7',
    padding: 12,
    margin: 12,
    borderRadius: 8,
    borderLeftWidth: 5,
    borderLeftColor: '#16A34A',
  },
  notificationText: {
    marginLeft: 8,
    fontSize: 14,
    color: '#16A34A',
    fontWeight: '600',
  },
  content: {
    flex: 1,
    paddingHorizontal: 24,
  },
  assignmentCard: {
    backgroundColor: theme.surface,
    padding: 16,
    borderRadius: 12,
    marginBottom: 12,
  },
  assignmentHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 8,
  },
  subject: {
    fontSize: 16,
    fontWeight: 'bold',
    color: '#2563EB',
    flex: 1,
  },
  actions: {
    flexDirection: 'row',
    gap: 12,
  },
  requirement: {
    fontSize: 14,
    color: theme.text,
    marginBottom: 8,
  },
  details: {
    fontSize: 12,
    color: theme.textSecondary,
    marginBottom: 8,
  },
  dueDate: {
    fontSize: 12,
    color: theme.textSecondary,
    marginBottom: 8,
  },
  uploadInfo: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  uploadType: {
    fontSize: 12,
    color: '#EA580C',
  },
  linkButton: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 4,
  },
  linkText: {
    fontSize: 12,
    color: '#2563EB',
  },
  dateGroup: {
    marginBottom: 16,
  },
  dateHeader: {
    fontSize: 18,
    fontWeight: 'bold',
    color: theme.text,
    marginBottom: 8,
  },
  calendarAssignment: {
    backgroundColor: theme.surface,
    padding: 12,
    borderRadius: 8,
    marginBottom: 8,
  },
  calendarSubject: {
    fontSize: 14,
    fontWeight: 'bold',
    color: '#2563EB',
  },
  calendarRequirement: {
    fontSize: 12,
    color: theme.text,
    marginTop: 4,
  },
  bottomButtons: {
    flexDirection: 'row',
    justifyContent: 'space-around',
    margin: 24,
    marginTop: 0,
    gap: 12,
  },
  exportButton: {
    backgroundColor: '#16A34A',
    padding: 16,
    borderRadius: 12,
    alignItems: 'center',
    flexDirection: 'row',
    gap: 8,
    flex: 1,
  },
  exportButtonText: {
    color: '#FFFFFF',
    fontSize: 16,
    fontWeight: 'bold',
  },
  whatsappButton: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: '#25D366',
    padding: 16,
    borderRadius: 12,
    gap: 8,
    flex: 1,
  },
  whatsappButtonText: {
    color: '#FFFFFF',
    fontSize: 16,
    fontWeight: 'bold',
  },
  modalOverlay: {
    flex: 1,
    backgroundColor: 'rgba(0,0,0,0.5)',
    justifyContent: 'center',
    alignItems: 'center',
  },
  modalContent: {
    backgroundColor: theme.surface,
    padding: 24,
    borderRadius: 16,
    width: '90%',
    maxWidth: 500,
    maxHeight: '80%',
  },
  modalTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    color: theme.text,
    marginBottom: 16,
  },
  input: {
    borderWidth: 1,
    borderColor: theme.border,
    borderRadius: 8,
    padding: 12,
    marginBottom: 12,
    color: theme.text,
    backgroundColor: theme.background,
  },
  textArea: {
    height: 80,
    textAlignVertical: 'top',
  },
  dateButton: {
    borderWidth: 1,
    borderColor: theme.border,
    borderRadius: 8,
    padding: 12,
    marginBottom: 12,
    backgroundColor: theme.background,
  },
  dateButtonText: {
    color: theme.text,
  },
  label: {
    fontSize: 14,
    fontWeight: '600',
    color: theme.text,
    marginBottom: 8,
  },
  pickerContainer: {
    marginBottom: 12,
  },
  uploadTypes: {
    flexDirection: 'row',
    gap: 8,
  },
  uploadTypeButton: {
    paddingHorizontal: 16,
    paddingVertical: 8,
    borderRadius: 20,
    borderWidth: 1,
    borderColor: theme.border,
  },
  selectedUploadType: {
    backgroundColor: '#2563EB',
    borderColor: '#2563EB',
  },
  uploadTypeText: {
    color: theme.text,
    fontSize: 12,
  },
  selectedUploadTypeText: {
    color: '#FFFFFF',
  },
  modalButtons: {
    flexDirection: 'row',
    gap: 12,
    marginTop: 16,
  },
  modalButton: {
    flex: 1,
    padding: 12,
    borderRadius: 8,
    alignItems: 'center',
  },
  cancelButton: {
    backgroundColor: theme.border,
  },
  saveButton: {
    backgroundColor: '#16A34A',
  },
  cancelButtonText: {
    color: theme.text,
    fontWeight: '600',
  },
  saveButtonText: {
    color: '#FFFFFF',
    fontWeight: '600',
  },
});